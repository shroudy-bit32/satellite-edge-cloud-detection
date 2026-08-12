"""U-Net ile piksel bazli bulut segmentasyonu egitimi (genisletilmis hedef).

Kullanim:
    python -m src.train_seg
    python -m src.train_seg --epochs 40 --tag unet_v2

Siniflandirici hattiyla ayni kisitlar gecerlidir: model birkac MB'i asmamali,
CPU cikarim suresi hedefin altinda kalmalidir. Bu yuzden model MobileNet
tabanli hafif U-Net'tir ve ayni ONNX/INT8 hattina girer.
"""

import argparse
import json
import time

import numpy as np
import torch
from tqdm import tqdm

from src import config
from src.dataset import build_seg_loaders
from src.losses import DiceBCELoss, accumulate_metrics, segmentation_metrics
from src.train import set_seed
from src.unet import build_unet, mask_to_decision


@torch.no_grad()
def evaluate_seg(model, loader, device, threshold: float = 0.5):
    model.eval()
    counts, decisions, decision_targets = [], [], []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.autocast("cuda", enabled=device.type == "cuda"):
            logits = model(x)
        logits = logits.float()

        counts.append(segmentation_metrics(logits, y, threshold)["_counts"])

        # Maskeden turetilen goruntu-seviyesi karar: fayda analizine koprii
        decisions.append(mask_to_decision(logits, threshold).cpu())
        true_fraction = y.mean(dim=(-2, -1)).squeeze(1)
        decision_targets.append((true_fraction >= config.CLOUD_PIXEL_THRESHOLD).float().cpu())

    metrics = accumulate_metrics(counts)
    decisions = torch.cat(decisions).numpy()
    decision_targets = torch.cat(decision_targets).numpy()
    metrics["image_level_accuracy"] = float((decisions == decision_targets).mean())
    return metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--encoder", default=config.MODEL_NAME)
    p.add_argument("--epochs", type=int, default=config.EPOCHS)
    p.add_argument("--batch-size", type=int, default=16, help="segmentasyon daha cok bellek yer")
    p.add_argument("--lr", type=float, default=config.LR)
    p.add_argument("--weight-decay", type=float, default=config.WEIGHT_DECAY)
    p.add_argument("--workers", type=int, default=config.NUM_WORKERS)
    p.add_argument("--bce-weight", type=float, default=0.5)
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--tag", default="unet")
    p.add_argument("--accum-steps", type=int, default=1,
                   help="gradyan biriktirme; 256x256 segmentasyonda 12 GB VRAM "
                        "kucuk batch'e zorluyor, efektif batch'i bununla buyut")
    p.add_argument("--clip-grad", type=float, default=0.0,
                   help="gradyan norm siniri (0 = kapali); Dice kaybinin "
                        "egitim basindaki sicramalarini bastirir")
    args = p.parse_args()

    if args.accum_steps < 1:
        raise SystemExit("--accum-steps en az 1 olmali")
    print(f"efektif batch: {args.batch_size * args.accum_steps} "
          f"({args.batch_size} x {args.accum_steps})")

    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"cihaz: {device}")

    train_loader, val_loader, test_loader = build_seg_loaders(args.batch_size, args.workers)
    print(f"train {len(train_loader.dataset)} | val {len(val_loader.dataset)} | test {len(test_loader.dataset)}")

    model = build_unet(args.encoder, pretrained=not args.no_pretrained).to(device)
    params = sum(p_.numel() for p_ in model.parameters())
    print(f"parametre {params:,} | FP32 {params * 4 / 1024**2:.2f} MB")

    pos_weight = train_loader.dataset.pos_weight()
    print(f"bulut pikseli pos_weight {pos_weight.item():.2f}")

    criterion = DiceBCELoss(bce_weight=args.bce_weight, pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(config.TB_LOGS / args.tag)
    ckpt_path = config.CHECKPOINTS / f"{args.tag}_best.pt"
    best_iou = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running, n = 0.0, 0
        t0 = time.perf_counter()

        optimizer.zero_grad(set_to_none=True)
        num_batches = len(train_loader)

        for step, (x, y) in enumerate(
            tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}", leave=False), start=1
        ):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with torch.autocast("cuda", enabled=device.type == "cuda"):
                loss = criterion(model(x).float(), y)
            scaler.scale(loss / args.accum_steps).backward()

            if step % args.accum_steps == 0 or step == num_batches:
                if args.clip_grad > 0:
                    scaler.unscale_(optimizer)  # AMP: clipping oncesi zorunlu
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            running += loss.item() * x.size(0)
            n += x.size(0)
        scheduler.step()

        train_loss = running / n
        val = evaluate_seg(model, val_loader, device)
        dt = time.perf_counter() - t0

        print(f"epoch {epoch:3d} | loss {train_loss:.4f} | val IoU {val['iou']:.4f} "
              f"dice {val['dice']:.4f} rec {val['recall']:.4f} "
              f"img-acc {val['image_level_accuracy']:.4f} | {dt:.1f}s")

        writer.add_scalar("loss/train", train_loss, epoch)
        for k, v in val.items():
            writer.add_scalar(f"val/{k}", v, epoch)

        if val["iou"] > best_iou:
            best_iou = val["iou"]
            torch.save({"model": model.state_dict(), "task": "segmentation",
                        "model_name": args.encoder, "in_channels": config.IN_CHANNELS,
                        "epoch": epoch, "val_iou": best_iou}, ckpt_path)

    model.load_state_dict(torch.load(ckpt_path, map_location=device)["model"])
    test = evaluate_seg(model, test_loader, device)

    summary = {"tag": args.tag, "task": "segmentation", "encoder": args.encoder,
               "parameters": params, "best_val_iou": best_iou, "test": test,
               "hyperparameters": {"lr": args.lr, "weight_decay": args.weight_decay,
                                   "batch_size": args.batch_size,
                                   "bce_weight": args.bce_weight,
                                   "bands": config.BANDS},
               "optimizations": {"accum_steps": args.accum_steps,
                                 "effective_batch": args.batch_size * args.accum_steps,
                                 "clip_grad": args.clip_grad, "bce_weight": args.bce_weight},
               "checkpoint": str(ckpt_path)}
    out = config.REPORTS / f"{args.tag}_train_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\ntest sonuclari:")
    for k, v in test.items():
        print(f"  {k:<22} {v:.4f}")
    print(f"\nozet: {out}")
    writer.close()


if __name__ == "__main__":
    main()
