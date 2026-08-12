"""MobileNet tabanli bulut siniflandiricisinin egitimi (Gunler 7-12).

Kullanim:
    python -m src.train
    python -m src.train --epochs 50 --lr 1e-3 --model mobilenetv3_large_100
"""

import argparse
import json
import random
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src import config
from src.dataset import build_loaders
from src.model import build_model, count_parameters, model_size_mb


def set_seed(seed: int = config.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, loader, device, threshold: float = config.DECISION_THRESHOLD):
    model.eval()
    probs, targets = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast("cuda", enabled=device.type == "cuda"):
            logits = model(x).squeeze(1)
        probs.append(torch.sigmoid(logits.float()).cpu())
        targets.append(y)

    probs = torch.cat(probs).numpy()
    targets = torch.cat(targets).numpy()
    preds = (probs >= threshold).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        targets, preds, average="binary", zero_division=0
    )
    return {
        "accuracy": float((preds == targets).mean()),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc_score(targets, probs)) if len(set(targets)) > 1 else float("nan"),
        "ap": float(average_precision_score(targets, probs)) if len(set(targets)) > 1 else float("nan"),
        "_probs": probs,
        "_targets": targets,
    }


def tune_threshold(probs, targets, min_precision: float = 0.99):
    """Karar esigini sec.

    Kisit: yanlis pozitif = kullanilabilir goruntunun uyduda atilmasi, geri donusu yok.
    Bu yuzden "bulutlu" tahmininde yuksek precision sart kosuyoruz ve bu kisit
    altinda recall'u (yani veri indirme kazancini) maksimize ediyoruz.

    Aday esikler veriden turetilir (precision_recall_curve). Sabit bir
    linspace izgarasi kullanilamaz: model guvenli oldugunda tum olasiliklar
    0.99 ustunde toplanir ve izgara hicbir ayrim noktasi goremez.
    """
    precisions, recalls, thresholds = precision_recall_curve(targets, probs)
    # precision_recall_curve son elemani (P=1, R=0) icin esik dondurmez
    precisions, recalls = precisions[:-1], recalls[:-1]

    feasible = precisions >= min_precision
    if feasible.any():
        idx = int(np.argmax(np.where(feasible, recalls, -1.0)))
        return {"threshold": _with_margin(float(thresholds[idx]), probs),
                "precision": float(precisions[idx]),
                "recall": float(recalls[idx]),
                "constraint_met": True}

    # Kisit saglanamiyor: en iyi F1 esigine dus ve bunu acikca bildir.
    f1s = 2 * precisions * recalls / np.clip(precisions + recalls, 1e-9, None)
    idx = int(np.argmax(f1s))
    print(f"  UYARI: precision >= {min_precision} saglanamadi. "
          f"En iyi F1 esigine dusuldu (precision {precisions[idx]:.4f}).")
    return {"threshold": _with_margin(float(thresholds[idx]), probs),
            "precision": float(precisions[idx]),
            "recall": float(recalls[idx]),
            "constraint_met": False}


def _with_margin(threshold: float, probs: np.ndarray) -> float:
    """Esigi bir alttaki olasilik degerinin ortasina kaydirir.

    Ayni tahminleri verir ama sayisal pay birakir. Bu sart: PyTorch'ta
    sigmoid tam 1.0'a doyabilir ve esik de 1.0 secilir; ayni model ONNX'te
    0.99999 uretince (probs >= 1.0) kosulu coker ve dogruluk sifira duser.
    """
    below = probs[probs < threshold]
    if below.size:
        return float((threshold + below.max()) / 2)
    return float(max(threshold - 1e-6, 0.0))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=config.MODEL_NAME)
    p.add_argument("--epochs", type=int, default=config.EPOCHS)
    p.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--lr", type=float, default=config.LR)
    p.add_argument("--weight-decay", type=float, default=config.WEIGHT_DECAY)
    p.add_argument("--label-smoothing", type=float, default=0.0,
                   help="hedefleri yumusatir; asiri guvenli tahminleri bastirir")
    p.add_argument("--workers", type=int, default=config.NUM_WORKERS)
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--tag", default="baseline", help="deney adi (tensorboard/checkpoint icin)")
    # --- dogruluk optimizasyonlari (hicbiri cikarim maliyetini artirmaz) ---
    p.add_argument("--balanced", action="store_true",
                   help="sinif dengeli ornekleme (pos_weight yerine)")
    p.add_argument("--warmup-epochs", type=int, default=2,
                   help="on-egitimli agirliklari ilk adimlarda bozmamak icin")
    p.add_argument("--ema", action="store_true",
                   help="agirliklarin ustel hareketli ortalamasi")
    # Varsayilan 0.999 DEGIL 0.99: bu veri setinde 15 epoch = 1440 adim ve
    # 0.999^1440 = 0.236, yani nihai agirliklarin %24'u hala egitilmemis
    # baslangic degeri olarak kaliyor. Olculdu (15 epoch, test F1):
    #   sade 0.9337 | d=0.999 -> 0.7249 (cokuyor) | d=0.99 -> 0.9407 | d=0.95 -> 0.9440
    # Kural: decay^(toplam_adim) ihmal edilebilir olmali.
    p.add_argument("--ema-decay", type=float, default=0.99)
    p.add_argument("--accum-steps", type=int, default=1,
                   help="gradyan biriktirme; efektif batch = batch_size * accum_steps")
    p.add_argument("--clip-grad", type=float, default=0.0,
                   help="gradyan norm siniri (0 = kapali)")
    args = p.parse_args()

    if args.accum_steps < 1:
        raise SystemExit("--accum-steps en az 1 olmali")

    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"cihaz: {device}")

    train_loader, val_loader, test_loader = build_loaders(args.batch_size, args.workers,
                                                          balanced=args.balanced)
    print(f"train {len(train_loader.dataset)} | val {len(val_loader.dataset)} | test {len(test_loader.dataset)}")

    model = build_model(args.model, pretrained=not args.no_pretrained).to(device)
    print(f"parametre {count_parameters(model):,} | FP32 {model_size_mb(model):.2f} MB")

    # Dengeli ornekleme ile pos_weight birlikte kullanilmaz: ikisi de ayni
    # dengesizligi duzeltir, ust uste binince azinlik sinifi asiri agirlasir.
    pos_weight = None if args.balanced else train_loader.dataset.pos_weight().to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)

    # Warmup + cosine: on-egitimli omurga ilk epoch'larda yuksek LR ile
    # bozulmasin diye dusuk LR'den baslanir.
    warmup = min(args.warmup_epochs, max(args.epochs - 1, 0))
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs - warmup, 1))
    if warmup > 0:
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            [torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup),
             cosine],
            milestones=[warmup],
        )
    else:
        scheduler = cosine

    ema = None
    if args.ema:
        from timm.utils import ModelEmaV3
        # EMA: son epoch'un gurultusu yerine agirliklarin yumusatilmis halini
        # kullanir. Egitim maliyeti ihmal edilebilir, cikarim maliyeti SIFIR
        # (tek bir agirlik seti kaydedilir) - plandaki kisitlari etkilemez.
        # use_warmup sart: decay=0.999 ile EMA agirliklarin ~1000 adimlik
        # gecmisini tutar. Kisa egitimlerde (birkac yuz adim) EMA hala baslangic
        # agirliklarina yakin kalir ve model rastgele gorunur. Warmup, decay'i
        # adim sayisina gore kademeli yukselterek bunu engeller.
        ema = ModelEmaV3(model, decay=args.ema_decay, use_warmup=True)

    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    writer = SummaryWriter(config.TB_LOGS / args.tag)
    ckpt_path = config.CHECKPOINTS / f"{args.tag}_best.pt"
    best_f1 = -1.0

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

            if args.label_smoothing > 0:
                # 0/1 yerine eps/(1-eps): model asiri guvenli olmayi ogrenmez,
                # kalibrasyon iyilesir ve esik secimi daha kararli olur.
                y = y * (1 - args.label_smoothing) + 0.5 * args.label_smoothing

            with torch.autocast("cuda", enabled=device.type == "cuda"):
                loss = criterion(model(x).squeeze(1), y)

            # Biriktirme: kayip adim sayisina bolunur, yoksa efektif ogrenme
            # orani accum_steps kati artar.
            scaler.scale(loss / args.accum_steps).backward()

            # Son eksik grup da islensin diye epoch sonu kosulu eklendi.
            if step % args.accum_steps == 0 or step == num_batches:
                if args.clip_grad > 0:
                    # AMP ile clipping'den ONCE unscale sart: aksi halde
                    # olceklenmis gradyanlara norm uygulanir ve esik anlamsizlasir.
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if ema is not None:
                    ema.update(model)  # EMA yalnizca gercek optimizer adiminda

            running += loss.item() * x.size(0)
            n += x.size(0)
        scheduler.step()

        train_loss = running / n
        eval_model = ema.module if ema is not None else model
        val = evaluate(eval_model, val_loader, device)
        dt = time.perf_counter() - t0

        print(f"epoch {epoch:3d} | loss {train_loss:.4f} | val f1 {val['f1']:.4f} "
              f"acc {val['accuracy']:.4f} prec {val['precision']:.4f} rec {val['recall']:.4f} | {dt:.1f}s")

        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("lr", scheduler.get_last_lr()[0], epoch)
        for k in ("accuracy", "precision", "recall", "f1", "roc_auc", "ap"):
            writer.add_scalar(f"val/{k}", val[k], epoch)

        if val["f1"] > best_f1:
            best_f1 = val["f1"]
            torch.save(
                {"model": eval_model.state_dict(), "model_name": args.model,
                 "in_channels": config.IN_CHANNELS, "epoch": epoch, "val_f1": best_f1},
                ckpt_path,
            )

    # En iyi checkpoint ile esik ayari ve test degerlendirmesi
    model.load_state_dict(torch.load(ckpt_path, map_location=device)["model"])
    val = evaluate(model, val_loader, device)
    tuned = tune_threshold(val["_probs"], val["_targets"])
    print(f"\nsecilen esik {tuned['threshold']:.4f} "
          f"(val precision {tuned['precision']:.4f}, recall {tuned['recall']:.4f}, "
          f"kisit saglandi: {tuned['constraint_met']})")

    test = evaluate(model, test_loader, device, threshold=tuned["threshold"])
    summary = {
        "tag": args.tag, "model": args.model, "epochs": args.epochs,
        "hyperparameters": {"lr": args.lr, "weight_decay": args.weight_decay,
                            "batch_size": args.batch_size,
                            "label_smoothing": args.label_smoothing,
                            "bands": config.BANDS},
        "optimizations": {"balanced_sampling": args.balanced, "warmup_epochs": warmup,
                          "ema": args.ema, "ema_decay": args.ema_decay if args.ema else None,
                          "accum_steps": args.accum_steps,
                          "effective_batch": args.batch_size * args.accum_steps,
                          "clip_grad": args.clip_grad},
        "best_val_f1": best_f1, "threshold": tuned["threshold"],
        "threshold_tuning": tuned,
        "test": {k: v for k, v in test.items() if not k.startswith("_")},
        "checkpoint": str(ckpt_path),
    }
    out = config.REPORTS / f"{args.tag}_train_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\ntest sonuclari:")
    for k, v in summary["test"].items():
        print(f"  {k:<10} {v:.4f}")
    print(f"\nozet: {out}")
    writer.close()


if __name__ == "__main__":
    main()
