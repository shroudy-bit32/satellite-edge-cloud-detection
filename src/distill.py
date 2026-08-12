"""Bilgi damitma: buyuk ogretmen modelden kucuk ogrenci modele.

Amac: modelin BOYUTUNU buyutmeden dogrulugu artirmak. Ogretmen yalnizca
egitim sirasinda kullanilir; dagitilan model ogrencidir, dolayisiyla cikarim
maliyeti ve boyut degismez - PDF'in kisitlari korunur.

Neden ise yarar: ogretmenin yumusak olasiliklari ("bu kare %70 bulutlu
gorunuyor") sert etiketlerden ("bulutlu") daha fazla bilgi tasir. Ozellikle
belirsiz karelerde ogrenciye "ne kadar emin olmasi gerektigini" ogretir.

Kullanim:
    python -m src.distill --teacher-model mobilenetv2_140 --student-model mobilenetv2_100
    python -m src.distill --teacher outputs/checkpoints/teacher_best.pt --student-model mobilenetv2_050
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from src import config
from src.dataset import build_loaders
from src.export import load_from_checkpoint
from src.model import build_model, count_parameters, model_size_mb
from src.train import evaluate, set_seed, tune_threshold


def distillation_loss(student_logits, teacher_logits, targets, alpha, temperature, criterion):
    """Sert etiket kaybi + ogretmenin yumusak hedefleri.

    Ikili tek-logit durumunda KL diverjansi, sicaklikla olceklenmis
    logitler uzerinde BCE ile esdegerdir. T^2 carpani, sicakligin
    gradyan buyuklugunu kucultmesini telafi eder (Hinton ve ark.).
    """
    hard = criterion(student_logits, targets)

    soft_targets = torch.sigmoid(teacher_logits / temperature)
    soft = F.binary_cross_entropy_with_logits(student_logits / temperature, soft_targets)

    return alpha * hard + (1 - alpha) * (temperature**2) * soft


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher", type=Path, default=None,
                   help="hazir ogretmen checkpoint'i; yoksa once egitilir")
    p.add_argument("--teacher-model", default="mobilenetv2_140")
    p.add_argument("--teacher-epochs", type=int, default=15)
    p.add_argument("--student-model", default="mobilenetv2_100")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--lr", type=float, default=config.LR)
    p.add_argument("--weight-decay", type=float, default=config.WEIGHT_DECAY)
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument("--workers", type=int, default=config.NUM_WORKERS)
    p.add_argument("--alpha", type=float, default=0.3,
                   help="sert etiket agirligi (0=yalnizca ogretmen, 1=damitma yok)")
    p.add_argument("--temperature", type=float, default=3.0)
    p.add_argument("--ema", action="store_true", default=True)
    p.add_argument("--tag", default="distilled")
    args = p.parse_args()

    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader = build_loaders(args.batch_size, args.workers)

    # --- Ogretmen ---
    if args.teacher and args.teacher.exists():
        teacher, _ = load_from_checkpoint(args.teacher)
        print(f"ogretmen yuklendi: {args.teacher}")
    else:
        print(f"ogretmen egitiliyor: {args.teacher_model} ({args.teacher_epochs} epoch)")
        import subprocess
        import sys

        teacher_tag = f"{args.tag}_teacher"
        cmd = [sys.executable, "-m", "src.train", "--tag", teacher_tag,
               "--model", args.teacher_model, "--epochs", str(args.teacher_epochs),
               "--batch-size", str(args.batch_size), "--workers", str(args.workers), "--ema"]
        r = subprocess.run(cmd, cwd=config.PROJECT_ROOT)
        if r.returncode != 0:
            raise SystemExit("ogretmen egitimi basarisiz")
        teacher, _ = load_from_checkpoint(config.CHECKPOINTS / f"{teacher_tag}_best.pt")

    teacher = teacher.to(device).eval()
    for param in teacher.parameters():
        param.requires_grad = False

    teacher_metrics = evaluate(teacher, test_loader, device)
    print(f"ogretmen test F1: {teacher_metrics['f1']:.4f} "
          f"({count_parameters(teacher):,} parametre, {model_size_mb(teacher):.2f} MB)")

    # --- Ogrenci ---
    student = build_model(args.student_model).to(device)
    print(f"ogrenci: {args.student_model}, {count_parameters(student):,} parametre, "
          f"{model_size_mb(student):.2f} MB")

    criterion = nn.BCEWithLogitsLoss(pos_weight=train_loader.dataset.pos_weight().to(device))
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    warmup = min(2, max(args.epochs - 1, 0))
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs - warmup, 1))
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        [torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup), cosine],
        milestones=[warmup]) if warmup > 0 else cosine
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    ema = None
    if args.ema:
        from timm.utils import ModelEmaV3
        ema = ModelEmaV3(student, decay=0.99, use_warmup=True)

    ckpt_path = config.CHECKPOINTS / f"{args.tag}_best.pt"
    best_f1 = -1.0

    for epoch in range(1, args.epochs + 1):
        student.train()
        running, n = 0.0, 0
        t0 = time.perf_counter()

        for x, y in tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}", leave=False):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            if args.label_smoothing > 0:
                y = y * (1 - args.label_smoothing) + 0.5 * args.label_smoothing

            with torch.no_grad():
                teacher_logits = teacher(x).squeeze(1).float()

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=device.type == "cuda"):
                student_logits = student(x).squeeze(1)
                loss = distillation_loss(student_logits.float(), teacher_logits, y,
                                         args.alpha, args.temperature, criterion)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if ema is not None:
                ema.update(student)

            running += loss.item() * x.size(0)
            n += x.size(0)
        scheduler.step()

        eval_model = ema.module if ema is not None else student
        val = evaluate(eval_model, val_loader, device)
        print(f"epoch {epoch:3d} | loss {running / n:.4f} | val f1 {val['f1']:.4f} "
              f"acc {val['accuracy']:.4f} | {time.perf_counter() - t0:.1f}s")

        if val["f1"] > best_f1:
            best_f1 = val["f1"]
            torch.save({"model": eval_model.state_dict(), "model_name": args.student_model,
                        "in_channels": config.IN_CHANNELS, "epoch": epoch,
                        "val_f1": best_f1, "distilled_from": args.teacher_model}, ckpt_path)

    student.load_state_dict(torch.load(ckpt_path, map_location=device)["model"])
    val = evaluate(student, val_loader, device)
    tuned = tune_threshold(val["_probs"], val["_targets"])
    test = evaluate(student, test_loader, device, threshold=tuned["threshold"])

    summary = {"tag": args.tag, "model": args.student_model,
               "teacher": args.teacher_model,
               "teacher_test_f1": round(teacher_metrics["f1"], 4),
               "alpha": args.alpha, "temperature": args.temperature,
               "epochs": args.epochs, "best_val_f1": best_f1,
               "threshold": tuned["threshold"], "threshold_tuning": tuned,
               "test": {k: v for k, v in test.items() if not k.startswith("_")},
               "checkpoint": str(ckpt_path)}
    out = config.REPORTS / f"{args.tag}_train_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\ntest sonuclari (ogrenci):")
    for k, v in summary["test"].items():
        print(f"  {k:<10} {v:.4f}")
    print(f"\nogretmen test F1: {teacher_metrics['f1']:.4f}")
    print(f"ogrenci  test F1: {test['f1']:.4f}")
    print(f"ozet: {out}")


if __name__ == "__main__":
    main()
