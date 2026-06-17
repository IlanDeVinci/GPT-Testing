"""Visualise l'entrainement (loss + accuracy) depuis metrics.jsonl.

Trois usages :

    # 1. courbes d'un run, rendu fixe + PNG
    python scripts/plot_training.py --metrics checkpoints-v3/metrics.jsonl

    # 2. suivi en direct pendant l'entrainement
    python scripts/plot_training.py --metrics checkpoints-v3/metrics.jsonl --live

    # 3. comparer plusieurs runs sur un meme graphe
    python scripts/plot_training.py --compare checkpoints-v3/metrics.jsonl checkpoints-sft/metrics.jsonl \
        --labels "pre-entrainement" "SFT" --out outputs/compare.png

Par defaut l'accuracy est tracee en UNE seule ligne (lissee). Pour revoir les
trois courbes top-1 / top-5 / top-10, ajoute --all_accuracy.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


# Palette douce et lisible, reutilisee pour la comparaison multi-runs.
PALETTE = ["#534ab7", "#0f6e56", "#a05a2c", "#2f7ed8", "#b13b6b", "#7a7a7a"]
ACCURACY_FIELDS = ("top1_accuracy", "top5_accuracy", "top10_accuracy")
ACCURACY_LABELS = {
    "accuracy": "accuracy",
    "top1_accuracy": "top-1 (stricte)",
    "top5_accuracy": "top-5",
    "top10_accuracy": "top-10",
}


def read_metrics(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    # Si le fichier contient plusieurs runs concatenes (reprise -> le compteur
    # d'etapes redescend), on ne garde que le dernier run.
    start = 0
    for i in range(1, len(records)):
        if records[i]["step"] <= records[i - 1]["step"]:
            start = i
    return records[start:]


def smooth(values: list[float], window: int) -> list[float]:
    """Moyenne glissante centree, pour une courbe lisible sans gommer la tendance."""
    if window <= 1 or len(values) < 3:
        return values
    window = min(window, len(values))
    half = window // 2
    out: list[float] = []
    for i in range(len(values)):
        lo = max(0, i - half)
        hi = min(len(values), i + half + 1)
        chunk = values[lo:hi]
        out.append(sum(chunk) / len(chunk))
    return out


def pick_accuracy_field(records: list[dict], requested: str) -> str:
    """Choisit le champ d'accuracy a tracer : celui demande s'il existe, sinon
    le meilleur disponible (top-5 de preference)."""
    last = records[-1]
    if requested in last:
        return requested
    for field in ("top5_accuracy", "top1_accuracy", "accuracy", "top10_accuracy"):
        if field in last:
            return field
    return "accuracy"


def style_axis(ax) -> None:
    ax.grid(True, alpha=0.25, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def draw_single(axes, records, accuracy_field: str, all_accuracy: bool, smooth_window: int) -> None:
    loss_ax, acc_ax = axes
    loss_ax.clear()
    acc_ax.clear()

    if not records:
        loss_ax.set_title("Aucune metrique pour le moment")
        return

    steps = [r["step"] for r in records]

    # --- Loss : courbe brute pale + courbe lissee nette ---
    losses = [r["loss"] for r in records]
    loss_ax.plot(steps, losses, color=PALETTE[0], alpha=0.25, linewidth=1)
    loss_ax.plot(steps, smooth(losses, smooth_window), color=PALETTE[0], linewidth=2.2, label="loss")
    loss_ax.set_ylabel("loss")
    loss_ax.set_title("Evolution de la loss")
    loss_ax.annotate(
        f"{losses[-1]:.3f}",
        (steps[-1], losses[-1]),
        textcoords="offset points",
        xytext=(-4, 8),
        ha="right",
        fontsize=9,
        color=PALETTE[0],
    )
    style_axis(loss_ax)

    # --- Accuracy ---
    field = pick_accuracy_field(records, accuracy_field)
    if all_accuracy and any(f in records[-1] for f in ACCURACY_FIELDS):
        for i, name in enumerate(f for f in ACCURACY_FIELDS if f in records[-1]):
            vals = [r[name] for r in records if name in r]
            xs = [r["step"] for r in records if name in r]
            acc_ax.plot(xs, smooth(vals, smooth_window), color=PALETTE[i + 1], linewidth=2, label=ACCURACY_LABELS[name])
        acc_ax.legend(loc="lower right", frameon=False)
    else:
        vals = [r[field] for r in records if field in r]
        xs = [r["step"] for r in records if field in r]
        acc_ax.plot(xs, vals, color=PALETTE[1], alpha=0.25, linewidth=1)
        acc_ax.plot(xs, smooth(vals, smooth_window), color=PALETTE[1], linewidth=2.4, label=ACCURACY_LABELS.get(field, field))
        if vals:
            acc_ax.annotate(
                f"{ACCURACY_LABELS.get(field, field)} = {vals[-1]:.3f}",
                (xs[-1], vals[-1]),
                textcoords="offset points",
                xytext=(-4, -14),
                ha="right",
                fontsize=9,
                color=PALETTE[1],
            )

    acc_ax.set_ylabel("accuracy")
    acc_ax.set_xlabel("etape (mise a jour)")
    acc_ax.set_title("Accuracy de prediction du token suivant")
    acc_ax.set_ylim(0, 1)
    style_axis(acc_ax)


def draw_compare(metrics_paths: list[Path], labels: list[str], accuracy_field: str, smooth_window: int):
    fig, (loss_ax, acc_ax) = plt.subplots(2, 1, figsize=(9, 7.5), sharex=True)
    for i, path in enumerate(metrics_paths):
        records = read_metrics(path)
        if not records:
            print(f"(ignore, vide) {path}")
            continue
        color = PALETTE[i % len(PALETTE)]
        label = labels[i] if i < len(labels) else path.parent.name
        steps = [r["step"] for r in records]
        losses = [r["loss"] for r in records]
        loss_ax.plot(steps, smooth(losses, smooth_window), color=color, linewidth=2.2, label=label)

        field = pick_accuracy_field(records, accuracy_field)
        vals = [r[field] for r in records if field in r]
        xs = [r["step"] for r in records if field in r]
        if vals:
            acc_ax.plot(xs, smooth(vals, smooth_window), color=color, linewidth=2.2, label=f"{label} ({ACCURACY_LABELS.get(field, field)})")

    loss_ax.set_ylabel("loss")
    loss_ax.set_title("Comparaison des runs - loss")
    loss_ax.legend(loc="upper right", frameon=False)
    style_axis(loss_ax)

    acc_ax.set_ylabel("accuracy")
    acc_ax.set_xlabel("etape (mise a jour)")
    acc_ax.set_title("Comparaison des runs - accuracy")
    acc_ax.set_ylim(0, 1)
    acc_ax.legend(loc="lower right", frameon=False)
    style_axis(acc_ax)
    fig.tight_layout()
    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trace la loss et l'accuracy d'entrainement.")
    parser.add_argument("--metrics", type=Path, default=Path("checkpoints-v3/metrics.jsonl"))
    parser.add_argument("--compare", type=Path, nargs="+", default=None,
                        help="Plusieurs metrics.jsonl a superposer sur un meme graphe.")
    parser.add_argument("--labels", type=str, nargs="+", default=[],
                        help="Libelles des runs en mode --compare (meme ordre).")
    parser.add_argument("--out", type=Path, default=Path("outputs/training-curves.png"))
    parser.add_argument("--accuracy_field",
                        choices=["accuracy", "top1_accuracy", "top5_accuracy", "top10_accuracy"],
                        default="top5_accuracy",
                        help="Champ d'accuracy a tracer quand il existe.")
    parser.add_argument("--all_accuracy", action="store_true",
                        help="Trace les 3 courbes top-1/top-5/top-10 au lieu d'une seule.")
    parser.add_argument("--smooth", type=int, default=5,
                        help="Fenetre de lissage (moyenne glissante). 1 = pas de lissage.")
    parser.add_argument("--show", action="store_true", help="Ouvre aussi la fenetre matplotlib.")
    parser.add_argument("--live", action="store_true", help="Rafraichit en continu.")
    parser.add_argument("--refresh", type=float, default=3.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        pass

    if args.compare:
        fig = draw_compare(args.compare, args.labels, args.accuracy_field, args.smooth)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.out, dpi=130, bbox_inches="tight")
        print(f"Graphe de comparaison sauvegarde : {args.out}")
        if args.show:
            plt.show()
        return

    fig, axes = plt.subplots(2, 1, figsize=(9, 7.5), sharex=True)
    if args.live:
        plt.ion()
        print("Mode live - Ctrl-C pour arreter le graphe (l'entrainement continue).")
        try:
            while True:
                draw_single(axes, read_metrics(args.metrics), args.accuracy_field, args.all_accuracy, args.smooth)
                fig.tight_layout()
                plt.pause(args.refresh)
        except KeyboardInterrupt:
            pass
    else:
        records = read_metrics(args.metrics)
        if not records:
            raise SystemExit(f"Aucune metrique dans {args.metrics}.")
        draw_single(axes, records, args.accuracy_field, args.all_accuracy, args.smooth)
        fig.tight_layout()
        args.out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.out, dpi=130, bbox_inches="tight")
        print(f"Graphe sauvegarde : {args.out}")
        if args.show:
            plt.show()


if __name__ == "__main__":
    main()
