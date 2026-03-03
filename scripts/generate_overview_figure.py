from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> int:
    out = Path("assets/overview_pipeline.png")
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11.5, 4.2), dpi=180)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 30)
    ax.axis("off")

    boxes = [
        (4, 10, 20, 10, "Input Traces\n+ Source Wavelet"),
        (28, 10, 20, 10, "Zero-Phase\nFilter Stage"),
        (52, 10, 20, 10, "Time-Iteration\nDecon"),
        (76, 10, 20, 10, "Batch Parallel\n+ Metrics"),
    ]

    for x, y, w, h, txt in boxes:
        rect = plt.Rectangle((x, y), w, h, facecolor="#f1f5f9", edgecolor="#334155", linewidth=1.4)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=10, color="#0f172a")

    arrows = [
        (24, 15, 28, 15),
        (48, 15, 52, 15),
        (72, 15, 76, 15),
    ]
    for x0, y0, x1, y1 in arrows:
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", lw=1.8, color="#1d4ed8"))

    ax.text(50, 26, "VertiRF Processing Pipeline", ha="center", va="center", fontsize=14, color="#0f172a", fontweight="bold")
    ax.text(50, 3.2, "baseline vs optimized | gaussian / butterworth / raised-cosine / tukey", ha="center", fontsize=9, color="#475569")

    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(str(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
