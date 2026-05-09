#cear_pilot/scripts/plot_assay_pca.py
#!/usr/bin/env python
from pathlib import Path
import argparse
import pandas as pd
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pca", default="outputs/p3_v3_assay_geometry/assay_g_pca_points.parquet")
    ap.add_argument("--outdir", default="outputs/p3_v3_assay_geometry/figures")
    ap.add_argument("--seed", type=int, default=-1, help="If >=0, plot one seed only.")
    args = ap.parse_args()

    df = pd.read_parquet(args.pca)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.seed >= 0:
        df = df[df["seed"] == args.seed].copy()

    cohorts = ["full", "no_body_in_g", "no_conative"]
    conditions = ["control", "body_shock"]
    phases = ["pre", "shock", "recovery"]

    for cohort in cohorts:
        sub = df[df["cohort"] == cohort].copy()
        if sub.empty:
            continue

        fig, ax = plt.subplots(figsize=(7, 6))

        for condition in conditions:
            for phase in phases:
                d = sub[(sub["condition"] == condition) & (sub["phase"] == phase)]
                if d.empty:
                    continue
                label = f"{condition}:{phase}"
                ax.scatter(d["PC1"], d["PC2"], s=6, alpha=0.35, label=label)

                # centroid marker
                cx, cy = d["PC1"].mean(), d["PC2"].mean()
                ax.scatter([cx], [cy], s=80, marker="x")

        title = cohort if args.seed < 0 else f"{cohort} seed {args.seed}"
        ax.set_title(f"g-space PCA: {title}")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()

        suffix = "allseeds" if args.seed < 0 else f"s{args.seed}"
        fig.savefig(outdir / f"g_pca_{cohort}_{suffix}.png", dpi=200)
        plt.close(fig)

    print(f"[save] figures to {outdir}")

if __name__ == "__main__":
    main()