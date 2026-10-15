"""
Quick-look plots from a .com.csv: the horizontal sway trace over time and a
CoM scatter. This is a sanity check on ONE clip, not the paper's agreement
analysis (that compares single-cam vs multi-cam and comes later).

Usage:
    python analyze.py INPUT.com.csv --view side
"""

from __future__ import annotations
import argparse
import csv

import numpy as np
import matplotlib.pyplot as plt


def load(path):
    t, x, y = [], [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row["com_x_norm"] == "":
                continue
            t.append(float(row["time_s"]))
            x.append(float(row["com_x_norm"]))
            y.append(float(row["com_y_norm"]))
    return np.array(t), np.array(x), np.array(y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--view", default="side", choices=["front", "side"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    t, x, y = load(args.csv)
    if len(t) == 0:
        raise SystemExit("No CoM samples in CSV (pose never detected?).")
    sway_label = "medio-lateral" if args.view == "front" else "antero-posterior"
    # Report sway relative to the mean, in % of frame width.
    sway = (x - x.mean()) * 100.0

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.plot(t, sway, lw=1.2)
    a1.axhline(0, color="k", lw=0.5)
    a1.set_xlabel("time (s)")
    a1.set_ylabel(f"{sway_label} sway (% frame width)")
    a1.set_title(f"CoM sway  (RMS = {np.sqrt(np.mean(sway**2)):.2f}%)")

    a2.plot(x * 100, (1 - y) * 100, lw=0.8, alpha=0.7)
    a2.scatter([x.mean() * 100], [(1 - y.mean()) * 100], c="r", zorder=3, label="mean")
    a2.set_xlabel("CoM x (% width)")
    a2.set_ylabel("CoM height (% height, up)")
    a2.set_title("CoM path")
    a2.legend()
    fig.tight_layout()

    out = args.out or args.csv.replace(".csv", ".png")
    fig.savefig(out, dpi=120)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
