"""
This script takes a list of files an ordered list of CSV files as input. These file
must contain three unnamed columns:

- Step (float)
- Timestamp (int)
- Loss (float)

The script will output a plot as a PNG file that concatenates the loss curves of the
input files.
"""

import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

STEP_COLUMN = "step"
TIMESTAMP_COLUMN = "timestamp"
LOSS_COLUMN = "loss"
SMOOTHED_LOSS_COLUMN = "smoothed_loss"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot loss curves from CSV files")
    parser.add_argument("-a", nargs="+", help="List of files contributing to curve A.")
    parser.add_argument("-b", nargs="+", help="List of files contributing to curve B.")
    parser.add_argument("--output", "-o", metavar="FILE", type=str, default="losscurve.png", help="Output file (default: losscurve.png)")
    parser.add_argument("--ylim", "-y", metavar="Y", type=float, nargs=2, default=[1, 3], help="Y-axis limits (default: [1, 3])")
    args = parser.parse_args()

    # Read CSV files for A
    dfs = []
    for file in args.a:
        df = pd.read_csv(file, header=None, names=["step", "timestamp", "loss"])
        dfs.append(df)
    df_a = pd.concat(dfs, ignore_index=True)
    df_a[SMOOTHED_LOSS_COLUMN] = df_a[LOSS_COLUMN].rolling(window=512).mean()

    dfs = []
    for file in args.b:
        df = pd.read_csv(file, header=None, names=["step", "timestamp", "loss"])
        dfs.append(df)
    df_b = pd.concat(dfs, ignore_index=True)
    df_b[SMOOTHED_LOSS_COLUMN] = df_b[LOSS_COLUMN].rolling(window=512).mean()

    plt.figure(figsize=(4, 2))  # Increase the size as needed
    sns.lineplot(data=df_a[SMOOTHED_LOSS_COLUMN], label="Custom-built tokenizer")
    sns.lineplot(data=df_b[SMOOTHED_LOSS_COLUMN], label="HuggingFace tokenizer")
    plt.xlim(left=0, right=min(df_a.index.max(), df_b.index.max()))
    plt.ylim(args.ylim[0], args.ylim[1])
    plt.ylabel('Loss')
    plt.xlabel('Steps')
    plt.grid(True)
    plt.savefig(args.output, dpi=300, bbox_inches='tight')
