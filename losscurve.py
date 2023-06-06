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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot loss curves from CSV files")
    parser.add_argument("files", metavar="FILE", type=str, nargs="+", help="CSV files to plot (in order)")
    parser.add_argument("--output", "-o", metavar="FILE", type=str, default="losscurve.png", help="Output file (default: losscurve.png)")
    args = parser.parse_args()

    fig, ax = plt.subplots()

    for i, file in enumerate(args.files):
        df = pd.read_csv(file, header=None)
        ax.plot(df[0], df[2], label=f"Run {i+1}")

    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.legend()
    fig.savefig(args.output)
