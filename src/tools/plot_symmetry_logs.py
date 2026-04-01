"""
Script to plot symmetry error logs from CSV file.
Usage: python plot_symmetry_logs.py logs/symmetry_test/YYYY-MM-DD_HH-MM-SS/symmetry_error.csv
"""

import pandas as pd
import matplotlib.pyplot as plt
import argparse
import sys
import os

def main():
    parser = argparse.ArgumentParser(description="Plot Symmetry Error Logs")
    parser.add_argument("file", help="Path to the symmetry_error.csv file")
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"File not found: {args.file}")
        sys.exit(1)

    print(f"Loading {args.file}...")
    df = pd.read_csv(args.file)

    # Convert step to index
    if "step" in df.columns:
        df.set_index("step", inplace=True)

    # Plotting
    # We have 48+ columns. We should group them.
    # Groups: LinVel(3), AngVel(3), Grav(3), Cmd(3), JointPos(12), JointVel(12), PrevAct(12)
    
    groups = {
        "Base Linear Velocity": ["Lin Vel X", "Lin Vel Y", "Lin Vel Z"],
        "Base Angular Velocity": ["Ang Vel X", "Ang Vel Y", "Ang Vel Z"],
        "Gravity": ["Grav X", "Grav Y", "Grav Z"],
        "Commands": ["Cmd X", "Cmd Y", "Cmd Yaw"],
        "Joint Positions": [c for c in df.columns if c.startswith("Pos ")],
        "Joint Velocities": [c for c in df.columns if c.startswith("Vel ")],
        "Previous Actions": [c for c in df.columns if c.startswith("Act ")]
    }

    n_groups = len(groups)
    fig, axes = plt.subplots(n_groups, 1, figsize=(10, 3 * n_groups), sharex=True)
    
    if n_groups == 1:
        axes = [axes]

    for i, (name, cols) in enumerate(groups.items()):
        ax = axes[i]
        # Check if cols exist
        valid_cols = [c for c in cols if c in df.columns]
        if not valid_cols:
            continue
            
        df[valid_cols].plot(ax=ax)
        ax.set_title(f"{name} Error (Abs Diff)")
        ax.set_ylabel("Error")
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize='small', ncol=3)

    plt.xlabel("Step")
    plt.tight_layout()
    
    # Save plot to same dir
    out_path = args.file.replace(".csv", "_plot.png")
    plt.savefig(out_path)
    print(f"Plot saved to {out_path}")
    
    plt.show()

if __name__ == "__main__":
    main()
