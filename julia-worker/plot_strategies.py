"""
Generate PNG charts for strategy comparison.
Usage: python3 plot_strategies.py logs/curves_strategies.json logs/ logs/benchmark_strategies.md
"""
import json
import sys
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


def plot_all(data_path, output_dir, md_path=None):
    with open(data_path) as f:
        data = json.load(f)

    xs = data["episodes"]
    curves = data["curves"]

    # Separate alpha-mode configs from cutoff configs
    alpha_configs = [k for k in curves if "cutoff" not in k]
    cutoff_configs = ["global_decay"] + [k for k in curves if "cutoff" in k]

    # === Chart 1: Alpha mode comparison ===
    fig, ax = plt.subplots(figsize=(12, 7))
    for key in alpha_configs:
        ax.plot(xs, curves[key], label=key, linewidth=2)
    ax.set_title("N=6 — Learning Rate Strategies (200K episodes, avg of 5 runs)", fontsize=14)
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Avg Score (last 100 episodes)", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    path = os.path.join(output_dir, "strategies_alpha.png")
    fig.savefig(path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {path}")

    # === Chart 2: Cutoff comparison ===
    fig, ax = plt.subplots(figsize=(12, 7))
    for key in cutoff_configs:
        if key in curves:
            label = {"global_decay": "cutoff=13 (original)", "cutoff_14": "cutoff=14", "cutoff_15": "cutoff=15 (no clamp)"}.get(key, key)
            ax.plot(xs, curves[key], label=label, linewidth=2)
    ax.set_title("N=6 — Cutoff Comparison (200K episodes, avg of 5 runs)", fontsize=14)
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Avg Score (last 100 episodes)", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    path = os.path.join(output_dir, "strategies_cutoff.png")
    fig.savefig(path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {path}")

    # === Chart 3: All configs together ===
    fig, ax = plt.subplots(figsize=(14, 8))
    for key in curves:
        ax.plot(xs, curves[key], label=key, linewidth=1.5)
    ax.set_title("N=6 — All Strategies (200K episodes, avg of 5 runs)", fontsize=14)
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Avg Score (last 100 episodes)", fontsize=12)
    ax.legend(fontsize=10, ncol=2)
    ax.grid(True, alpha=0.3)
    path = os.path.join(output_dir, "strategies_all.png")
    fig.savefig(path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {path}")

    # Append charts to markdown
    if md_path:
        with open(md_path, "a") as f:
            f.write("\n## Learning Curves — Alpha Strategies\n\n")
            f.write("![Alpha strategies](strategies_alpha.png)\n\n")
            f.write("## Learning Curves — Cutoff Comparison\n\n")
            f.write("![Cutoff comparison](strategies_cutoff.png)\n\n")
            f.write("## All Strategies\n\n")
            f.write("![All strategies](strategies_all.png)\n")


if __name__ == "__main__":
    data_path = sys.argv[1] if len(sys.argv) > 1 else "logs/curves_strategies.json"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "logs"
    md_path = sys.argv[3] if len(sys.argv) > 3 else None
    plot_all(data_path, output_dir, md_path)
