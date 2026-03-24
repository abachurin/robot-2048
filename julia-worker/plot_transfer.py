"""Generate PNG charts for transfer benchmark."""
import json, sys, os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

def plot_all(data_path, output_dir, md_path=None):
    with open(data_path) as f:
        data = json.load(f)
    xs = data["episodes"]
    curves = data["curves"]

    fig, ax = plt.subplots(figsize=(12, 7))
    for key in curves:
        ax.plot(xs, curves[key], label=key, linewidth=2)
    ax.set_title("N=6 — Transfer vs No Transfer (constant alpha=0.15, avg of 5 runs)", fontsize=14)
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Avg Score (last 100 episodes)", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    path = os.path.join(output_dir, "transfer_curves.png")
    fig.savefig(path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {path}")

    if md_path:
        with open(md_path, "a") as f:
            f.write("\n## Learning Curves\n\n")
            f.write(f"![Transfer curves](/Users/abachurin/2048-react/julia-worker/logs/transfer_curves.png)\n")

if __name__ == "__main__":
    data_path = sys.argv[1] if len(sys.argv) > 1 else "logs/curves_transfer.json"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "logs"
    md_path = sys.argv[3] if len(sys.argv) > 3 else None
    plot_all(data_path, output_dir, md_path)
