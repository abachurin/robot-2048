"""
Generate PNG charts from benchmark JSON data.
- Learning curves (Float32 vs Float64 per N)
- Combined learning curves (all N, Float32)
- Histograms of final scores (distribution of 100 runs)
Usage: python3 plot_curves.py logs/curves_data.json logs/
"""
import json
import sys
import os
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
matplotlib.use('Agg')


def plot_all(data_path, output_dir):
    with open(data_path) as f:
        data = json.load(f)

    xs = data["episodes"]
    curves = data["curves"]
    finals = data.get("finals", {})
    n_runs = data.get("n_runs", 24)
    trim = data.get("trim", 2)
    n_values = sorted(set(int(k.split("_")[0][2:]) for k in curves))
    float_types = sorted(set(k.split("_")[1] for k in curves))

    # === Learning curves: one chart per N ===
    for n in n_values:
        fig, ax = plt.subplots(figsize=(10, 6))
        for T in float_types:
            key = f"N={n}_{T}"
            if key in curves:
                ax.plot(xs, curves[key], label=T, linewidth=2)
        ax.set_title(f"N={n} — Learning Curve (avg of {n_runs - 2*trim} trimmed runs)", fontsize=14)
        ax.set_xlabel("Episode", fontsize=12)
        ax.set_ylabel("Avg Score (last 100 episodes)", fontsize=12)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        path = os.path.join(output_dir, f"curve_n{n}.png")
        fig.savefig(path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved {path}")

    # === Combined chart: all N values, Float32 ===
    fig, ax = plt.subplots(figsize=(10, 6))
    for n in n_values:
        key = f"N={n}_Float32"
        if key in curves:
            ax.plot(xs, curves[key], label=f"N={n}", linewidth=2)
    ax.set_title("Learning Curves by N (Float32)", fontsize=14)
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Avg Score (last 100 episodes)", fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    path = os.path.join(output_dir, "curve_all_n.png")
    fig.savefig(path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {path}")

    # === Histograms: log(final score) distribution — should be ~normal by CLT ===
    if finals:
        for n in n_values:
            fig, ax = plt.subplots(figsize=(10, 6))
            has_data = False
            for T in float_types:
                key = f"N={n}_{T}"
                if key in finals:
                    log_vals = np.log(finals[key])
                    mu, sigma = np.mean(log_vals), np.std(log_vals)
                    ax.hist(log_vals, bins=20, alpha=0.6, label=f"{T} (μ={mu:.2f}, σ={sigma:.3f})",
                            edgecolor='black', linewidth=0.5, density=True)
                    # Overlay normal fit
                    x_fit = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
                    y_fit = (1 / (sigma * np.sqrt(2*np.pi))) * np.exp(-0.5*((x_fit - mu)/sigma)**2)
                    ax.plot(x_fit, y_fit, linewidth=2, linestyle='--')
                    has_data = True
            if has_data:
                ax.set_title(f"N={n} — log(Final Score) Distribution ({n_runs} runs each)", fontsize=14)
                ax.set_xlabel("log(Avg Score of last 1K episodes)", fontsize=12)
                ax.set_ylabel("Density", fontsize=12)
                ax.legend(fontsize=11)
                ax.grid(True, alpha=0.3, axis='y')
                path = os.path.join(output_dir, f"hist_n{n}.png")
                fig.savefig(path, dpi=100, bbox_inches='tight')
                print(f"Saved {path}")
            plt.close(fig)

        # === Combined histograms ===
        fig, axes = plt.subplots(1, len(n_values), figsize=(4 * len(n_values), 5))
        if len(n_values) == 1:
            axes = [axes]
        for ax, n in zip(axes, n_values):
            for T in float_types:
                key = f"N={n}_{T}"
                if key in finals:
                    log_vals = np.log(finals[key])
                    ax.hist(log_vals, bins=15, alpha=0.6, label=T,
                            edgecolor='black', linewidth=0.5, density=True)
            ax.set_title(f"N={n}", fontsize=12)
            ax.set_xlabel("log(Score)", fontsize=10)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3, axis='y')
        axes[0].set_ylabel("Density", fontsize=12)
        fig.suptitle(f"log(Final Score) Distributions ({n_runs} runs)", fontsize=14, y=1.02)
        fig.tight_layout()
        path = os.path.join(output_dir, "hist_all_n.png")
        fig.savefig(path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved {path}")


if __name__ == "__main__":
    data_path = sys.argv[1] if len(sys.argv) > 1 else "logs/curves_data.json"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "logs"
    plot_all(data_path, output_dir)
