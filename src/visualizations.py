import json
import matplotlib.pyplot as plt
import numpy as np
import os

def load_data(data_path):
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return None

    with open(data_path, "r") as f:
        data = json.load(f)
    return data


def plot_results_from_json(data_path, filename):
    json_path = os.path.join(data_path, filename)
    data = load_data(json_path)
    plot = plot_overall(data)
    save_plt(plot, data_path, "overall_performance")
    plot = plot_fine_tuned(data)
    if plot:
        save_plt(plot, data_path, "per_class_performance")


def plot_overall(data):
    # --- PLOT 1: OVERALL COMPARISON ---
    metrics_keys = ["precision", "recall", "mAP50", "mAP50_95"]

    # Extract overall values (handling cases where a model might be missing)
    baseline = data.get("pre_trained_baseline", {}).get("overall", {})
    fine_tuned = data.get("fine_tuned_model", {}).get("overall", {})

    baseline_vals = [baseline.get(k, 0) for k in metrics_keys]
    tuned_vals = [fine_tuned.get(k, 0) for k in metrics_keys]

    x = np.arange(len(metrics_keys))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width / 2, baseline_vals, width, label="Pre-trained Baseline", color="#e74c3c")
    ax.bar(x + width / 2, tuned_vals, width, label="Fine-tuned Model", color="#2ecc71")

    ax.set_ylabel("Score")
    ax.set_title("Overall Performance Comparison (Baseline vs Fine-tuned)")
    ax.set_xticks(x)
    ax.set_xticklabels([m.capitalize() for m in metrics_keys])
    ax.set_ylim(0, 1.1)
    ax.legend()

    # Add text labels on top of bars
    for i, val in enumerate(baseline_vals):
        ax.text(i - width / 2, val + 0.02, f"{val:.2f}", ha="center", fontsize=10)
    for i, val in enumerate(tuned_vals):
        ax.text(i + width / 2, val + 0.02, f"{val:.2f}", ha="center", fontsize=10)

    plt.tight_layout()
    plt.show(block=False)
    return plt

def plot_fine_tuned(data):
    per_class_data = data.get("fine_tuned_model", {}).get("per_class", {})

    if per_class_data:
        class_names = list(per_class_data.keys())
        class_mAP50 = [per_class_data[cls]["mAP50"] for cls in class_names]

        plt.figure(figsize=(10, 6))
        bars = plt.bar(class_names, class_mAP50, color=plt.cm.viridis(np.linspace(0, 0.8, len(class_names))))

        plt.ylabel("mAP50")
        plt.title("Fine-tuned Model: mAP50 per Class")
        plt.ylim(0, 1.1)

        # Add values on top
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.02, f"{yval:.2f}", ha="center", fontweight="bold")

        plt.tight_layout()

        plt.show(block=False)
        return plt
    else:
        print("No fine-tuned per-class data found to plot.")
        return None

def save_plt(plot, path, name):
    output_path = os.path.join(path, name)
    plot.savefig(output_path)
    print(f"Saved {name} plot to: {output_path}")