import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_subjects_metrics_heatmap(df, save=False, path=None, metrics_cols=[], title="Cross-Subject Performance Comparison"):
    """
    Creates a heatmap comparing metrics (rows) across all subjects (columns).
    """
    # Select only the relevant performance metrics
    if len(metrics_cols) == 0:
        metrics_cols = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    
    metrics_sel = [m for m in metrics_cols if m in df.columns]
    
    # Transpose so Subjects are on X-axis and Metrics on Y-axis
    plot_data = df[metrics_sel].astype(float).T
    
    plt.figure(figsize=(12, 6))
    sns.heatmap(plot_data, annot=True, fmt=".3f", cmap="YlGnBu", linewidths=.5)
    
    plt.title(title, fontsize=15, pad=20)
    plt.ylabel("Performance Metrics", fontweight='bold')
    plt.xlabel("Subjects", fontweight='bold')
    
    if save and path is not None:
        plt.savefig(path, bbox_inches='tight')
    plt.show()

def plot_all_confusion_matrices(eval_dict, save=False, path=None):
    """
    Plots confusion matrices for all subjects in a single horizontal row.
    """
    n_subj = len(eval_dict)
    fig, axes = plt.subplots(1, n_subj, figsize=(4 * n_subj, 4), sharey=True)
    
    # handle the case of a single subject (where axes is not an array)
    if n_subj == 1: axes = [axes]

    for i, (subj_name, data) in enumerate(eval_dict.items()):
        cm = confusion_matrix(data["y_true"], data["y_pred"])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i], cbar=False)
        axes[i].set_title(f"Subject: {subj_name}")
        axes[i].set_xlabel("Predicted")
        if i == 0: axes[i].set_ylabel("Actual")

    plt.tight_layout()
    if save and path is not None:
        fig.savefig(path)
    plt.show()