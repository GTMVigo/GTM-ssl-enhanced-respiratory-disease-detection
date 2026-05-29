"""
Script to plot ROC curves with cross-validation folds from pickle files.
Combines data from acoustic + wavlm + wav2vec models across folds.
Creates mean ROC curves with variability bands.
"""

import pickle
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from sklearn.metrics import auc

def get_title_from_folder(moment_name):
    """
    Convert folder name to title format.
    Examples: after_a -> "Fusion model /a/ after"
              before_cough -> "Fusion model /cough/ before"
    """
    if 'after' in moment_name and '_a' in moment_name:
        return "Fusion model /a/ after"
    elif 'after' in moment_name and 'cough' in moment_name:
        return "Fusion model cough after"
    elif 'before' in moment_name and '_a' in moment_name:
        return "Fusion model /a/ before"
    elif 'before' in moment_name and 'cough' in moment_name:
        return "Fusion model cough before"
    else:
        return moment_name


def load_fold_data(roc_data_folder):
    """
    Load ROC data from fold pickle files.
    Returns lists of fpr and tpr for each fold, along with fold indices.
    
    Args:
        roc_data_folder: Path to folder containing fold pickle files
    
    Returns:
        Dictionary with fold_fprs, fold_tprs, fold_aucs lists
    """
    fold_fprs = []
    fold_tprs = []
    fold_aucs = []
    fold_indices = []
    
    # Find all combined model fold pickle files (containing wavlm or wav2vec)
    pkl_files = sorted(roc_data_folder.glob("*wavlm*.pkl")) + \
                sorted(roc_data_folder.glob("*wav2vec*.pkl"))
    
    # Remove duplicates while preserving order
    pkl_files = list(dict.fromkeys(pkl_files))
    
    if not pkl_files:
        print(f"  No combined model pickle files found in {roc_data_folder.name}")
        return None
    
    for pkl_file in pkl_files:
        try:
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)
            
            fold_fprs.append(data['fpr'])
            fold_tprs.append(data['tpr'])
            fold_aucs.append(data['auc_score'])
            fold_indices.append(len(fold_fprs) - 1)
            
        except Exception as e:
            print(f"  Error loading {pkl_file.name}: {e}")
            continue
    
    if not fold_fprs:
        return None
    
    return {
        'fold_fprs': fold_fprs,
        'fold_tprs': fold_tprs,
        'fold_aucs': fold_aucs,
        'fold_indices': fold_indices
    }


def plot_cv_roc_curves(fold_fprs, fold_tprs, fold_aucs, title, output_path):
    """
    Plot ROC curves for each fold with mean ROC and confidence band.
    
    Args:
        fold_fprs: List of FPR arrays for each fold
        fold_tprs: List of TPR arrays for each fold
        fold_aucs: List of AUC scores for each fold
        title: Title for the plot
        output_path: Path to save the image
    """
    n_splits = len(fold_fprs)
    
    # Set up colors for folds
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]
    
    fig, ax = plt.subplots(figsize=(8, 7))
    
    # Interpolate TPR for each fold at common FPR values
    mean_fpr = np.linspace(0, 1, 100)
    interp_tprs = []
    
    for idx, (fpr, tpr) in enumerate(zip(fold_fprs, fold_tprs)):
        # Plot individual fold ROC curves
        ax.plot(fpr, tpr, 
                alpha=0.3, lw=1.5, 
                color=colors[idx % len(colors)],
                label=f"ROC fold {idx} (AUC = {fold_aucs[idx]:.3f})")
        
        # Interpolate TPR at mean FPR points
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        interp_tprs.append(interp_tpr)
    
    # Calculate mean and std of interpolated TPRs
    mean_tpr = np.mean(interp_tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(fold_aucs)
    
    # Plot mean ROC curve
    ax.plot(
        mean_fpr,
        mean_tpr,
        color="b",
        label=f"Mean ROC (AUC = {mean_auc:.3f} ± {std_auc:.3f})",
        lw=2.5,
        alpha=0.9,
    )
    
    # Plot confidence band (±1 std)
    std_tpr = np.std(interp_tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(
        mean_fpr,
        tprs_lower,
        tprs_upper,
        color="grey",
        alpha=0.2,
        label="± 1 std. dev.",
    )
    
    # Plot random classifier line
    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier', alpha=0.5)
    
    # Formatting
    ax.set(
        xlim=[-0.02, 1.02],
        ylim=[-0.02, 1.02],
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title=f"{title}\n",
    )
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: {output_path.name}")
    print(f"    Mean AUC: {mean_auc:.4f} ± {std_auc:.4f}")
    print(f"    Number of folds: {len(fold_fprs)}")


def main():
    # Base paths
    base_path = Path(r"C:\Users\dario\Documents\Eu\TFM Álvaro\Dicoperia\metrics_auc_roc")
    output_base = base_path.parent / "roc_cv_plots"
    output_base.mkdir(exist_ok=True)
    
    # Get all moment-task folders
    moment_folders = sorted([d for d in base_path.iterdir() if d.is_dir()])
    
    print(f"Found {len(moment_folders)} moment-task folders\n")
    
    for moment_folder in moment_folders:
        moment_name = moment_folder.name
        roc_data_folder = moment_folder / "roc_data"
        
        if not roc_data_folder.exists():
            print(f"Skipping {moment_name}: roc_data folder not found")
            continue
        
        print(f"Processing {moment_name} (combined model: acoustic + wavlm + wav2vec)...")
        
        # Load fold data
        fold_data = load_fold_data(roc_data_folder)
        
        if not fold_data:
            print(f"  Skipping: No fold data found\n")
            continue
        
        # Generate title
        title = get_title_from_folder(moment_name)
        
        # Plot CV ROC curves
        output_path = output_base / f"{moment_name}_cv_roc_curves.png"
        plot_cv_roc_curves(
            fold_data['fold_fprs'],
            fold_data['fold_tprs'],
            fold_data['fold_aucs'],
            title,
            output_path
        )
        print()
    
    print(f"✓ ROC curve plotting complete!")
    print(f"  Plots saved to: {output_base}")


if __name__ == "__main__":
    main()
