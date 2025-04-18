"""
Visualization utilities for model predictions.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from typing import List, Tuple, Optional

def plot_prediction_grid(
    model: torch.nn.Module,
    test_loader: DataLoader,
    class_names: List[str],
    device: torch.device,
    rows: int = 10,
    cols: int = 3,
    figsize: Tuple[int, int] = (10, 40),
    font_size_labels: int = 11,
    main_title_fontsize: int = 20
) -> None:
    """
    Visualize model predictions on a test dataset in a grid of images.

    Args:
        model (torch.nn.Module): Trained PyTorch model in evaluation mode.
        test_loader (DataLoader): DataLoader for the test dataset.
        class_names (List[str]): List of class names for label mapping.
        device (torch.device): Device to run inference (e.g., 'cuda' or 'cpu').
        rows (int, optional): Number of rows in the grid. Defaults to 10.
        cols (int, optional): Number of columns in the grid. Defaults to 3.
        figsize (Tuple[int, int], optional): Figure size (width, height). Defaults to (10, 12).
        font_size_labels (int, optional): Font size for true/predicted labels. Defaults to 11.
        main_title_fontsize (int, optional): Font size for the main title. Defaults to 20.

    Returns:
        None: Displays a matplotlib plot of the prediction grid.
    """
    # Calculate total samples to collect
    num_samples = rows * cols
    print(f"Attempting to collect {num_samples} samples for the grid...")

    # Collect samples for visualization
    images, true_labels, pred_labels = [], [], []
    model.eval()
    with torch.no_grad():
        for batch_idx, (batch_images, batch_labels) in enumerate(test_loader):
            print(f"Processing batch {batch_idx + 1}/{len(test_loader)}...")
            batch_images, batch_labels = batch_images.to(device), batch_labels.to(device)
            outputs = model(batch_images)
            _, preds = torch.max(outputs, 1)

            # Move to CPU for storage
            batch_images_cpu = batch_images.cpu()
            batch_labels_cpu = batch_labels.cpu()
            preds_cpu = preds.cpu()

            # Collect samples
            for img, true, pred in zip(batch_images_cpu, batch_labels_cpu, preds_cpu):
                if len(images) < num_samples:
                    images.append(img)
                    true_labels.append(true.item())
                    pred_labels.append(pred.item())
                else:
                    break
            if len(images) >= num_samples:
                print(f"Collected {len(images)} samples.")
                break
        else:
            print(f"Warning: Test loader exhausted. Collected {len(images)}/{num_samples} samples.")

    # Plot the grid if samples were collected
    if not images:
        print("No samples collected, cannot plot.")
        return

    # Create figure and axes
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = [axes] if rows * cols == 1 else axes.flatten()

    # Plot each sample
    for i, (img, true_idx, pred_idx) in enumerate(zip(images, true_labels, pred_labels)):
        ax = axes[i]
        # Convert tensor (C,H,W) to numpy (H,W,C) for display
        img_np = img.permute(1, 2, 0).numpy()
        img_np = np.clip(img_np, 0, 1)

        # Display image
        ax.imshow(img_np)
        ax.set_xticks([])
        ax.set_yticks([])

        # Add border
        for spine in ax.spines.values():
            spine.set_edgecolor('darkgrey')
            spine.set_linewidth(1.5)
            spine.set_visible(True)

        # Add true/predicted labels below image
        is_correct = pred_idx == true_idx
        symbol = "✔" if is_correct else "✘"
        color = 'forestgreen' if is_correct else 'crimson'
        label_text = f"True: {class_names[true_idx]}\nPred: {class_names[pred_idx]} {symbol}"
        ax.set_xlabel(label_text, color=color, fontsize=font_size_labels, labelpad=4)

    # Turn off unused axes
    for j in range(len(images), len(axes)):
        axes[j].axis('off')

    # Adjust layout and add title
    plt.tight_layout(pad=1.0, h_pad=2.0, w_pad=1.0)
    fig.suptitle("Model Predictions on Test Data", fontsize=main_title_fontsize, y=1.02, fontweight='bold')
    plt.subplots_adjust(top=0.94, bottom=0.05)
    plt.show()
    print("Plotting complete.")