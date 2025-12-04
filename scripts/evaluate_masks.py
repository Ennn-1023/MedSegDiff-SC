"""
Evaluate individual masks against ground truth
Compare mask0 ~ mask4 and ensemble result with GT
"""

import os
import argparse
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import pandas as pd


def load_mask(path, threshold=0.5):
    """Load mask and convert to binary"""
    img = Image.open(path).convert('L')  # Convert to grayscale
    img_array = np.array(img) / 255.0  # Normalize to [0, 1]
    binary_mask = (img_array > threshold).astype(np.uint8)
    return binary_mask


def calculate_metrics(pred, gt):
    """Calculate Dice, IoU, Precision, Recall"""
    pred = pred.flatten()
    gt = gt.flatten()

    # Intersection and Union
    intersection = np.sum(pred * gt)
    union = np.sum(pred) + np.sum(gt)

    # Dice Score
    dice = (2.0 * intersection) / (union + 1e-7)

    # IoU (Jaccard)
    iou = intersection / (np.sum(pred) + np.sum(gt) - intersection + 1e-7)

    # Precision and Recall
    tp = intersection
    fp = np.sum(pred * (1 - gt))
    fn = np.sum((1 - pred) * gt)

    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)

    # F1 Score (same as Dice for binary)
    f1 = 2 * precision * recall / (precision + recall + 1e-7)

    return {
        'Dice': dice,
        'IoU': iou,
        'Precision': precision,
        'Recall': recall,
        'F1': f1
    }


def evaluate_sample(output_dir, slice_id, num_ensemble=5, threshold=0.5):
    """Evaluate all masks for a single sample"""

    # Load GT
    gt_path = os.path.join(output_dir, f'{slice_id}_gt.jpg')
    if not os.path.exists(gt_path):
        print(f"Warning: GT not found for {slice_id}")
        return None

    gt = load_mask(gt_path, threshold)

    results = {'slice_id': slice_id}

    # Evaluate individual masks
    for i in range(num_ensemble):
        mask_path = os.path.join(output_dir, f'{slice_id}_mask{i}.jpg')
        if os.path.exists(mask_path):
            pred = load_mask(mask_path, threshold)
            metrics = calculate_metrics(pred, gt)
            for metric_name, value in metrics.items():
                results[f'mask{i}_{metric_name}'] = value
        else:
            print(f"Warning: mask{i} not found for {slice_id}")

    # Evaluate ensemble result
    ens_path = os.path.join(output_dir, f'{slice_id}_output_ens.jpg')
    if os.path.exists(ens_path):
        pred_ens = load_mask(ens_path, threshold)
        metrics_ens = calculate_metrics(pred_ens, gt)
        for metric_name, value in metrics_ens.items():
            results[f'ensemble_{metric_name}'] = value
    else:
        print(f"Warning: ensemble not found for {slice_id}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory containing generated masks')
    parser.add_argument('--num_ensemble', type=int, default=5,
                        help='Number of ensemble samples')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Threshold for binary mask')
    parser.add_argument('--save_csv', type=str, default='evaluation_results.csv',
                        help='Path to save results CSV')
    args = parser.parse_args()

    # Find all unique slice IDs
    all_files = os.listdir(args.output_dir)
    slice_ids = set()
    for f in all_files:
        if '_gt.jpg' in f:
            slice_id = f.replace('_gt.jpg', '')
            slice_ids.add(slice_id)

    print(f"Found {len(slice_ids)} samples to evaluate")

    # Evaluate all samples
    all_results = []
    for slice_id in sorted(slice_ids):
        print(f"Evaluating {slice_id}...")
        result = evaluate_sample(args.output_dir, slice_id, args.num_ensemble, args.threshold)
        if result:
            all_results.append(result)

    # Convert to DataFrame
    df = pd.DataFrame(all_results)

    # Calculate statistics
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)

    # Individual masks statistics
    print("\n📊 Individual Masks Performance:")
    for i in range(args.num_ensemble):
        if f'mask{i}_Dice' in df.columns:
            dice_mean = df[f'mask{i}_Dice'].mean()
            dice_std = df[f'mask{i}_Dice'].std()
            iou_mean = df[f'mask{i}_IoU'].mean()
            print(f"  Mask {i}: Dice={dice_mean:.4f}±{dice_std:.4f}, IoU={iou_mean:.4f}")

    # Ensemble statistics
    print("\n🎯 Ensemble Performance:")
    if 'ensemble_Dice' in df.columns:
        ens_dice_mean = df['ensemble_Dice'].mean()
        ens_dice_std = df['ensemble_Dice'].std()
        ens_iou_mean = df['ensemble_IoU'].mean()
        ens_prec_mean = df['ensemble_Precision'].mean()
        ens_rec_mean = df['ensemble_Recall'].mean()
        print(f"  Dice:      {ens_dice_mean:.4f}±{ens_dice_std:.4f}")
        print(f"  IoU:       {ens_iou_mean:.4f}")
        print(f"  Precision: {ens_prec_mean:.4f}")
        print(f"  Recall:    {ens_rec_mean:.4f}")

    # Compare ensemble with best individual
    print("\n📈 Comparison:")
    individual_dice_cols = [col for col in df.columns if col.startswith('mask') and col.endswith('_Dice')]
    if individual_dice_cols and 'ensemble_Dice' in df.columns:
        best_individual_mean = df[individual_dice_cols].mean().max()
        best_individual_idx = df[individual_dice_cols].mean().idxmax()
        print(f"  Best individual: {best_individual_idx.replace('_Dice', '')} = {best_individual_mean:.4f}")
        print(f"  Ensemble:        {ens_dice_mean:.4f}")
        improvement = ((ens_dice_mean - best_individual_mean) / best_individual_mean) * 100
        print(f"  Improvement:     {improvement:+.2f}%")

    # Save to CSV
    df.to_csv(args.save_csv, index=False)
    print(f"\n✅ Results saved to: {args.save_csv}")
    print("="*80)


if __name__ == '__main__':
    main()

