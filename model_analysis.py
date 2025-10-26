#!/usr/bin/env python3
"""
Simple Model Analysis Tool

Loads saved model weights and prints training loss, regret, MSE, and test performance.
"""

import numpy as np
import torch
import os
import glob
import argparse
import cProfile
import pstats
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from learners_reg import Net, compute_model_metrics

device = 'cpu'

def plot_treatment_effects(checkpoint_path, data_path='informs.npz', n_train=None, figsize=(10, 8)):
    """
    Create a 2D plot showing treatment effect predictions for training data.
    
    Args:
        checkpoint_path: Path to the model checkpoint file
        data_path: Path to the data file (default: 'informs.npz')
        n_train: Number of training samples to use (if None, use all)
        figsize: Figure size tuple (width, height)
    """
    # Load data
    try:
        data = np.load(data_path)
        xs_train = data['xs_train']
        treats_train = data['treats_train']
        outcomes_train = data['outcomes_train']
        ctrl_train = data['ctrl_train']
        trt_train = data['trt_train']
        print(f"✓ Loaded data from '{data_path}'")
    except FileNotFoundError:
        print(f"✗ Data file '{data_path}' not found!")
        return
    
    # Load checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        print(f"✓ Loaded checkpoint from '{checkpoint_path}'")
    except FileNotFoundError:
        print(f"✗ Checkpoint file '{checkpoint_path}' not found!")
        return
    
    # Get training data size
    if n_train is None:
        n_train = checkpoint.get('n_train', len(xs_train))
    
    # Limit to specified number of training samples
    xs_train = xs_train[:n_train]
    treats_train = treats_train[:n_train]
    outcomes_train = outcomes_train[:n_train]
    true_treatment_effects = trt_train[:n_train] - ctrl_train[:n_train]
    
    # Create model and load weights
    _, d = np.shape(xs_train)
    model = Net(d+1).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✓ Using {n_train} training samples with {d} features")
    
    # Compute treatment effects
    with torch.no_grad():
        # Predictions for treatment (t=1) and control (t=0)
        X_treated = np.concatenate((xs_train, np.ones((len(xs_train), 1))), axis=1)
        X_control = np.concatenate((xs_train, np.zeros((len(xs_train), 1))), axis=1)
        
        treated_pred = model(torch.tensor(X_treated, dtype=torch.float64)).numpy()
        control_pred = model(torch.tensor(X_control, dtype=torch.float64)).numpy()

        # Treatment effect = treated - control
        pred_treatment_effects = treated_pred - control_pred

    treatment_effects = pred_treatment_effects

    # Create the plot
    plt.figure(figsize=figsize)
    
    # Create scatter plot with treatment effects as colors
    scatter = plt.scatter(xs_train[:, 0], xs_train[:, 1], 
                         c=treatment_effects, 
                         cmap='RdBu_r', 
                         s=50, 
                         alpha=0.7,
                         edgecolors='black', 
                         linewidth=0.5)
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Treatment Effect (Treated - Control)', fontsize=12)
    
    # Add labels and title
    plt.xlabel('Feature 1 (x₁)', fontsize=12)
    plt.ylabel('Feature 2 (x₂)', fontsize=12)
    plt.title(f'Treatment Effect Predictions\nModel: {os.path.basename(checkpoint_path)}', fontsize=14, fontweight='bold')
    
    # Add statistics text
    mean_effect = np.mean(treatment_effects)
    std_effect = np.std(treatment_effects)
    positive_effects = np.sum(treatment_effects > 0)
    total_effects = len(treatment_effects)
    
    stats_text = f'Mean Effect: {mean_effect:.3f}\nStd Effect: {std_effect:.3f}\nPositive: {positive_effects}/{total_effects}'
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Set equal aspect ratio
    plt.axis('equal')
    
    plt.tight_layout()
    plt.show()
    
    print(f"📊 Treatment Effect Statistics:")
    print(f"   Mean: {mean_effect:.6f}")
    print(f"   Std: {std_effect:.6f}")
    print(f"   Positive effects: {positive_effects}/{total_effects} ({100*positive_effects/total_effects:.1f}%)")
    print(f"   Range: [{np.min(treatment_effects):.6f}, {np.max(treatment_effects):.6f}]")
    
    return treatment_effects

def analyze_model_performance(model_save_dir="model_weights", csv=False):
    """Analyze performance of saved models."""
    
    if not os.path.exists(model_save_dir):
        print(f"Model directory '{model_save_dir}' not found!")
        return
    
    # Load test data
    try:
        data = np.load('informs.npz')
        xs_train = data['xs_train'] # below we'll truncate
        xs_test = data['xs_test'] # use all of it
        treats_train = data['treats_train']
        outcomes_train = data['outcomes_train']
        ctrl_test = data['ctrl_test']
        trt_test = data['trt_test']
        ctrl_train = data['ctrl_train']
        trt_train = data['trt_train']

        if not csv:
            print("✓ Loaded data from 'generated_data.npz'")
    except FileNotFoundError:
        print("✗ 'generated_data.npz' not found. Please run data generation first.")
        return
    
    # Find all model files
    model_files = glob.glob(os.path.join(model_save_dir, "*.pth"))
    
    if not model_files:
        print(f"✗ No model files found in '{model_save_dir}'!")
        return
    
    if not csv:
        print(f"✓ Found {len(model_files)} model files")
        print("=" * 80)

    if csv:
        # header
        print("epoch,train_loss,train_regret,test_regret,train_mse")

    
    # Analyze each model
    for model_file in sorted(model_files):
        filename = os.path.basename(model_file)
        if not csv:
            print(f"\nAnalyzing: {filename}")
            print("-" * 50)
        
        if True:
            # Load checkpoint
            checkpoint = torch.load(model_file, map_location='cpu', weights_only=False)
            
            # Print checkpoint info
            epoch = checkpoint.get('epoch', 'Unknown')
            n_train = checkpoint.get('n_train', 0)
            train_loss = checkpoint.get('train_loss', 0)
            k_param = checkpoint.get('k', 'N/A')
            
            if not csv:
                print(f"Epoch: {epoch}")
                print(f"Num Training Points: {n_train}")
                if k_param != 'N/A':
                    print(f"K Parameter: {k_param}")

            # Check if metrics are already saved in checkpoint
            if 'train_mse' in checkpoint and 'train_regret' in checkpoint and 'test_regret' in checkpoint:
                # Use pre-computed metrics from checkpoint
                train_mse = checkpoint['train_mse']
                train_regret = checkpoint['train_regret']
                test_regret = checkpoint['test_regret']
                
                if not csv:
                    print("✓ Using pre-computed metrics from checkpoint")
            else:
                # Compute metrics using the new function
                _, d = np.shape(xs_train)
                model = Net(d+1).to(device)
                model.load_state_dict(checkpoint['model_state_dict'])
                
                metrics = compute_model_metrics(
                    model, xs_train, treats_train, outcomes_train, xs_test,
                    ctrl_test, trt_test, ctrl_train, trt_train, n_train
                )
                
                train_mse = metrics['train_mse']
                train_regret = metrics['train_regret']
                test_regret = metrics['test_regret']
                
                if not csv:
                    print("✓ Computed metrics on-the-fly")
                
                # Print results
                if csv:
                    print(f"{epoch},{train_loss:.6f}, {train_regret:.6f}, {test_regret:.6f}, {train_mse:.6f}")
                else:
                    print(f"Training Loss: {train_loss:.6f}")
                    print(f"Train Regret: {train_regret:.6f}")
                    print(f"Test Regret: {test_regret:.6f}")
                    print(f"Train MSE: {train_mse:.6f}")

                # print(f"Training Pred Mean: {np.mean(train_pred):.6f}")
                # print(f"Training Pred Std:  {np.std(train_pred):.6f}")
                # print(f"Test Pred Mean:     {np.mean(test_treated_pred):.6f}")
                # print(f"Test Pred Std:      {np.std(test_treated_pred):.6f}")
                

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Analyze saved model weights and print performance metrics')
    parser.add_argument('directory', nargs='?', default='model_weights', 
                       help='Directory containing model files (default: model_weights)')
    parser.add_argument('--csv', '-c', action='store_true', 
                       help='Enable CSV output')
    parser.add_argument('--profile', '-p', action='store_true', 
                       help='Enable profiling and save results to profile_results.prof')
    parser.add_argument('--profile-output', default='profile_results.prof',
                       help='Output file for profiling results (default: profile_results.prof)')
    parser.add_argument('--plot-effects', 
                       help='Path to specific checkpoint file to plot treatment effects')
    parser.add_argument('--plot-n-train', type=int, default=None,
                       help='Number of training samples to use for plotting (default: from checkpoint)')
    
    args = parser.parse_args()
    
    if not args.csv:
        print("🔍 ESR Loss - Simple Model Analysis")
        print("=" * 50)
        print(f"Analyzing models in directory: {args.directory}")
    
    if args.profile:
        print(f"🔍 Profiling enabled - results will be saved to {args.profile_output}")
        profiler = cProfile.Profile()
        profiler.enable()
    
    analyze_model_performance(args.directory, args.csv)
    
    if args.profile:
        profiler.disable()
        profiler.dump_stats(args.profile_output)
        print(f"✓ Profiling results saved to {args.profile_output}")
        print("📊 Top 10 functions by cumulative time:")
        stats = pstats.Stats(profiler)
        stats.sort_stats('cumulative').print_stats(10)
    
    # Plot treatment effects if requested
    if args.plot_effects:
        print(f"\n📊 Creating treatment effects plot for: {args.plot_effects}")
        plot_treatment_effects(args.plot_effects, n_train=args.plot_n_train)

if __name__ == "__main__":
    main()
