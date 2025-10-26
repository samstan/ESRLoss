#!/usr/bin/env python3
"""
Simple Model Analysis Tool

Loads saved model weights and prints training loss, regret, MSE, and test performance.
"""

import numpy as np
import torch
import os
import glob
from sklearn.metrics import mean_squared_error
from learners_reg import Net

device = 'cpu'

def analyze_model_performance(model_save_dir="model_weights"):
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

        print("✓ Loaded data from 'generated_data.npz'")
    except FileNotFoundError:
        print("✗ 'generated_data.npz' not found. Please run data generation first.")
        return
    
    # Find all model files
    model_files = glob.glob(os.path.join(model_save_dir, "*.pth"))
    
    if not model_files:
        print(f"✗ No model files found in '{model_save_dir}'!")
        return
    
    print(f"✓ Found {len(model_files)} model files")
    print("=" * 80)
    
    # Analyze each model
    for model_file in sorted(model_files):
        filename = os.path.basename(model_file)
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
            
            print(f"Epoch: {epoch}")
            print(f"Num Training Points: {n_train}")
            if k_param != 'N/A':
                print(f"K Parameter: {k_param}")

            # Create model and load weights
            _, d = np.shape(xs_train)
            model = Net(d+1).to(device)
            
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            # Compute metrics
            with torch.no_grad():
                # Training predictions
                X_train = np.concatenate((xs_train[:n_train,:], np.expand_dims(treats_train[:n_train], axis=1)), axis=1)
                train_pred = model(torch.tensor(X_train, dtype=torch.float64)).numpy()
                X_train_treated = np.concatenate((xs_test, np.ones((len(xs_test), 1))), axis=1)
                X_train_control = np.concatenate((xs_test, np.zeros((len(xs_test), 1))), axis=1)
                train_treated_pred = model(torch.tensor(X_train_treated, dtype=torch.float64)).numpy()
                train_control_pred = model(torch.tensor(X_train_control, dtype=torch.float64)).numpy()
                
                # Test predictions
                X_test_treated = np.concatenate((xs_test, np.ones((len(xs_test), 1))), axis=1)
                X_test_control = np.concatenate((xs_test, np.zeros((len(xs_test), 1))), axis=1)
                test_treated_pred = model(torch.tensor(X_test_treated, dtype=torch.float64)).numpy()
                test_control_pred = model(torch.tensor(X_test_control, dtype=torch.float64)).numpy()
                
                # Compute metrics
                train_mse = mean_squared_error(outcomes_train[:n_train], train_pred)

                def hard_regret(xs, ctrl, trt, ctrl_pred, trt_pred):
                    regret = 0 
                    for i in range(len(xs)):
                        if trt_pred[i] > ctrl_pred[i]:
                            decision = 1
                        else:
                            decision = 0
                        opts = [ctrl[i], trt[i]] # outcomes
                        regret += np.max(opts) - opts[decision]
                    return regret / len(xs)
                 
                # Compute test regret
                test_regret = hard_regret(xs_test, ctrl_test, trt_test, test_control_pred, test_treated_pred)
                train_regret = hard_regret(xs_train, ctrl_train, trt_train, train_control_pred, train_treated_pred)
                
                # Print results
                print(f"Training Loss: {train_loss:.6f}")
                print(f"Train Regret: {train_regret:.6f}")
                print(f"Test Regret: {test_regret:.6f}")
                print(f"Train MSE: {train_mse:.6f}")
                print(f"{train_loss:.6f}, {train_regret:.6f}, {test_regret:.6f}, {train_mse:.6f}")

                # print(f"Training Pred Mean: {np.mean(train_pred):.6f}")
                # print(f"Training Pred Std:  {np.std(train_pred):.6f}")
                # print(f"Test Pred Mean:     {np.mean(test_treated_pred):.6f}")
                # print(f"Test Pred Std:      {np.std(test_treated_pred):.6f}")
                

def main():
    """Main function."""
    print("🔍 ESR Loss - Simple Model Analysis")
    print("=" * 50)
    analyze_model_performance("XP6")

if __name__ == "__main__":
    main()
