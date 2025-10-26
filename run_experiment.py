#!/usr/bin/env python3
import numpy as np
from learners_reg import reg_learner

MAX_TRAIN = int(1e5) # max amount of training data
MAX_TEST = int(1e5) # max amount of test data
N_FEATURES = 2

def reward(xs, is_treated, a=5):
    # xs is a 2-d numpy array with shape (n, d), where n is the number of samples and d is the number of features
    # is_treated is a 1-d numpy array with shape (n,), indicating whether the sample is treated or not
    # returns a 1-d numpy array with shape (n,), indicating the reward for the sample
    
    n, d = xs.shape
    
    reward = np.zeros(n)
    
    # Add quadratic terms (all pairwise products)
    for i in range(n):
        x = xs[i,:]
        sum_x = np.sum(x)
        reward[i] = np.cos(1+a*sum_x) + np.sin(a*sum_x) * is_treated[i]

    return reward

def generate_data():
    # Generate some sample data
    np.random.seed(42)
    
    # Features are normally distributed.
    xs_train = np.random.randn(MAX_TRAIN, N_FEATURES)
    xs_test = np.random.randn(MAX_TEST, N_FEATURES)

    # Training data treatments and outcomes
    frac_treated = 0.5 # fraction treated in training data
    ctrl_train = reward(xs_train, np.zeros(MAX_TRAIN)) # not used
    trt_train = reward(xs_train, np.ones(MAX_TRAIN)) # not used
    treats_train = np.random.binomial(1, frac_treated, MAX_TRAIN)
    outcomes_train = reward(xs_train, treats_train)
    
    # Test data outcomes for control and treatment
    ctrl_test = reward(xs_test, np.zeros(MAX_TEST))
    trt_test = reward(xs_test, np.ones(MAX_TEST))

    # Save data to a file
    data_dict = {
        'xs_train': xs_train,
        'xs_test': xs_test,
        'treats_train': treats_train,
        'outcomes_train': outcomes_train,
        'ctrl_train': ctrl_train,
        'trt_train': trt_train,
        'ctrl_test': ctrl_test,
        'trt_test': trt_test
    }
    
    np.savez('informs.npz', **data_dict)
    print(f"Data saved to 'informs.npz'")
    print(f"Training data shape: {xs_train.shape}")
    print(f"Test data shape: {xs_test.shape}")
    return data_dict


 
def run_experiment(n_train = 10, n_test = MAX_TEST, n_epochs = 2, L = 0, model_save_dir="model_weights"):
    # L is the weight on MSE 

    assert(n_train <= MAX_TRAIN)
    assert(n_test <= MAX_TEST)

    # Load data saved to a file by generate_data
    try:
        data = np.load('informs.npz')
        print(f"Data loaded from 'informs.npz'")
    except FileNotFoundError:
        print("Data file not found. Running generate_data() first...")
        data = generate_data()
    
    # Extract and limit data to specified sizes
    xs_train = data['xs_train'][:n_train]
    xs_test = data['xs_test'][:n_test]
    treats_train = data['treats_train'][:n_train]
    outcomes_train = data['outcomes_train'][:n_train]
    ctrl_train = data['ctrl_train'][:n_train]
    trt_train = data['trt_train'][:n_train]
    ctrl_test = data['ctrl_test'][:n_test]
    trt_test = data['trt_test'][:n_test]
    
    print(f"Using {n_train} training samples and {n_test} test samples")
    print(f"Training data shape: {xs_train.shape}")
    print(f"Test data shape: {xs_test.shape}")
   
    # Example with reg_learner - save weights to default directory
    train_loss, regret = reg_learner(
        xs_train, treats_train, outcomes_train, 
        xs_test, ctrl_test, trt_test, 
        ctrl_train, trt_train, 
        k=5, 
        L=L,
        n_epochs = n_epochs,
        save_weights=True,  # Enable weight saving
        model_save_dir=model_save_dir
    )
    print(f"Training completed. Final regret: {regret:.4f}")
    print(f"Model weights saved to '{model_save_dir}/' directory")
    print()
    
if __name__ == "__main__":
    # run_experiment(n_train=10,n_epochs=100,L=0,model_save_dir="our_loss")
    #run_experiment(n_train=10,n_epochs=100,L=0.03,model_save_dir="XP2")
    #run_experiment(n_train=10,n_epochs=100,L=0.1,model_save_dir="XP3")
    #run_experiment(n_train=10,n_epochs=100,L=0.5,model_save_dir="XP4")
    #run_experiment(n_train=100,n_epochs=100,L=0,model_save_dir="XP5")
    run_experiment(n_train=1000,n_epochs=100,L=0,model_save_dir="XP6")
    # run_experiment(n_train=1,n_epochs=100,L=0,model_save_dir="XP7")