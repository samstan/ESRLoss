#!/usr/bin/env python3
import numpy as np
from learners_reg import reg_learner

MAX_TRAIN = int(1e5) # max amount of training data
MAX_TEST = int(1e5) # max amount of test data
N_FEATURES = 2

def reward(xs, is_treated, a=0.1):
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
    # L is the weight on mse 

    assert(n_train <= MAX_TRAIN)
    assert(n_test <= MAX_TEST)

    # load data saved to a file by generate_data
    try:
        data = np.load('informs.npz')
        print(f"data loaded from 'informs.npz'")
    except filenotfounderror:
        print("data file not found. running generate_data() first...")
        data = generate_data()
    
    # extract and limit data to specified sizes

    # pick a random subset of our data for training
    xs_train = data['xs_train'][:n_train]


    xs_test = data['xs_test'][:n_test]
    treats_train = data['treats_train'][:n_train]
    outcomes_train = data['outcomes_train'][:n_train]
    ctrl_train = data['ctrl_train'][:n_train]
    trt_train = data['trt_train'][:n_train]
    ctrl_test = data['ctrl_test'][:n_test]
    trt_test = data['trt_test'][:n_test]
    
    print(f"using {n_train} training samples and {n_test} test samples")
    print(f"training data shape: {xs_train.shape}")
    print(f"test data shape: {xs_test.shape}")
   
    # example with reg_learner - save weights to default directory
    train_loss, regret = reg_learner(
        xs_train, treats_train, outcomes_train, 
        xs_test, ctrl_test, trt_test, 
        ctrl_train, trt_train, 
        k=25, 
        L=L,
        n_epochs = n_epochs,
        save_weights=True,  # enable weight saving
        model_save_dir=model_save_dir
    )
    print(f"training completed. final regret: {regret:.4f}")
    print(f"model weights saved to '{model_save_dir}/' directory")
    print()

def run_experiment_avg(n_train=10, n_test=MAX_TEST, n_epochs=2, l=0, num_replications=5):
    """
    Run experiment with multiple replications and compute mean and standard deviation of regret.
    
    Args:
        n_train: Number of training samples
        n_test: Number of test samples  
        n_epochs: Number of training epochs
        l: Weight on MSE loss
        num_replications: Number of replications to run
    
    Returns:
        tuple: (mean_regret, std_regret)
    """
    assert(n_train <= MAX_TRAIN)
    assert(n_test <= MAX_TEST)

    # load data saved to a file by generate_data
    try:
        data = np.load('informs.npz')
        print(f"data loaded from 'informs.npz'")
    except FileNotFoundError:
        print("data file not found. running generate_data() first...")
        data = generate_data()

    # Extract test data (same for all replications)
    xs_test = data['xs_test'][:n_test]
    ctrl_test = data['ctrl_test'][:n_test]
    trt_test = data['trt_test'][:n_test]
    
    print(f"Running {num_replications} replications with {n_train} training samples and {n_test} test samples")
    print(f"L (MSE weight): {l}, Epochs: {n_epochs}")
    print("=" * 60)
    
    regrets = []
    
    for rep in range(num_replications):
        # print(f"Replication {rep + 1}/{num_replications}...")
        
        # Pick a random subset of our data for training
        xs_train_idx = np.random.choice(range(MAX_TRAIN), size=n_train, replace=False)
        xs_train = data['xs_train'][xs_train_idx]
        treats_train = data['treats_train'][xs_train_idx]
        outcomes_train = data['outcomes_train'][xs_train_idx]
        ctrl_train = data['ctrl_train'][xs_train_idx]
        trt_train = data['trt_train'][xs_train_idx]

        train_loss, regret = reg_learner(
            xs_train, treats_train, outcomes_train, 
            xs_test, ctrl_test, trt_test, 
            ctrl_train, trt_train, 
            k=25, 
            L=l,
            n_epochs=n_epochs,
            save_weights=False
        )
        
        regrets.append(regret)
        # print(f"  Regret: {regret:.6f}")
    
    # Calculate statistics
    mean_regret = np.mean(regrets)
    std_regret = np.std(regrets, ddof=1)  # Sample standard deviation
    
    print("=" * 60)
    print(f"Results after {num_replications} replications:")
    print(f"Mean regret: {mean_regret:.6f}")
    print(f"Standard deviation: {std_regret:.6f}")
    # print(f"Individual regrets: {[f'{r:.6f}' for r in regrets]}")
    print()
    
    return mean_regret, std_regret
 
    

def run_experiment1():
    run_experiment(n_train=10,n_epochs=100,l=0,model_save_dir="xp01")
    run_experiment(n_train=10,n_epochs=100,l=0.05,model_save_dir="xp02")
    run_experiment(n_train=10,n_epochs=100,l=0.1,model_save_dir="xp03")
    run_experiment(n_train=10,n_epochs=100,l=0.15,model_save_dir="xp04")
    run_experiment(n_train=10,n_epochs=100,l=0.2,model_save_dir="xp05")
    run_experiment(n_train=10,n_epochs=100,l=0.25,model_save_dir="xp06")
    run_experiment(n_train=10,n_epochs=100,l=0.3,model_save_dir="xp07")
    run_experiment(n_train=10,n_epochs=100,l=0.35,model_save_dir="xp08")

    run_experiment(n_train=10,n_epochs=100,l=0.4,model_save_dir="xp09")
    run_experiment(n_train=10,n_epochs=100,l=0.45,model_save_dir="xp10")
    run_experiment(n_train=10,n_epochs=100,l=0.5,model_save_dir="xp11")
    run_experiment(n_train=10,n_epochs=100,l=0.55,model_save_dir="xp12")
    run_experiment(n_train=10,n_epochs=100,l=0.6,model_save_dir="xp13")
    run_experiment(n_train=10,n_epochs=100,l=0.65,model_save_dir="xp14")

    run_experiment(n_train=10,n_epochs=100,l=0.7,model_save_dir="xp15")
    run_experiment(n_train=10,n_epochs=100,l=0.75,model_save_dir="xp16")
    run_experiment(n_train=10,n_epochs=100,l=0.8,model_save_dir="xp17")
    run_experiment(n_train=10,n_epochs=100,l=0.85,model_save_dir="xp18")
    run_experiment(n_train=10,n_epochs=100,l=0.9,model_save_dir="xp19")
    run_experiment(n_train=10,n_epochs=100,L=0.95,model_save_dir="XP20")
    run_experiment(n_train=10,n_epochs=100,L=1.0,model_save_dir="XP21")


def run_experiment2():
    print('L=0')
    mean_regret, std_regret = run_experiment_avg(n_train=10, n_epochs=100, l=0, num_replications=100)
    print(f"Final result: {mean_regret:.6f} ± {std_regret:.6f}")

    print('\nL=0.025')
    mean_regret, std_regret = run_experiment_avg(n_train=10, n_epochs=100, l=0.025, num_replications=100)
    print(f"Final result: {mean_regret:.6f} ± {std_regret:.6f}")

    print('\nL=0.050')
    mean_regret, std_regret = run_experiment_avg(n_train=10, n_epochs=100, l=0.05, num_replications=100)
    print(f"Final result: {mean_regret:.6f} ± {std_regret:.6f}")

    print('\nL=1.00')
    mean_regret, std_regret = run_experiment_avg(n_train=10, n_epochs=100, l=1, num_replications=100)
    print(f"Final result: {mean_regret:.6f} ± {std_regret:.6f}")


if __name__ == "__main__":
    run_experiment(n_train=100,n_epochs=100,L=0,model_save_dir="v2-XP1")

    #run_experiment(n_train=1000,n_epochs=100,L=0,model_save_dir="XP6")



    # L=0
    # n=10   0.316106
    # n=100  0.312343
    # n=1000 0.303147

    # n=10
    # L=0