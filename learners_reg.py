import numpy as np
import os

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error
from sklearn.utils.random import sample_without_replacement
import torch
import torch.nn.functional as F
from tqdm import tqdm


torch.set_default_dtype(torch.float64)
torch.manual_seed(0)
device = torch.device('cpu')

EPOCHS = 100

def compute_model_metrics(model, xs_train, treats_train, outcomes_train, xs_test, 
                         ctrl_test, trt_test, ctrl_train, trt_train, n_train=None):
    """
    Compute comprehensive metrics for a trained model.
    
    Args:
        model: Trained PyTorch model
        xs_train: Training covariates
        treats_train: Training treatment assignments
        outcomes_train: Training outcomes
        xs_test: Test covariates
        ctrl_test: Test control outcomes
        trt_test: Test treatment outcomes
        ctrl_train: Training control outcomes
        trt_train: Training treatment outcomes
        n_train: Number of training samples to use (if None, use all)
    
    Returns:
        dict: Dictionary containing computed metrics
    """
    if n_train is None:
        n_train = len(xs_train)
    
    model.eval()
    
    with torch.no_grad():
        # Training predictions
        X_train = np.concatenate((xs_train[:n_train,:], np.expand_dims(treats_train[:n_train], axis=1)), axis=1)
        train_pred = model(torch.tensor(X_train, dtype=torch.float64)).numpy()
        
        # Training counterfactual predictions
        X_train_treated = np.concatenate((xs_train, np.ones((len(xs_train), 1))), axis=1)
        X_train_control = np.concatenate((xs_train, np.zeros((len(xs_train), 1))), axis=1)
        train_treated_pred = model(torch.tensor(X_train_treated, dtype=torch.float64)).numpy()
        train_control_pred = model(torch.tensor(X_train_control, dtype=torch.float64)).numpy()
        
        # Test counterfactual predictions
        X_test_treated = np.concatenate((xs_test, np.ones((len(xs_test), 1))), axis=1)
        X_test_control = np.concatenate((xs_test, np.zeros((len(xs_test), 1))), axis=1)
        test_treated_pred = model(torch.tensor(X_test_treated, dtype=torch.float64)).numpy()
        test_control_pred = model(torch.tensor(X_test_control, dtype=torch.float64)).numpy()
        
        # Compute MSE
        train_mse = mean_squared_error(outcomes_train[:n_train], train_pred)
        
        # Vectorized regret computation
        def hard_regret(xs, ctrl, trt, ctrl_pred, trt_pred):
            decisions = (trt_pred > ctrl_pred).astype(int)
            best_outcomes = np.maximum(ctrl, trt)
            actual_outcomes = np.where(decisions, trt, ctrl)
            regrets = best_outcomes - actual_outcomes
            return np.mean(regrets)
        
        # Compute regrets
        test_regret = hard_regret(xs_test, ctrl_test, trt_test, test_control_pred, test_treated_pred)
        train_regret = hard_regret(xs_train, ctrl_train, trt_train, train_control_pred, train_treated_pred)
        
        return {
            'train_mse': train_mse,
            'train_regret': train_regret,
            'test_regret': test_regret,
            'train_pred_mean': np.mean(train_pred),
            'train_pred_std': np.std(train_pred),
            'test_treated_pred_mean': np.mean(test_treated_pred),
            'test_treated_pred_std': np.std(test_treated_pred)
        }

class Net(torch.nn.Module):
    def __init__(self, dim): # dim should be the dimension of the covariates x
        super().__init__()
        self.lin = torch.nn.Linear(dim,5)
        # self.lin2 = torch.nn.Linear(dim,dim)
        self.lin3 = torch.nn.Linear(5,1)

    def forward(self, data):
        out = F.relu(self.lin(data))
        # out = F.relu(self.lin2(out))
        out = self.lin3(out)
        return out.view(-1) #+ out_f.view(-1)

def our_loss(y_hat, y_true, closest, k):
    closests_hat= y_hat[closest]
    closests_true = y_true[closest]

    return torch.dot(torch.abs(y_true - closests_true),F.sigmoid(-(y_hat - closests_hat)*torch.sign(y_true - closests_true)*k))

def calc_pairs(X_feat, treats):
    dist_matrix = torch.cdist(X_feat, X_feat)# , compute_mode = 'donot_use_mm_for_euclid_dist')
    closest = torch.zeros(X_feat.shape[0], dtype = torch.int32)
    dist_close = torch.zeros(X_feat.shape[0])
    for i in range(X_feat.shape[0]):
        dist_matrix[i,i] = torch.inf
        for j in range(X_feat.shape[0]):
            if treats[i] == treats[j]:
                dist_matrix[i,j] = torch.inf
        argmins = (dist_matrix[i] == torch.min(dist_matrix[i])).nonzero()
        closest[i] = argmins[np.random.choice(len(argmins))].item()
        dist_close[i] = torch.min(dist_matrix[i])

    return closest

def weighted_mse_loss(input, target, weight):
    return (weight * (input - target) ** 2).sum() / weight.sum()



#later, gradient boosting, or maybe lgb? or linear regression
def reg_learner(xs_train, treats_train, outcomes_train, xs_test, ctrl_test, trt_test, ctrl_train, trt_train, \
    k = 5, L = 0, n_epochs = EPOCHS, save_weights=False, model_save_dir="model_weights"):
    # ctrl_train and trt_train are not used
    n_train = len(outcomes_train)

    # concatenate the covariates (xs) with the treatment decision
    if len(np.shape(xs_train))==1:
        X = np.concatenate((np.expand_dims(xs_train, axis = 1), np.expand_dims(treats_train, axis = 1)), axis = 1)
    else:
        X = np.concatenate((xs_train, np.expand_dims(treats_train, axis = 1)), axis = 1)
    y = outcomes_train

    X = torch.tensor(X)
    y = torch.tensor(y)

    _, d = np.shape(xs_train)
    model = Net(d+1).to(device) # add one for the treatment decision
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Create directory for saving model weights if requested
    if save_weights:
        os.makedirs(model_save_dir, exist_ok=True)

    train_loss = np.zeros(n_epochs)
    train_reg = np.zeros(n_epochs)

    closest = calc_pairs(X[:,:-1], X[:,-1])

    def train(X,y, closest, L = 0):
        model.train()
        loss_all = 0
        X = X.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        # Weighted combination between our loss and MSE
        # By default we use our loss. To use MSE, set the weight L to 1.
        loss = (1-L)*our_loss(model(X), y, closest, k) + L*F.mse_loss(model(X), y)
        loss.backward()
        loss_all += loss.item()
        optimizer.step()
        return loss_all

    for epoch in range(n_epochs):
        print(f'Epoch={epoch}')
        train_loss[epoch] = train(X, y, closest, L = L)

        # Benchmark algorithm
        
        # Save model weights after each epoch if requested
        if save_weights:
            # Compute comprehensive metrics
            metrics = compute_model_metrics(
                model, xs_train, treats_train, outcomes_train, xs_test,
                ctrl_test, trt_test, ctrl_train, trt_train, n_train
            )
            
            model_path = os.path.join(model_save_dir, f"reg_learner_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'n_train' : n_train,
                'L' : L, # weight on MSE instead of. our loss
                'train_loss': train_loss[epoch],
                'k': k,
                # Add computed metrics to checkpoint
                'train_mse': metrics['train_mse'],
                'train_regret': metrics['train_regret'],
                'test_regret': metrics['test_regret'],
                'train_pred_mean': metrics['train_pred_mean'],
                'train_pred_std': metrics['train_pred_std'],
                'test_treated_pred_mean': metrics['test_treated_pred_mean'],
                'test_treated_pred_std': metrics['test_treated_pred_std']
            }, model_path)

    metrics = compute_model_metrics(
        model, xs_train, treats_train, outcomes_train, xs_test,
        ctrl_test, trt_test, ctrl_train, trt_train, n_train)

    return train_loss, metrics['test_regret']