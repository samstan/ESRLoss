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
    n_test = len(xs_test)

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
        
        # Save model weights after each epoch if requested
        if save_weights:
            model_path = os.path.join(model_save_dir, f"reg_learner_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'n_train' : n_train,
                'L' : L, # weight on MSE instead of. our loss
                'train_loss': train_loss[epoch],
                'k': k
            }, model_path)

        # reg_train = 0
        # for i in range(n_train):
        #     treated = model(torch.tensor(np.concatenate((xs_train[i], np.array([1]))))).detach().numpy()
        #     control = model(torch.tensor(np.concatenate((xs_train[i], np.array([0]))))).detach().numpy()

        #     decision = np.argmax([control,treated])
        #     opts = [ctrl_train[i],trt_train[i]]
        #     reg_train += np.max(opts) - opts[decision] 
        # train_reg[epoch] = reg_train


    if len(np.shape(xs_test))==1:
        X_test = np.expand_dims(xs_test, axis = 1)
    else:
        X_test = xs_test
    
    reg = 0 # regret
    for i in range(n_test):
        treated = model(torch.tensor(np.concatenate((X_test[i], np.array([1]))))).detach().numpy()
        control = model(torch.tensor(np.concatenate((X_test[i], np.array([0]))))).detach().numpy()

        probs_trt = np.exp(k*treated)[0]
        probs_ctr = np.exp(k*control)[0]

        decision = np.random.binomial(1, probs_trt/(probs_trt+probs_ctr))

        # print(probs_trt, probs_ctr, decision)

        # decision = np.argmax([control,treated])
        opts = [ctrl_test[i],trt_test[i]]
        reg += np.max(opts) - opts[decision]
        
        
    
    return train_loss, reg/n_test