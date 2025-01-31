import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error
from sklearn.utils.random import sample_without_replacement
from sklearn.neighbors import NearestNeighbors
import torch
import torch.nn.functional as F
from tqdm import tqdm


torch.set_default_dtype(torch.float64)
torch.manual_seed(0)
device = torch.device('cpu')

EPOCHS = 100

class Net(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.lin = torch.nn.Linear(dim,dim//2)
        self.lin3 = torch.nn.Linear(dim//2,1)

    def forward(self, data):
        out = F.relu(self.lin(data))
        out = self.lin3(out)
        return out.view(-1)

class ClassNet(torch.nn.Module):
    def __init__(self, dim, num_actions):
        super().__init__()
        self.lin = torch.nn.Linear(dim,dim//2)
        self.lin3 = torch.nn.Linear(dim//2,num_actions)
        self.softmax = torch.nn.Softmax(1)

    def forward(self, data):
        out = F.relu(self.lin(data))
        out = self.lin3(out)
        out = self.softmax(out)
        return out

def our_loss(y_hat, y_true, closest, k):
    return torch.mean(torch.max(y_true[closest.long()],dim = 1)[0] - torch.sum(y_true[closest.long()] * F.softmax(k*y_hat[closest.long()], dim = 1), dim = 1))

def bandit_net_loss(output, action, delta, prop, lamb):
    risk =  1.0 - delta
    loss = (risk - lamb) * (output[np.indices(action.shape)[0], action.long()] / prop)
    return torch.mean(loss)

def pl_loss(output, action, delta, prop, base_policy, beta):
    risk =  1.0 - delta
    loss = risk * (output[np.indices(action.shape)[0], action.long()]) / prop
    to_add = beta * torch.sum(torch.div(output, base_policy)) / output.shape[0]
    return torch.mean(loss) + to_add

def ls_loss(output, action, delta, prop, lamb):
    risk = -1.0 * delta #costs are assumed negative
    loss = risk * (output[np.indices(action.shape)[0], action.long()]) / prop
    loss = torch.log( 1 - lamb * loss)/lamb
    return -1.0 * torch.mean(loss)

def lse_loss(output, action, delta, prop, lamb):
    risk = 1.0 - delta
    loss = risk * (output[np.indices(action.shape)[0], action.long()]) / prop
    loss = torch.exp(lamb * loss)
    return torch.log(torch.mean(loss))/lamb

def dr_shrink_loss(output, action, delta, prop, eta, lamb):
    risk = 1.0 - delta
    loss = torch.sum(torch.mul(output, eta))
    corr = risk - eta[np.indices(action.shape)[0], action.long()]
    w = (output[np.indices(action.shape)[0], action.long()]) / prop
    if lamb < torch.inf:
        loss += torch.sum(torch.mul(corr, torch.mul(w, lamb/(w**2 + lamb))))
    else:
        loss += torch.sum(torch.mul(corr, w))
    return loss/output.shape[0]

def calc_pairs(X_feat, treats):
    dist_matrix = torch.cdist(X_feat, X_feat)
    closest = -1 * torch.ones((X_feat.shape[0], 20), dtype = torch.int32)
    for i in range(X_feat.shape[0]):
        for j in range(20):
            idxs = torch.where(treats == j)
            dists = dist_matrix[i][idxs]
            closest[i,j] = idxs[0][torch.argmin(dists)]

    return closest

def calc_pairs_ind(X_feat, treats):
    closest = -1 * torch.ones((X_feat.shape[0], 20), dtype = torch.int32)
    for i in range(X_feat.shape[0]):
        dists_from_i = torch.cdist(X_feat[i].unsqueeze(0), X_feat)[0]
        for j in range(20):
            idxs = torch.where(treats == j)
            dists = dists_from_i[idxs]
            closest[i,j] = idxs[0][torch.argmin(dists)]

    return closest

def calc_pairs_approx(X_feat, treats):
    unique_treats = treats.unique()
    closest = -1 * np.ones((X_feat.shape[0], len(unique_treats)), dtype = np.int32)
    idxs = []
    Xs = []
    nbrs = []
    for i in unique_treats:
        idx = torch.where(treats == i)
        idxs.append(idx)
        Xi = X_feat[idx]
        Xs.append(Xi)
        nbr = NearestNeighbors(n_neighbors = 1, algorithm = 'kd_tree').fit(Xi)
        nbrs.append(nbr)
    for i in range(len(unique_treats)):
        _, idxs = nbrs[i].kneighbors(X_feat)
        closest[:,i] = idxs.flatten()
    return torch.tensor(closest)

def weighted_mse_loss(input, target, weight):
    return (weight * (input - target) ** 2).sum() / weight.sum()

def reg_learner(xs_train, treats_train, outcomes_train, df_test, df_test_all, k = 5, schedule_k = 'constant'):
    n_train = len(outcomes_train)

    if len(np.shape(xs_train))==1:
        X = np.concatenate((np.expand_dims(xs_train, axis = 1), np.expand_dims(treats_train, axis = 1)), axis = 1)
    else:
        X = np.concatenate((xs_train, np.expand_dims(treats_train, axis = 1)), axis = 1)
    y = outcomes_train

    X = torch.tensor(X)
    y = torch.tensor(y)
    model = Net(6).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)


    train_loss = np.zeros(EPOCHS)
    train_reg = np.zeros(EPOCHS)

    closest = calc_pairs_approx(X[:,:-1], X[:,-1])

    if schedule_k != 'constant':
        ks = k + np.arange(EPOCHS)
    else:
        ks = k * np.ones(EPOCHS)

    def train(X,y, closest, k):
        model.train()
        loss_all = 0
        X = X.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        loss = our_loss(model(X), y, closest, k)
        loss.backward()
        loss_all += loss.item()
        optimizer.step()
        return loss_all

    for epoch in range(EPOCHS):
        train_loss[epoch] = train(X, y, closest, ks[epoch])

    def off_policy_eval(df_test, df_test_all):
        df_arg = df_test_all.copy()
        X_arg = torch.tensor(df_arg.values)
        X_arg = X_arg.to(device)
        y_arg = model(X_arg)
        df_arg.insert(len(df_arg.columns), "Clicked_hat", y_arg.detach().cpu().numpy())
        df_arg_pred = df_arg.sort_values('Clicked_hat', ascending=False).drop_duplicates(['Feat 1', 'Feat 2', 'Feat 3', 'Feat 4', 'Feat 5'])
        df_res = df_test.merge(df_arg_pred, on = list(df_arg.columns[:-1]), how = 'inner')
        return df_res.Clicked.mean()
        
    
    return off_policy_eval(df_test, df_test_all)

def s_learner(xs_train, treats_train, outcomes_train, df_test, df_test_all):
    n_train = len(outcomes_train)

    if len(np.shape(xs_train))==1:
        X = np.concatenate((np.expand_dims(xs_train, axis = 1), np.expand_dims(treats_train, axis = 1)), axis = 1)
    else:
        X = np.concatenate((xs_train, np.expand_dims(treats_train, axis = 1)), axis = 1)
    y = outcomes_train

    X = torch.tensor(X)
    y = torch.tensor(y)
    model = Net(6).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)


    def train(X,y):
        model.train()
        loss_all = 0
        X = X.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        loss = F.mse_loss(model(X), y)
        loss.backward()
        loss_all += loss.item()
        optimizer.step()
        return loss_all

    train_loss = np.zeros(EPOCHS)

    for epoch in range(EPOCHS):
        train_loss[epoch] = train(X, y)

    def off_policy_eval(df_test, df_test_all):
        df_arg = df_test_all.copy()
        X_arg = torch.tensor(df_arg.values)
        X_arg = X_arg.to(device)
        y_arg = model(X_arg)
        df_arg.insert(len(df_arg.columns), "Clicked_hat", y_arg.detach().cpu().numpy())
        df_arg_pred = df_arg.sort_values('Clicked_hat', ascending=False).drop_duplicates(['Feat 1', 'Feat 2', 'Feat 3', 'Feat 4', 'Feat 5'])
        df_res = df_test.merge(df_arg_pred, on = list(df_arg.columns[:-1]), how = 'inner')
        return df_res.Clicked.mean()
        
    
    return off_policy_eval(df_test, df_test_all)

def bandit_net(xs_train, treats_train, outcomes_train, df_test, lamb = 0.9):
    n_train = len(outcomes_train)
    num_actions = len(treats_train.unique())
    model = ClassNet(5, num_actions).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    X = torch.tensor(xs_train)
    y = torch.tensor(outcomes_train)
    treats = torch.tensor(treats_train)
    props = 1/num_actions * torch.ones_like(treats_train)

    def train(X,y, treats, props):
        model.train()
        loss_all = 0
        X = X.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        loss = bandit_net_loss(model(X), treats, y, props, lamb)
        loss.backward()
        loss_all += loss.item()
        optimizer.step()
        return loss_all

    train_loss = np.zeros(EPOCHS)

    for epoch in range(EPOCHS):
        train_loss[epoch] = train(X, y, treats, props)

    def off_policy_eval(df_test):
        X_test = df_test.drop(['Clicked', 'Treat'], axis = 1)
        treats_pred = model(torch.tensor(X_test.values))
        treats_test = torch.tensor(df_test.Treat.astype(np.int32).values)
        y_test = torch.tensor(df_test.Clicked.astype(np.int32).values)
        policy = treats_pred[np.indices(treats_test.shape)[0], treats_test.long()]
        return torch.mean(torch.mul(policy,y_test)*num_actions)
        
    return off_policy_eval(df_test)


def pseudo_loss(xs_train, treats_train, outcomes_train, df_test, beta = 1e-3):
    n_train = len(outcomes_train)
    num_actions = len(treats_train.unique())
    model = ClassNet(5, num_actions).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    X = torch.tensor(xs_train)
    y = torch.tensor(outcomes_train)
    treats = torch.tensor(treats_train)
    props = 1/num_actions * torch.ones_like(treats_train)
    base_policy = 1/num_actions * torch.ones((len(X), num_actions))

    def train(X,y, treats, props, base_policy):
        model.train()
        loss_all = 0
        X = X.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        loss = pl_loss(model(X), treats, y, props, base_policy, beta)
        loss.backward()
        loss_all += loss.item()
        optimizer.step()
        return loss_all

    train_loss = np.zeros(EPOCHS)

    for epoch in range(EPOCHS):
        train_loss[epoch] = train(X, y, treats, props, base_policy)

    def off_policy_eval(df_test):
        X_test = df_test.drop(['Clicked', 'Treat'], axis = 1)
        treats_pred = model(torch.tensor(X_test.values))
        treats_test = torch.tensor(df_test.Treat.astype(np.int32).values)
        y_test = torch.tensor(df_test.Clicked.astype(np.int32).values)
        policy = treats_pred[np.indices(treats_test.shape)[0], treats_test.long()]
        return torch.mean(torch.mul(policy,y_test)*num_actions)
        
    return off_policy_eval(df_test)

def log_smooth(xs_train, treats_train, outcomes_train, df_test, lamb = 1):
    n_train = len(outcomes_train)
    num_actions = len(treats_train.unique())
    model = ClassNet(5, num_actions).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    X = torch.tensor(xs_train)
    y = torch.tensor(outcomes_train)
    treats = torch.tensor(treats_train)
    props = 1/num_actions * torch.ones_like(treats_train)

    def train(X,y, treats, props):
        model.train()
        loss_all = 0
        X = X.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        loss = ls_loss(model(X), treats, y, props, lamb)
        loss.backward()
        loss_all += loss.item()
        optimizer.step()
        return loss_all

    train_loss = np.zeros(EPOCHS)

    for epoch in tqdm(range(EPOCHS)):
        train_loss[epoch] = train(X, y, treats, props)

    def off_policy_eval(df_test):
        X_test = df_test.drop(['Clicked', 'Treat'], axis = 1)
        treats_pred = model(torch.tensor(X_test.values))
        treats_test = torch.tensor(df_test.Treat.astype(np.int32).values)
        y_test = torch.tensor(df_test.Clicked.astype(np.int32).values)
        policy = treats_pred[np.indices(treats_test.shape)[0], treats_test.long()]
        return torch.mean(torch.mul(policy,y_test)*num_actions)
        
    return off_policy_eval(df_test)

def log_sum_exp(xs_train, treats_train, outcomes_train, df_test, lamb = 1):
    n_train = len(outcomes_train)
    num_actions = len(treats_train.unique())
    model = ClassNet(5, num_actions).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    X = torch.tensor(xs_train)
    y = torch.tensor(outcomes_train)
    treats = torch.tensor(treats_train)
    props = 1/num_actions * torch.ones_like(treats_train)

    def train(X,y, treats, props):
        model.train()
        loss_all = 0
        X = X.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        loss = lse_loss(model(X), treats, y, props, lamb)
        loss.backward()
        loss_all += loss.item()
        optimizer.step()
        return loss_all

    train_loss = np.zeros(EPOCHS)

    for epoch in tqdm(range(EPOCHS)):
        train_loss[epoch] = train(X, y, treats, props)

    def off_policy_eval(df_test):
        X_test = df_test.drop(['Clicked', 'Treat'], axis = 1)
        treats_pred = model(torch.tensor(X_test.values))
        treats_test = torch.tensor(df_test.Treat.astype(np.int32).values)
        y_test = torch.tensor(df_test.Clicked.astype(np.int32).values)
        policy = treats_pred[np.indices(treats_test.shape)[0], treats_test.long()]
        return torch.mean(torch.mul(policy,y_test)*num_actions)
        
    return off_policy_eval(df_test)

def dr_shrink(xs_train, treats_train, outcomes_train, df_test, lamb=1):

    n_train = len(outcomes_train)
    num_actions = len(treats_train.unique())
    if len(np.shape(xs_train))==1:
        X = np.concatenate((np.expand_dims(xs_train, axis = 1), np.expand_dims(treats_train, axis = 1)), axis = 1)
    else:
        X = np.concatenate((xs_train, np.expand_dims(treats_train, axis = 1)), axis = 1)
    y = outcomes_train

    X = torch.tensor(X)
    y = torch.tensor(y)
    model_reward = Net(6).to(device)
    optimizer_reward = torch.optim.Adam(model_reward.parameters(), lr=0.1)



    def train_reward(X,y):
        model_reward.train()
        loss_all = 0
        X = X.to(device)
        y = y.to(device)
        optimizer_reward.zero_grad()
        loss_reward = F.mse_loss(model_reward(X), y)
        loss_reward.backward()
        loss_all += loss_reward.item()
        optimizer_reward.step()
        return loss_all

    train_loss_reward = np.zeros(EPOCHS)

    for epoch in range(EPOCHS):
        train_loss_reward[epoch] = train_reward(X, y)

    eta = torch.zeros((n_train, num_actions))

    for i in range(num_actions):
        treats_tmp = i*torch.ones(n_train)
        if len(np.shape(xs_train))==1:
            X_tmp = np.concatenate((np.expand_dims(xs_train, axis = 1), np.expand_dims(treats_tmp, axis = 1)), axis = 1)
        else:
            X_tmp = np.concatenate((xs_train, np.expand_dims(treats_tmp, axis = 1)), axis = 1)
        eta[:,i] = model_reward(torch.tensor(X_tmp).to(device))

    n_train = len(outcomes_train)

    X = torch.tensor(xs_train)

    model_policy = ClassNet(5, num_actions).to(device)
    optimizer_policy = torch.optim.Adam(model_policy.parameters(), lr=1e-3)
    treats = torch.tensor(treats_train)
    props = 1/num_actions * torch.ones_like(treats_train)

    def train_policy(X,y, treats, props):
        model_policy.train()
        loss_all = 0
        X = X.to(device)
        y = y.to(device)
        optimizer_policy.zero_grad()
        loss = dr_shrink_loss(model_policy(X), treats, y, props, eta.detach(), lamb)
        loss.backward()
        loss_all += loss.item()
        optimizer_policy.step()
        return loss_all

    train_loss_policy = np.zeros(EPOCHS)

    for epoch in tqdm(range(EPOCHS)):
        train_loss_policy[epoch] = train_policy(X, y, treats, props)

    def off_policy_eval(df_test):
        X_test = df_test.drop(['Clicked', 'Treat'], axis = 1)
        treats_pred = model_policy(torch.tensor(X_test.values))
        treats_test = torch.tensor(df_test.Treat.astype(np.int32).values)
        y_test = torch.tensor(df_test.Clicked.astype(np.int32).values)
        policy = treats_pred[np.indices(treats_test.shape)[0], treats_test.long()]
        return torch.mean(torch.mul(policy,y_test)*num_actions)

    return off_policy_eval(df_test)
