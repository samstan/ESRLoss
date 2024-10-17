import numpy as np

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
    def __init__(self, dim):
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

def reg_learner(xs_train, treats_train, outcomes_train, xs_test, ctrl_test, trt_test, ctrl_train, trt_train, k = 5):
    n_train = len(outcomes_train)
    n_test = len(xs_test)

    if len(np.shape(xs_train))==1:
        X = np.concatenate((np.expand_dims(xs_train, axis = 1), np.expand_dims(treats_train, axis = 1)), axis = 1)
    else:
        X = np.concatenate((xs_train, np.expand_dims(treats_train, axis = 1)), axis = 1)
    y = outcomes_train

    X = torch.tensor(X)
    y = torch.tensor(y)
    model = Net(26).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


    train_loss = np.zeros(EPOCHS)
    train_reg = np.zeros(EPOCHS)

    closest = calc_pairs(X[:,:-1], X[:,-1])

    def train(X,y, closest):
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
        train_loss[epoch] = train(X, y, closest)

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
    
    reg = 0
    for i in range(n_test):
        treated = model(torch.tensor(np.concatenate((X_test[i], np.array([1]))))).detach().numpy()
        control = model(torch.tensor(np.concatenate((X_test[i], np.array([0]))))).detach().numpy()

        decision = np.argmax([control,treated])
        opts = [ctrl_test[i],trt_test[i]]
        reg += np.max(opts) - opts[decision]
        
    
    return train_loss, reg/n_test

def s_learner(xs_train, treats_train, outcomes_train, xs_test, ctrl_test, trt_test, ctrl_train, trt_train):
    n_train = len(outcomes_train)
    n_test = len(xs_test)

    if len(np.shape(xs_train))==1:
        X = np.concatenate((np.expand_dims(xs_train, axis = 1), np.expand_dims(treats_train, axis = 1)), axis = 1)
    else:
        X = np.concatenate((xs_train, np.expand_dims(treats_train, axis = 1)), axis = 1)
    y = outcomes_train

    X = torch.tensor(X)
    y = torch.tensor(y)
    model = Net(26).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


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

    if len(np.shape(xs_test))==1:
        X_test = np.expand_dims(xs_test, axis = 1)
    else:
        X_test = xs_test
    
    reg = 0
    for i in range(n_test):
        treated = model(torch.tensor(np.concatenate((X_test[i], np.array([1]))))).detach().numpy()
        control = model(torch.tensor(np.concatenate((X_test[i], np.array([0]))))).detach().numpy()

        decision = np.argmax([control,treated])
        opts = [ctrl_test[i],trt_test[i]]
        reg += np.max(opts) - opts[decision]
        
    
    return train_loss, reg/n_test
    # return mean_squared_error(CATEs, cates_test)


def t_learner(xs_train, treats_train, outcomes_train, xs_test, ctrl_test, trt_test, ctrl_train, trt_train):
    n_train = len(outcomes_train)
    n_test = len(xs_test)

    if len(np.shape(xs_train))==1:
        X = np.expand_dims(xs_train, axis = 1)
    else:
        X = xs_train
    y = outcomes_train

    X_trt = torch.tensor(X[treats_train==1])
    y_trt = torch.tensor(y[treats_train==1])
    X_ctr = torch.tensor(X[treats_train==0])
    y_ctr = torch.tensor(y[treats_train==0])


    model1 = Net(25).to(device)
    optimizer1 = torch.optim.Adam(model1.parameters(), lr=0.01)

    model0 = Net(25).to(device)
    optimizer0 = torch.optim.Adam(model0.parameters(), lr=0.01)

    train_loss1 = np.zeros(EPOCHS)
    train_loss0 = np.zeros(EPOCHS)

    def train(X,y, model, optimizer):
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

    for epoch in range(EPOCHS):
        train_loss1[epoch] = train(X_trt, y_trt, model1, optimizer1)
        train_loss0[epoch] = train(X_ctr, y_ctr, model0, optimizer0)


    if len(np.shape(xs_test))==1:
        X_test = np.expand_dims(xs_test, axis = 1)
    else:
        X_test = xs_test
    
    reg = 0
    for i in range(n_test):
        treated = model1(torch.tensor(X_test[i])).detach().numpy()
        control = model0(torch.tensor(X_test[i])).detach().numpy()
        decision = np.argmax([control,treated])

        opts = [ctrl_test[i],trt_test[i]]
        reg += np.max(opts) - opts[decision]
    
    return reg/n_test

def r_learner(xs_train, treats_train, outcomes_train, xs_test, ctrl_test, trt_test, ctrl_train, trt_train):
    n_train = len(outcomes_train)
    n_test = len(xs_test)

    if len(np.shape(xs_train))==1:
        X = np.expand_dims(xs_train, axis = 1)
    else:
        X = xs_train
    y = outcomes_train

    half1_idxs_train = sample_without_replacement(n_train, n_train//2)
    n1 = len(half1_idxs_train)
    half2_idxs_train = np.setdiff1d(np.arange(n_train), half1_idxs_train, True)
    n2 = len(half2_idxs_train)

    X1, treats_train1, y1 = X[half1_idxs_train], treats_train[half1_idxs_train], y[half1_idxs_train]
    X2, treats_train2, y2 = X[half2_idxs_train], treats_train[half2_idxs_train], y[half2_idxs_train]

    X1 = torch.tensor(X1)
    treats_train1 = torch.tensor(treats_train1)
    y1 = torch.tensor(y1)
    X2 = torch.tensor(X2)
    treats_train2 = torch.tensor(treats_train2)
    y2 = torch.tensor(y2)

    def train(X,y, model, optimizer):
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

    def train_prob(X,y, model, optimizer):
        model.train()
        loss_all = 0
        X = X.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        loss = F.binary_cross_entropy_with_logits(model(X), y)
        loss.backward()
        loss_all += loss.item()
        optimizer.step()
        return loss_all

    def train_weight(X,y, weight, model, optimizer):
        model.train()
        loss_all = 0
        X = X.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        loss = weighted_mse_loss(model(X), y, weight)
        loss.backward()
        loss_all += loss.item()
        optimizer.step()
        return loss_all

    model1 = Net(25).to(device)
    optimizer1 = torch.optim.Adam(model1.parameters(), lr=0.01)

    model1p = Net(25).to(device)
    optimizer1p = torch.optim.Adam(model1p.parameters(), lr=0.01)

    model2 = Net(25).to(device)
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.01)

    model2p = Net(25).to(device)
    optimizer2p = torch.optim.Adam(model2p.parameters(), lr=0.01)


    train_loss1 = np.zeros(EPOCHS)
    train_loss1p = np.zeros(EPOCHS)
    train_loss2 = np.zeros(EPOCHS)
    train_loss2p = np.zeros(EPOCHS)

    for epoch in range(EPOCHS):
        train_loss1[epoch] = train(X1, y1, model1, optimizer1)
        train_loss1p[epoch] = train_prob(X1, treats_train1.float(), model1p, optimizer2p)
        train_loss2[epoch] = train(X2, y2, model2, optimizer2)
        train_loss2p[epoch] = train_prob(X2, treats_train2.float(), model2p, optimizer2p)

    y_residuals = torch.zeros(n_train)

    e_residuals = torch.zeros(n_train)

    half1_idxs_set = set(half1_idxs_train) #for ~O(1) membership checking
    half2_idxs_set = set(half2_idxs_train)

    treats_train = torch.tensor(treats_train)
    X = torch.tensor(X)
    y = torch.tensor(y)

    for i in range(n_train):
        if i in half1_idxs_set:
            y_residuals[i] = y[i] - model2(X[i])
            e_residuals[i] = treats_train[i] - model2p(X[i])
        else:
            y_residuals[i] = y[i] - model1(X[i])
            e_residuals[i] = treats_train[i] - model1p(X[i])
    
    pseudos = y_residuals/e_residuals
    weights = torch.pow(e_residuals, 2)

    
    pseudos = pseudos.clone().detach()
    weights = weights.clone().detach()

    model3 = Net(25).to(device)
    optimizer3 = torch.optim.Adam(model3.parameters(), lr=0.01)

    train_loss3 = np.zeros(EPOCHS)
    for epoch in range(EPOCHS):
        train_loss3[epoch] = train_weight(X, pseudos, weights, model3, optimizer3)

    if len(np.shape(xs_test))==1:
        X_test = np.expand_dims(xs_test, axis = 1)
    else:
        X_test = xs_test

    CATEs = model3(torch.tensor(X_test))

    reg = 0

    for i in range(n_test):
        if CATEs[i]>0:
            decision = 1
        elif CATEs[i]<0:
            decision = 0
        else:
            decision = np.random.choice([0,1])
        opts = [ctrl_test[i],trt_test[i]]
        reg += np.max(opts) - opts[decision]

    return reg/n_test


def dr_learner(xs_train, treats_train, outcomes_train, xs_test, ctrl_test, trt_test, ctrl_train, trt_train):
    n_train = len(outcomes_train)
    n_test = len(xs_test)

    if len(np.shape(xs_train))==1:
        X = np.expand_dims(xs_train, axis = 1)
    else:
        X = xs_train
    y = outcomes_train

    stage1_idxs_train = sample_without_replacement(n_train, n_train//2)
    n1 = len(stage1_idxs_train)
    stage2_idxs_train = np.setdiff1d(np.arange(n_train), stage1_idxs_train, True)
    n2 = len(stage2_idxs_train)

    def train(X,y, model, optimizer):
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

    def train_prob(X,y, model, optimizer):
        model.train()
        loss_all = 0
        X = X.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        loss = F.binary_cross_entropy_with_logits(model(X), y)
        loss.backward()
        loss_all += loss.item()
        optimizer.step()
        return loss_all

    X1, treats_train1, y1 = X[stage1_idxs_train], treats_train[stage1_idxs_train], y[stage1_idxs_train]
    X2, treats_train2, y2 = X[stage2_idxs_train], treats_train[stage2_idxs_train], y[stage2_idxs_train]

    X1_trt = torch.tensor(X1[treats_train1 == 1])
    y1_trt = torch.tensor(y1[treats_train1 == 1])
    X1_ctr = torch.tensor(X1[treats_train1 == 0])
    y1_ctr = torch.tensor(y1[treats_train1 == 0])

    X1 = torch.tensor(X1)
    treats_train1 = torch.tensor(treats_train1)
    y1 = torch.tensor(y1)

    X2 = torch.tensor(X2)
    treats_train2 = torch.tensor(treats_train2)
    y2 = torch.tensor(y2)


    model1 = Net(25).to(device)
    optimizer1 = torch.optim.Adam(model1.parameters(), lr=0.01)

    model0 = Net(25).to(device)
    optimizer0 = torch.optim.Adam(model0.parameters(), lr=0.01)
    
    modelp = Net(25).to(device)
    optimizerp = torch.optim.Adam(modelp.parameters(), lr=0.01)


    train_loss1 = np.zeros(EPOCHS)
    train_loss0 = np.zeros(EPOCHS)
    train_lossp = np.zeros(EPOCHS)

    for epoch in range(EPOCHS):
        train_loss1[epoch] = train(X1_trt, y1_trt, model1, optimizer1)
        train_loss0[epoch] = train(X1_ctr, y1_ctr, model0, optimizer0)
        train_lossp[epoch] = train_prob(X1, treats_train1.float(), modelp, optimizerp)


    pseudos = np.zeros(n2)

    for i in range(n2):
        pseudos[i] = model1(X2[i]) - model0(X2[i])
        if treats_train2[i]:
            pseudos[i]+= (1/modelp(X2[i]))*(y2[i] - model1(X2[i]))
        else:
            pseudos[i]-= (1/(1-modelp(X2[i])))*(y2[i] - model0(X2[i]))

    pseudos = torch.tensor(pseudos)


    model2 = Net(25).to(device)
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.01)

    train_loss2 = np.zeros(EPOCHS)

    for epoch in range(EPOCHS):
        train_loss2[epoch] = train(X2, pseudos, model2, optimizer2)


    if len(np.shape(xs_test))==1:
        X_test = np.expand_dims(xs_test, axis = 1)
    else:
        X_test = xs_test

    CATEs = model2(torch.tensor(X_test))

    reg = 0

    for i in range(n_test):
        if CATEs[i]>0:
            decision = 1
        elif CATEs[i]<0:
            decision = 0
        else:
            decision = np.random.choice([0,1])
        opts = [ctrl_test[i],trt_test[i]]
        reg += np.max(opts) - opts[decision]

    return reg/n_test