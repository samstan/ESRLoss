import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

from learners_reg import *

np.random.seed(6746)

num_rep = 1000
# num_split = 2

mse_our = np.zeros(num_rep)
mse_s = np.zeros(num_rep)
mse_t = np.zeros(num_rep)
mse_r = np.zeros(num_rep)
mse_dr = np.zeros(num_rep)

# mse_our_std = np.zeros(num_rep)
# mse_s_std = np.zeros(num_rep)
# mse_t_std = np.zeros(num_rep)
# mse_r_std = np.zeros(num_rep)
# mse_dr_std = np.zeros(num_rep)

for i in tqdm(range(num_rep)):
    # ihdp = pd.read_csv('./datasets/IHDP/csv/ihdp_npci_' + str(i+1)+'.csv', header = None)
    # ihdp = pd.read_csv('foo.csv')
    ihdp = pd.read_csv('./datasets/ihdp_A_' + str(i+1)+'.csv', header = None)
    num = len(ihdp)

    # kf = KFold(n_splits = num_split)

    # mse_our_cv = np.zeros(num_split)
    # mse_s_cv = np.zeros(num_split)
    # mse_t_cv = np.zeros(num_split)
    # mse_r_cv = np.zeros(num_split)
    # mse_dr_cv = np.zeros(num_split)

    j = 0

    ihdp_train, ihdp_test = train_test_split(ihdp, test_size = 0.3)
    xs = ihdp_train.iloc[:,-25:].values
    treats = ihdp_train.iloc[:,0].values
    outcomes = ihdp_train.iloc[:,1].values
    xs_test = ihdp_test.iloc[:,-25:].values

    ctrl_train = ihdp_train.iloc[:,3].values
    trt_train = ihdp_train.iloc[:,4].values

    ctrl_test = ihdp_test.iloc[:,3].values
    trt_test = ihdp_test.iloc[:,4].values


    bla, mse_our[i] = reg_learner(xs, treats, outcomes, xs_test, ctrl_test, trt_test, ctrl_train, trt_train, k = 1)
    # ble, mse_s[i] = s_learner(xs, treats, outcomes, xs_test, ctrl_test, trt_test, ctrl_train, trt_train)
    # mse_t[i] = t_learner(xs, treats, outcomes, xs_test, ctrl_test, trt_test, ctrl_train, trt_train)
    # mse_r[i] = r_learner(xs, treats, outcomes, xs_test, ctrl_test, trt_test, ctrl_train, trt_train)
    # mse_dr[i] = dr_learner(xs, treats, outcomes, xs_test, ctrl_test, trt_test, ctrl_train, trt_train)
    ble, mse_s[i] = reg_learner(xs, treats, outcomes, xs_test, ctrl_test, trt_test, ctrl_train, trt_train, k = 10)
    _, mse_t[i] = reg_learner(xs, treats, outcomes, xs_test, ctrl_test, trt_test, ctrl_train, trt_train, k = 25)
    _, mse_r[i] = reg_learner(xs, treats, outcomes, xs_test, ctrl_test, trt_test, ctrl_train, trt_train, k = 50)
    _, mse_dr[i] = reg_learner(xs, treats, outcomes, xs_test, ctrl_test, trt_test, ctrl_train, trt_train, k = 100)

    # for train_idxs, test_idxs in kf.split(ihdp.values):
    #     ihdp_train = ihdp.iloc[train_idxs]
    #     ihdp_test = ihdp.iloc[test_idxs]

    #     xs = ihdp_train.iloc[:,-25:].values
    #     treats = ihdp_train.iloc[:,0].values
    #     outcomes = ihdp_train.iloc[:,1].values
    #     xs_test = ihdp_test.iloc[:,-25:].values

    #     ctrl_train = ihdp_train.iloc[:,3].values
    #     trt_train = ihdp_train.iloc[:,4].values

    #     ctrl_test = ihdp_test.iloc[:,3].values
    #     trt_test = ihdp_test.iloc[:,4].values


    #     bla, mse_our_cv[j] = reg_learner(xs, treats, outcomes, xs_test, ctrl_test, trt_test, ctrl_train, trt_train)
    #     ble, mse_s_cv[j] = s_learner(xs, treats, outcomes, xs_test, ctrl_test, trt_test, ctrl_train, trt_train)
    #     # mse_t_cv[j] = t_learner(xs, treats, outcomes, xs_test, ctrl_test, trt_test, ctrl_train, trt_train)
    #     # mse_r_cv[j] = r_learner(xs, treats, outcomes, xs_test, ctrl_test, trt_test, ctrl_train, trt_train)
    #     # mse_dr_cv[j] = dr_learner(xs, treats, outcomes, xs_test, ctrl_test, trt_test, ctrl_train, trt_train)
    #     j+=1


    # mse_our[i] = np.mean(mse_our_cv)
    # mse_s[i] = np.mean(mse_s_cv)
    # mse_t[i] = np.mean(mse_t_cv)
    # mse_r[i] = np.mean(mse_r_cv)
    # mse_dr[i] = np.mean(mse_dr_cv)

    # mse_our_std[i] = np.std(mse_our_cv)*1.96/np.sqrt(num_split)
    # mse_s_std[i] = np.std(mse_s_cv)*1.96/np.sqrt(num_split)
    # mse_t_std[i] = np.std(mse_t_cv)*1.96/np.sqrt(num_split)
    # mse_r_std[i] = np.std(mse_r_cv)*1.96/np.sqrt(num_split)
    # mse_dr_std[i] = np.std(mse_dr_cv)*1.96/np.sqrt(num_split)


print(np.mean(mse_our))

print(np.mean(mse_s))
print(np.mean(mse_t))
print(np.mean(mse_r))

print(np.mean(mse_dr))

print(1.96*np.std(mse_our)/np.sqrt(1000))

print(1.96*np.std(mse_s)/np.sqrt(1000))
print(1.96*np.std(mse_t)/np.sqrt(1000))
print(1.96*np.std(mse_r)/np.sqrt(1000))

print(1.96*np.std(mse_dr)/np.sqrt(1000))

