import numpy as np
import pandas as pd

ihdp = pd.read_csv('./datasets/IHDP/csv/ihdp_npci_1.csv', header = None)

np.random.seed(2024)

xs = ihdp.iloc[:,-25:].values
treats = ihdp.iloc[:,0].values
# outcomes = ihdp.iloc[:,1].values
# ctrfacts = ihdp.iloc[:,2].values
# ctrl = ihdp.iloc[:,3].values
# trt = ihdp.iloc[:,4].values

replications = 1000

n = 747

for i in range(replications):
    beta = np.random.choice([0,0.1,0.2,0.3,0.4], size = 25, p = [0.6, 0.1, 0.1, 0.1, 0.1])
    ctrl = np.exp((xs+0.5).dot(beta))
    tmp = xs.dot(beta)
    omega = np.mean(tmp[treats]) - np.mean(ctrl[treats]) - 4
    trt = tmp - omega
    ctrl_noise = ctrl + np.random.normal(size = n)
    trt_noise = trt + np.random.normal(size = n)
    outcomes = treats * trt_noise + (1-treats) * ctrl_noise
    ctrfacts = (1-treats) * trt_noise + treats * ctrl_noise

    a = np.concatenate([np.expand_dims(treats, 1), np.expand_dims(outcomes, 1), np.expand_dims(ctrfacts, 1), np.expand_dims(ctrl, 1), np.expand_dims(trt, 1), xs], axis = 1)
    filename = './datasets/ihdp_A_' + str(i+1) + '.csv'
    np.savetxt(filename, a, delimiter=",")

