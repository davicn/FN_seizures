import os
import numpy as np
import pandas as pd

PATH = os.getcwd()

fs = 512

N = pd.read_csv(PATH + '/docs/N.txt', header=None)
F = pd.read_csv(PATH + '/docs/F.txt', header=None)


n_ = np.zeros((2,1))

for i in range(800):
    sig = pd.read_csv(PATH + '/data/' + N.iloc[i, 0], header=None).to_numpy().T
    n_ = np.hstack((n_,sig))

n_ = n_[:,1:]


f_ = np.zeros((2,1))

for i in range(800):
    sig = pd.read_csv(PATH + '/data/' + F.iloc[i, 0], header=None).to_numpy().T
    f_ = np.hstack((f_,sig))

f_ = f_[:,1:]

np.save('focal_janelas.npy',n_)
np.save('nfocal_janelas.npy',f_)


