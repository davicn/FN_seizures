import numpy as np 
import pandas as pd 
from functions import features as ft 
import os 

fs = 512
PATH = os.getcwd()

N = np.load(PATH + '/data/nfocal_janelas.npy')
F = np.load(PATH + '/data/focal_janelas.npy')


kn = ft.curtose(N,fs)
sn = ft.assimetria(N,fs)
vn = ft.variancia(N,fs)
en = ft.energia(N,fs)

kf = ft.curtose(F,fs)
sf = ft.assimetria(F,fs)
vf = ft.variancia(F,fs)
ef = ft.energia(F,fs)

np.save("N_features.npy",np.array([kn,sn,vn,en]).T)
np.save("F_features.npy",np.array([kf,sf,vf,ef]).T)