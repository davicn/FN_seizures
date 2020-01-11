import numpy as np 
import pandas as pd 
import os 
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

PATH = os.getcwd()

N = np.load(PATH + '/data/N_features.npy')
F = np.load(PATH + '/data/F_features.npy')

d = np.vstack((N,F))
col = ['Kur','Skew','Var','Eng']
data = pd.DataFrame(data=d,columns=col)

y = np.hstack((np.repeat('Não focal',len(d)//2),np.repeat('Focal',len(d)//2)))
data['Type'] = y
#teste

# Boxplot -----------------------------------------------
fig = plt.figure()
gs = GridSpec(ncols=2,nrows=2,figure=fig)

ax1 = fig.add_subplot(gs[0,0])
sns.boxplot(x='Type', y='Kur',
             hue="Type", data=data,ax=ax1)

ax2 = fig.add_subplot(gs[0,1])
sns.boxplot(x='Type', y='Skew',
             hue="Type", data=data,ax=ax2)

ax3 = fig.add_subplot(gs[1,0])
sns.boxplot(x='Type', y='Var',
             hue="Type", data=data,ax=ax3)

ax4 = fig.add_subplot(gs[1,1])
sns.boxplot(x='Type', y='Eng',
             hue="Type", data=data,ax=ax4)

plt.show()