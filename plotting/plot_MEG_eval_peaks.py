#%%
import os
import sys


sys.path.append('/home/reabt/experiments/ncc/MRI/code/')

from utils.load_cfg import load_config_instance
from utils.plots import *
from utils.rsa import *
from utils.subj import *

import joblib
import rsatoolbox as rsa
import matplotlib.pyplot as plt
from glob import glob
import numpy as np


#%% Load data for plotting

results_files = sorted(glob('/home/reabt/experiments/ncc/MEG/data/movie_eval/*'))

sensory_data, suprasensory_data = [], []

for file in results_files:
    results = joblib.load(file)
    sensory_data.append(results.evaluations[0][0])
    suprasensory_data.append(results.evaluations[0][1])


#%% 
'''
============================================================================
Plotting
============================================================================

Option 1: Plot as line plots with Red dots at peaks
---------------------------------------------------
'''

sensory_max = [(np.max(data).item(), np.argmax(data).item()) for data in sensory_data]
suprasensory_max = [(np.max(data).item(), np.argmax(data).item()) for data in suprasensory_data]

x = np.arange(200)

for n, data in enumerate(sensory_data):
    plt.plot(x, data, label=f"Subject {n}", alpha=0.3, linewidth=1)
[plt.scatter(m[1], 
        m[0], 
        alpha=1,
        s = 5,
        c = 'red') for n, m in enumerate(sensory_max)]
plt.title('Sensory')
plt.show()


for n, data in enumerate(suprasensory_data):
    plt.plot(x, data, label=f"Subject {n}", alpha=0.3, linewidth=1)
[plt.scatter(m[1], 
        m[0], 
        alpha=1,
        s = 5,
        c = 'red') for n, m in enumerate(suprasensory_max)]
plt.title('Suprasensory')
plt.show()

#%% 
''' 
Option 2: Scatter plot + Mean line plot with shaded error bar
-------------------------------------------------------------
'''

labels= ALL_MODELS
for i, data in enumerate([sensory_data, suprasensory_data]):

    # [1] Plot all subjects as scatter plot
    x = np.arange(200)

    [plt.scatter(x, 
                data, 
                label=str(n), 
                alpha=0.7,
                s = 1) for n, data in enumerate(data)]

    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title(f'{labels[i]} Data')
    plt.legend()
    plt.show()

    x = np.arange(200)
    sensory_array = np.array(data)

    mean_data = np.mean(data, axis=0)
    std_data = np.std(data, axis=0)

    # [2] Plot mean with shaded error bar
    plt.plot(x, mean_data, color='black', label='Mean', linewidth=2)
    plt.fill_between(x, mean_data - std_data, mean_data + std_data, color='gray', alpha=0.3, label='±1 std')

    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title(f'{labels[i]} Data (mean/std)')
    plt.legend()
    plt.show()

#%% 

'''
option 3: plot Sensory and suprasensory in one
-----------------------------------------------
'''

sensory_array = np.array(sensory_data)
sensory_mean = np.mean(sensory_data, axis=0)
sensory_std = np.std(sensory_data, axis=0)

suprasensory_array = np.array(suprasensory_data)
suprasensory_mean = np.mean(suprasensory_data, axis=0)
suprasensory_std = np.std(suprasensory_data, axis=0)

plt.plot(x, np.array(sensory_data), color='blue', label='Sensory', linewidth=2)

# [2] Plot mean with shaded error bar
plt.plot(x, sensory_mean, color='blue', label='Sensory', linewidth=2)
plt.fill_between(x, sensory_mean - sensory_std, sensory_mean + sensory_std, color='gray', alpha=0.3, label='±1 std')

plt.plot(x, suprasensory_mean, color='red', label='Suprasensory', linewidth=2)
plt.fill_between(x, suprasensory_mean - suprasensory_std, suprasensory_mean + suprasensory_std, color='gray', alpha=0.3, label='±1 std')


plt.xlabel('Index')
plt.ylabel('Value')
plt.title(f'{labels[i]} Data (mean/std)')
plt.legend()
plt.show()


# %%
