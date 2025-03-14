import os
import numpy as np

root = r'D:\naturehuman\output'
regress_stim = np.load(os.path.join(root, 'regress', f'regress_stim.npy'))
regress_resp = np.load(os.path.join(root, 'regress', f'regress_resp.npy'))
t_stim = np.load(os.path.join(root, 'regress', f't_stim.npy'))
t_resp = np.load(os.path.join(root, 'regress', f't_resp.npy'))

regress_stim = regress_stim * t_stim[0]
regress_resp = regress_resp * t_resp[1]

np.save(os.path.join(root, 'regress', f'regress_w_stim.npy'), regress_stim)
np.save(os.path.join(root, 'regress', f'regress_w_resp.npy'), regress_resp)