import os
import numpy as np
import pandas as pd
import statsmodels.api as sm


def main():
    for data_type in ['stim', 'resp']:
        root = r'D:\naturehuman\output'
        datab = pd.read_csv(os.path.join(root, 'merge', f'beh_{data_type}.csv'))
        mask = (datab['Choice1'] != -1) & (datab['SubNum'] != 5028)
        datae = np.load(os.path.join(root, 'merge', f'eeg_{data_type}.npy'))
        y = datae[mask]
        x = np.load(os.path.join(root, 'pca', f'ac_{data_type}.npy'))
        trials, channels, timepoints = y.shape

        y_hat = np.zeros([trials, channels, timepoints])
        t_map = np.zeros([2, channels, timepoints])
        for cid in range(channels):
            for tid in range(timepoints):
                model = sm.OLS(y[:, cid, tid], x).fit()
                y_hat[:, cid, tid] = model.predict(x)
                t_map[0, cid, tid] = model.tvalues[0]
                t_map[1, cid, tid] = model.tvalues[1]
                print(cid, tid)

        np.save(os.path.join(root, 'regress', f'regress_{data_type}'), y_hat)
        np.save(os.path.join(root, 'regress', f't_{data_type}'), t_map)


if __name__ == '__main__':
    main()