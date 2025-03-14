import os
import glob
import numpy as np
import pandas as pd


def main():
    root = r'D:\naturehuman\output'

    for data_type in ['stim', 'resp']:
        # eeg
        eeg = []
        for s in glob.glob(os.path.join(root, 'eeg', data_type, '*.npy')):
            tmp = np.load(s)
            if tmp.shape[0] > 0:
                eeg += [tmp]
        eeg = np.vstack(eeg)
        np.save(os.path.join(root, 'merge', f'eeg_{data_type}.npy'), eeg)

        # beh
        beh = []
        for s in glob.glob(os.path.join(root, 'beh', data_type, '*.csv')):
            tmp = pd.read_csv(s)
            if tmp.shape[0] > 0:
                beh += [tmp]
        beh = pd.concat(beh)
        beh.to_csv(os.path.join(root, 'merge', f'beh_{data_type}.csv'))


if __name__ == '__main__':
    main()
