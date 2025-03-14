import os
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA


def main():
    for data_type in ['stim', 'resp']:
        root = r'D:\naturehuman\output'
        datab = pd.read_csv(os.path.join(root, 'merge', f'beh_{data_type}.csv'))
        datab = datab[(datab['Choice1'] != -1) & (datab['SubNum'] != 5028)]
        datab = np.column_stack([
            datab['ChosenV'] / 10,
            datab['UnchosenV'] / 10,
            datab['MaxvMinBid'] / 10,
            datab['AvBid'] / 10,
            datab['AvBidSetSal'] / 5,
            datab['Anxious'] / 5,
            datab['Liking'] / 5,
            datab['Confident'] / 5
        ])

        pca = PCA(n_components=2)
        score = pca.fit_transform(datab)

        np.save(os.path.join(root, 'pca', f'ac_{data_type}'), score)


if __name__ == '__main__':
    main()