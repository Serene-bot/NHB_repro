import mne
import os
import glob
import numpy as np
import pandas as pd


def read_raw(file):
    inst = mne.io.read_raw_brainvision(file, preload=True)
    inst.set_channel_types({ch: 'eog' for ch in EOG_CHANNELS})
    montage = mne.channels.make_standard_montage('standard_1005')
    inst.set_montage(montage)

    return inst


def preprocess_raw(inst):
    # 全脑平均
    inst.set_eeg_reference(ref_channels='average')

    # 0.5~40 滤波
    inst.filter(l_freq=0.5, h_freq=40, method='fir')

    # 降采样
    inst.resample(sfreq=250)

    # ICA去眼电
    ica = mne.preprocessing.ICA(n_components=30)
    ica.fit(inst)
    eog_indices, _ = ica.find_bads_eog(inst)
    ica.exclude = eog_indices
    ica.apply(inst)

    return inst


# def detect_trends(data, max_slope=50e-6):
#     grad = np.gradient(data, axis=-1)
#     return np.any(np.abs(grad) > max_slope, axis=-1)


def create_epoch(inst, metadata):
    # 找到marker
    events_stim = mne.events_from_annotations(inst, {'Stimulus/S 10': 10})
    events_resp = mne.events_from_annotations(inst, {'Stimulus/S 20': 20})

    # 分段
    inst_stim = mne.Epochs(inst, events_stim[0],
                          tmin=-0.2, tmax=4.0,
                          baseline=(-0.2, 0),
                          preload=True)
    inst_resp = mne.Epochs(inst, events_resp[0],
                          tmin=-2.0, tmax=0.8,
                          baseline=(-0.2, 0),
                          preload=True)

    # 剔除最开始的4个练习试次
    inst_stim = inst_stim[4:]
    inst_resp = inst_resp[4:]

    # 行为数据与脑电数据匹配
    inst_stim.metadata = metadata
    inst_resp.metadata = metadata

    # 卡阈值
    inst_stim.drop_bad(reject=dict(eeg=150e-6))
    inst_resp.drop_bad(reject=dict(eeg=150e-6))

    # 梯度伪影检测 S1
    # 算法可能不合理，部分被试在伪影检测后所有试次都会被去除
    # 因此先不做伪影的检测
    # inst_stim_mask = np.any(detect_trends(inst_stim.get_data()), axis=1)
    # inst_resp_mask = np.any(detect_trends(inst_resp.get_data()), axis=1)
    #
    # # 梯度伪影检测 S2
    # # 根据检测结果去除坏段
    # inst_stim = inst_stim.drop(inst_stim_mask, 'TREND')
    # inst_resp = inst_resp.drop(inst_resp_mask, 'TREND')

    return inst_stim, inst_resp, inst_stim.metadata, inst_resp.metadata


if __name__ == '__main__':
    DATA_DIR = "D:/naturehuman/EEG_raw"
    OUTPUT_DIR = "D:/naturehuman/output"
    EOG_CHANNELS = ['LO1', 'LO2', 'IO1', 'IO2']

    behavior_data = pd.read_excel('D:/naturehuman/allSubDataTable.xls')

    vhdr_files = glob.glob(os.path.join(DATA_DIR, "**", "*.vhdr"), recursive=True)
    for vhdr_file in vhdr_files[:]:

        sid = os.path.basename(vhdr_file).split('.')[0]

        if sid == '5039':
            continue

        behavior_data_subj = behavior_data[behavior_data['SubNum'] == int(sid)]

        raw = read_raw(vhdr_file)
        raw = preprocess_raw(raw)
        epo_stim, epo_resp, beh_stim, beh_resp = create_epoch(raw, behavior_data_subj)

        np.save(os.path.join(OUTPUT_DIR, 'eeg', 'stim', sid), epo_stim)
        np.save(os.path.join(OUTPUT_DIR, 'eeg', 'resp', sid), epo_resp)

        beh_stim.to_csv(os.path.join(OUTPUT_DIR, 'beh', 'stim', f'{sid}.csv'))
        beh_resp.to_csv(os.path.join(OUTPUT_DIR, 'beh', 'resp', f'{sid}.csv'))

        epo_stim.save(os.path.join(OUTPUT_DIR, 'eeg_fif', 'stim', f'{sid}_stim-epo.fif'), overwrite=True)
        epo_resp.save(os.path.join(OUTPUT_DIR, 'eeg_fif', 'resp', f'{sid}_resp-epo.fif'), overwrite=True)

