import mne
import numpy as np
import glob
import os

# 定义全局参数
DATA_DIR = "D:/naturehuman/EEG_raw"
OUTPUT_DIR = "D:/naturehuman/output"
EOG_CHANNELS = ['LO1', 'LO2', 'IO1', 'IO2']
EVENT_CHANNELS = {'stim': 'S 10', 'resp': 'S 20'}


# 数据加载函数:加载bp数据，定义脑电
def load_raw_data(vhdr_path):
    raw = mne.io.read_raw_brainvision(vhdr_path, preload=True)
    raw.set_channel_types({ch: 'eog' for ch in EOG_CHANNELS})
    montage = mne.channels.make_standard_montage('standard_1005')
    raw.set_montage(montage)
    return raw


# 核心预处理函数
def preprocess_single_file(raw):

    # 重参考为平均参考
    raw.set_eeg_reference(ref_channels='average')

    # 40Hz低通滤波
    raw.filter(l_freq=0.5, h_freq=40, method='fir')

    # ICA去眼电
    ica = mne.preprocessing.ICA(n_components=20)
    ica.fit(raw)
    eog_indices, _ = ica.find_bads_eog(raw)
    ica.exclude = eog_indices
    ica.apply(raw)

    return raw


# 分段处理函数
def enhanced_create_epochs(raw, event_id=None):
    # 检测事件
    stim_events = mne.find_events(raw, stim_channel=EVENT_CHANNELS['stim'])
    resp_events = mne.find_events(raw, stim_channel=EVENT_CHANNELS['resp'])

    # 创建分段
    # Create both stimulus and response locked epochs
    stim_epochs = mne.Epochs(raw,
                             events=mne.find_events(raw, stim_channel=EVENT_CHANNELS['stim']),
                             event_id=event_id,
                             tmin=-0.2,  # -200ms
                             tmax=4.0,  # 4000ms like MATLAB code
                             baseline=(-0.2, 0),
                             reject={'eeg': 150e-6},  # 150 µV like MATLAB
                             reject_by_annotation=True,
                             preload=True)

    resp_epochs = mne.Epochs(raw,
                             events=mne.find_events(raw, stim_channel=EVENT_CHANNELS['resp']),
                             tmin=-2.0,  # -2000ms like MATLAB
                             tmax=0.8,  # 800ms like MATLAB
                             baseline=(-0.2, 0),
                             reject={'eeg': 150e-6},
                             reject_by_annotation=True,
                             preload=True)

    # Add trend rejection (similar to MATLAB rejtrend)
    def detect_trends(data, window_size=100, max_slope=50e-6):
        grad = np.gradient(data, axis=-1)
        return np.any(np.abs(grad) > max_slope, axis=-1)

    # Apply trend rejection
    trend_mask_stim = detect_trends(stim_epochs.get_data())
    trend_mask_resp = detect_trends(resp_epochs.get_data())

    stim_epochs.drop(trend_mask_stim, 'TREND')
    resp_epochs.drop(trend_mask_resp, 'TREND')

    # Remove practice trials (first 5 trials by default)
    if len(stim_epochs) > 5:
        stim_epochs.drop(indices=range(5))
    if len(resp_epochs) > 5:
        resp_epochs.drop(indices=range(5))

    return stim_epochs, resp_epochs


def compute_erp_averages(epochs, conditions=None):
    """Compute ERPs for different conditions"""
    if conditions is None:
        return epochs.average()

    averages = {}
    for condition in conditions:
        averages[condition] = epochs[condition].average()

    return averages


# Modified batch processing
def enhanced_batch_processing():
    all_results = []

    for file_path in vhdr_files:
        try:
            # Load and preprocess
            raw = load_raw_data(file_path)
            clean_raw = preprocess_single_file(raw)

            # Create epochs with enhanced function
            stim_epochs, resp_epochs = enhanced_create_epochs(
                clean_raw,
                event_id={'stimulus': 10, 'response': 20}  # Adjust based on your triggers
            )

            # Compute ERPs
            conditions = ['stimulus', 'response']
            erp_results = {
                'stim': compute_erp_averages(stim_epochs, conditions),
                'resp': compute_erp_averages(resp_epochs, conditions)
            }

            # Extract CPP
            cpp_results = extract_cpp(stim_epochs, conditions)

            # Save results
            save_all_results(
                subj_id=os.path.basename(file_path).split('.')[0],
                raw=clean_raw,
                epochs={'stim': stim_epochs, 'resp': resp_epochs},
                erps=erp_results,
                cpp=cpp_results
            )

        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            continue

def batch_processing():
    all_results = []
    vhdr_files = glob.glob(os.path.join(DATA_DIR, "**", "*.vhdr"), recursive=True)

    for file_path in vhdr_files:
        try:
            # Generate unique subject identifier
            subj_id = os.path.basename(file_path).split('.')[0]
            print(f"Processing {subj_id}...")

            # Processing pipeline
            raw = load_raw_data(file_path)  # Load raw data
            clean_raw = preprocess_single_file(raw)  # Perform preprocessing
            stim_epochs, resp_epochs = enhanced_create_epochs(clean_raw)  # Replace here

            # Save results
            save_results(subj_id, clean_raw, stim_epochs, resp_epochs)

        except Exception as e:
            print(f"Error processing {subj_id}: {str(e)}")
            continue

# 结果保存函数
def save_results(subj_id, raw, stim_epochs, resp_epochs):
    output_path = os.path.join(OUTPUT_DIR, subj_id)
    os.makedirs(output_path, exist_ok=True)

    # 保存原始数据
    raw.save(os.path.join(output_path, f"{subj_id}_raw.fif"))

    # 保存分段数据
    stim_epochs.save(os.path.join(output_path, f"{subj_id}_stim-epo.fif"))
    resp_epochs.save(os.path.join(output_path, f"{subj_id}_resp-epo.fif"))

    # 生成报告
    report = mne.Report(title=f"Report - {subj_id}")
    report.add_raw(raw=raw, title="Clean Raw")
    report.add_epochs(epochs=stim_epochs, title="Stimulus Epochs")
    report.save(os.path.join(output_path, f"{subj_id}_report.html"))


if __name__ == "__main__":
    batch_processing()


def extract_cpp(epochs, conditions=None):
    """提取跨条件CPP特征，支持多条件分析"""
    # 电极选择建议：中央顶叶区域组合
    cpp_channels = ['P3', 'Pz', 'P4', 'POz']

    # 条件处理逻辑
    if conditions is not None:
        epochs = epochs[conditions]  # 按条件筛选

    # 选择电极并计算ERP
    cpp_data = epochs.pick_channels(cpp_channels)
    evoked = cpp_data.average()  # 计算总平均

    # 时域分析（200-700ms窗口）
    cpp_window = evoked.copy().crop(tmin=0.2, tmax=0.7)

    # 添加拓扑分布分析
    peak_time = evoked.get_peak(tmin=0.2, tmax=0.7, mode='pos')[1]
    peak_amplitude = evoked.copy().crop(peak_time, peak_time).data.mean()

    return {
        'evoked': evoked,
        'time_window': cpp_window,
        'peak_latency': peak_time,
        'peak_amplitude': peak_amplitude,
        'topography': evoked.copy().crop(0.2, 0.7).data.mean(axis=1)
    }


def analyze_cpp_across_subjects(all_results):
    """跨被试CPP分析"""
    # 初始化存储结构
    group_data = {
        'amplitude': [],
        'latency': [],
        'topo_pattern': []
    }

    # 汇总数据
    for subj_data in all_results:
        group_data['amplitude'].append(subj_data['cpp']['peak_amplitude'])
        group_data['latency'].append(subj_data['cpp']['peak_latency'])
        group_data['topo_pattern'].append(subj_data['cpp']['topography'])

    # 统计分析
    stats_results = {
        'mean_amp': np.mean(group_data['amplitude']),
        'std_amp': np.std(group_data['amplitude']),
        'mean_latency': np.mean(group_data['latency']),
        'amp_ttest': ttest_1samp(group_data['amplitude'], 0)  # 示例检验
    }

    return group_data, stats_results


# 修改主处理流程
def batch_processing():
    all_results = []
    vhdr_files = glob.glob(...)  # 同前

    for file_path in vhdr_files:
        try:
            # ...前面处理步骤不变...

            # CPP分析（示例按条件分析）
            conditions = {'stim_type': ['target', 'nontarget']}  # 根据实际事件ID修改
            cpp_results = {
                'all': extract_cpp(stim_epochs),
                'by_condition': {
                    cond: extract_cpp(stim_epochs, [cond])
                    for cond in conditions['stim_type']
                }
            }

            # 保存CPP数据
            save_cpp_results(subj_id, cpp_results, output_path)

            # 收集跨被试数据
            all_results.append({
                'subject': subj_id,
                'cpp': cpp_results
            })

        except Exception as e:
            print(f"Error in {subj_id}: {str(e)}")
            continue

    # 执行组分析
    group_data, stats = analyze_cpp_across_subjects(all_results)
    save_group_analysis(group_data, stats)


def save_cpp_results(subj_id, results, path):
    """保存CPP分析结果"""
    # 保存时域数据
    results['all']['time_window'].save(
        os.path.join(path, f"{subj_id}_cpp_window-ave.fif")
    )

    # 保存特征指标
    features = {
        'peak_amp': results['all']['peak_amplitude'],
        'peak_latency': results['all']['peak_latency'],
        'conditions': list(results['by_condition'].keys())
    }
    np.savez(os.path.join(path, f"{subj_id}_cpp_features.npz"), **features)


def visualize_cpp(evoked, save_path):
    """生成CPP可视化"""
    fig = evoked.plot(spatial_colors=True, show=False)
    fig.savefig(os.path.join(save_path, 'cpp_waveform.png'))

    # 拓扑图
    topo_fig = evoked.plot_topomap(times=[0.3, 0.5, 0.7], show=False)
    topo_fig.savefig(os.path.join(save_path, 'cpp_topography.png'))