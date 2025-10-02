import numpy as np
import pandas as pd
import neurokit2 as nk
from neurokit2.signal import signal_smooth, signal_zerocrossings
from scipy import signal
from scipy.signal import find_peaks, resample
from scipy.fftpack import fft, fftfreq
import scipy
from librosa import feature
import pywt


def find_phasic_eda_peaks(eda_phasic, percent=0.05):
    derivative = np.gradient(eda_phasic)
    df = signal_smooth(derivative, kernel="bartlett", size=20)
    # Zero crossings
    pos_crossings = signal_zerocrossings(df, direction="positive")
    neg_crossings = signal_zerocrossings(df, direction="negative")
    if len(pos_crossings) != 0 and len(neg_crossings) != 0:
        if pos_crossings[0] > neg_crossings[0]:
            neg_crossings = neg_crossings[1:]
        # Sanitize consecutive crossings
        if len(pos_crossings) > len(neg_crossings):
            pos_crossings = pos_crossings[0:len(neg_crossings)]
        elif len(pos_crossings) < len(neg_crossings):
            neg_crossings = neg_crossings[0:len(pos_crossings)]
        peaks_list = []
        onsets_list = []
        amps_list = []
        for i, j in zip(pos_crossings, neg_crossings):
            window = eda_phasic[i:j]
            amp = np.max(window)
            # Detected SCRs with amplitudes less than 10% of max SCR amplitude will be eliminated
            diff = amp - eda_phasic[i]
            if not diff < (percent * amp):
                peaks = np.where(eda_phasic == amp)[0]
                peaks_list.append(peaks)
                onsets_list.append(i)
                amps_list.append(amp)
        # Sanitize
        peaks = np.array(peaks_list)
        amps = np.array(amps_list)
        onsets = np.array(onsets_list)
    else:
        peaks = np.array([0])
        amps = np.array([0])
        onsets = np.array([0])
    return peaks, amps, onsets


def spectrum(x, fs):
    n = len(x // 2)
    yf = fft(x)
    xf = fftfreq(len(x), 1 / fs)[:n // 2]
    fh = 2.0 / n * np.abs(yf[0:n // 2])
    return fh, xf


def mnf_(fh, xh):
    power_sum = np.sum(fh)
    weighted_power = np.sum(xh * fh)
    mnf = weighted_power / power_sum
    return mnf


def mdf_(fh, xh):
    matrix = np.array([fh, xh])
    sorted_matrix = matrix[:, matrix[0, :].argsort()]
    sorted_fh = sorted_matrix[0, :]
    idx = equilibrium_index(sorted_fh)
    mdf = xh[idx]
    return mdf


def zc_(sample):
    zc = np.sum(sample[:-1] * sample[1:] < 0)
    return zc


def frequency_ratio(fh, xh, mnf):
    idx = np.nonzero(xh == np.round(mnf))[0][0]
    low_frequencies = np.sum(fh[:idx])
    high_frequencies = np.sum(fh[idx + 1:])
    fr = low_frequencies / high_frequencies
    return fr


def equilibrium_index(sorted_array):
    diff = np.empty(len(sorted_array))
    for i in range(len(sorted_array)):
        diff[i] = np.abs((np.sum(sorted_array[0:i]) - np.sum(sorted_array[i + 1:])))
    equilibrium_idx = np.argmin(diff)
    return equilibrium_idx


def energy_(x, k):
    return np.sum(np.array(x[-k]) ** 2)


def entropy(energy):
    return -np.sum(energy * np.log(energy))


def std_(x, mav, i):
    return np.sqrt(1 / (len(x[i] - 1)) * np.sum(np.abs(x[i] - mav[i]) ** 2))


def ssc_(x):
    ssc = 0
    for i in range(1, len(x) - 1):
        if np.sign(x[i] - x[i - 1]) * np.sign(x[i + 1] - x[i]) < 0 and \
                np.abs(x[i] - x[i - 1]) > 0.001 and np.abs(x[i + 1] - x[i]) > 0.001:
            ssc += 1
        else:
            ssc += 0
    return ssc


def wamp_(x):
    wamp = 0
    for i in range(1, len(x) - 1):
        if np.abs(x[i] - x[i - 1]) > 0.5:
            wamp += 1
        else:
            wamp += 0
    return wamp


def extract_hrv_nk_features(ecg, fs):
    """
    compute all the HRV features available in neurokit2.
    """
    columns = ['HRV_MeanNN', 'HRV_SDNN', 'HRV_SDANN1', 'HRV_SDNNI1', 'HRV_SDANN2',
               'HRV_SDNNI2', 'HRV_SDANN5', 'HRV_SDNNI5', 'HRV_RMSSD', 'HRV_SDSD',
               'HRV_CVNN', 'HRV_CVSD', 'HRV_MedianNN', 'HRV_MadNN', 'HRV_MCVNN',
               'HRV_IQRNN', 'HRV_Prc20NN', 'HRV_Prc80NN', 'HRV_pNN50', 'HRV_pNN20',
               'HRV_MinNN', 'HRV_MaxNN', 'HRV_HTI', 'HRV_TINN', 'HRV_ULF', 'HRV_VLF',
               'HRV_LF', 'HRV_HF', 'HRV_VHF', 'HRV_LFHF', 'HRV_LFn', 'HRV_HFn',
               'HRV_LnHF', 'HRV_SD1', 'HRV_SD2', 'HRV_SD1SD2', 'HRV_S', 'HRV_CSI',
               'HRV_CVI', 'HRV_CSI_Modified', 'HRV_PIP', 'HRV_IALS', 'HRV_PSS',
               'HRV_PAS', 'HRV_GI', 'HRV_SI', 'HRV_AI', 'HRV_PI', 'HRV_C1d', 'HRV_C1a',
               'HRV_SD1d', 'HRV_SD1a', 'HRV_C2d', 'HRV_C2a', 'HRV_SD2d', 'HRV_SD2a',
               'HRV_Cd', 'HRV_Ca', 'HRV_SDNNd', 'HRV_SDNNa', 'HRV_DFA_alpha1',
               'HRV_MFDFA_alpha1_Width', 'HRV_MFDFA_alpha1_Peak',
               'HRV_MFDFA_alpha1_Mean', 'HRV_MFDFA_alpha1_Max',
               'HRV_MFDFA_alpha1_Delta', 'HRV_MFDFA_alpha1_Asymmetry',
               'HRV_MFDFA_alpha1_Fluctuation', 'HRV_MFDFA_alpha1_Increment',
               'HRV_ApEn', 'HRV_SampEn', 'HRV_ShanEn', 'HRV_FuzzyEn', 'HRV_MSEn',
               'HRV_CMSEn', 'HRV_RCMSEn', 'HRV_CD', 'HRV_HFD', 'HRV_KFD', 'HRV_LZC']

    if ecg.ndim == 2:
        ecg = np.squeeze(ecg)

    # Alternative methods:
    # 'pantompkins1985', 'engzeemod2012', 'nabian2018'
    r_peaks, r_info = nk.ecg_peaks(ecg, sampling_rate=fs, method='elgendi2010')
    info, r_peaks_corrected = nk.signal_fixpeaks(r_peaks, sampling_rate=fs)
    hrv = nk.hrv(r_peaks_corrected, sampling_rate=fs)[columns]

    features_df = pd.DataFrame(
        hrv,
        columns=columns
    )
    features_df = features_df.drop(
        columns=['HRV_SDANN1', 'HRV_SDNNI1', 'HRV_SDANN2',
                 'HRV_SDNNI2', 'HRV_SDANN5', 'HRV_SDNNI5']
    )
    return features_df


def extract_emg_features(emg, fs):
    """
    Extract EMG features.
    """
    if emg.ndim == 2:
        emg = np.squeeze(emg)

    fh, xh = spectrum(emg, fs)
    n = len(emg)

    # Features
    rmse = np.sqrt(1 / n * np.sum(emg ** 2))
    emg_mav = 1 / n * np.sum(np.abs(emg))
    var = 1 / (n - 1) * (np.sum(emg - np.mean(emg)) ** 2)
    energy = np.sum(np.abs(emg) ** 2)
    mnf = mnf_(fh, xh)
    mdf = mdf_(fh, xh)
    zc = zc_(emg)
    try:
        fr = frequency_ratio(fh, xh, mnf)
    except:
        fr = np.nan

    # DWT
    levels = 4
    dwav = pywt.Wavelet('db3')
    dwt_coeff = pywt.wavedec(emg, wavelet=dwav, level=levels)
    detailed_coeff = dwt_coeff[1:]
    mav = np.array([1 / len(detailed_coeff[i]) * np.sum(np.abs(detailed_coeff[i])) for i in range(levels)])
    std = np.array([std_(detailed_coeff, mav, i) for i in range(levels)])

    temp = np.hstack(
        [rmse, emg_mav, var, energy, mnf, mdf, zc, fr, mav, std]
    )

    columns = [
        "EMG_RMSE", "EMG_MAV", "EMG_VAR", "EMG_energy", "EMG_MNF", "EMG_MDF", "EMG_ZC",
        "EMG_FR", "EMG_DWT_MAV_1", "EMG_DWT_MAV_2", "EMG_DWT_MAV_3",
        "EMG_DWT_MAV_4", "EMG_DWT_STD_1", "EMG_DWT_STD_2", "EMG_DWT_STD_3", "EMG_DWT_STD_4"
    ]
    features_df = pd.DataFrame(
        [temp], 
        columns=columns
    )
    return features_df


def microV_to_microS(eda):
    r = 6.49e6 / ((3.3 / eda) - 1)
    return 1 / r


def extract_eda_time_and_frequency_features(data, fs, f_resampling=None, window=30, ):
    """
    Extract Time and Frequency EDA features.
    """
    if data.ndim == 2:
        data = np.squeeze(data)

    if f_resampling is None:
        f_resampling = fs

    new_num_samples = f_resampling * window
    eda = resample(data, new_num_samples)
    # eda = microV_to_microS(eda)
    eda_df = nk.eda_process(
        eda_signal=eda,
        sampling_rate=f_resampling,
    )[0]

    fh, xh = spectrum(eda, f_resampling)
    derivative = np.gradient(eda)
    second_derivative = np.gradient(derivative)
    eda_phasic = eda_df["EDA_Phasic"].to_numpy()
    eda_tonic = eda_df["EDA_Tonic"].to_numpy()

    peaks, amps, onsets = find_phasic_eda_peaks(eda_phasic)
    # EDA Tonic - SCL
    mean_eda = np.mean(eda_tonic)
    std_eda = np.std(eda_tonic)
    kurt_eda = scipy.stats.kurtosis(eda_tonic)
    skew_eda = scipy.stats.skew(eda_tonic)

    mean_derivative = np.mean(derivative)
    negative_derivative = [i for i in derivative if i < 0]
    mean_negative_derivative = np.mean(negative_derivative)

    # As done in Shukla et al. Feature Extraction and Selection for
    # Emotion Recognition from Electrodermal Activity
    activity = np.sum((eda - np.mean(eda)) ** 2)
    mobility = np.sqrt(np.var(derivative) / np.var(eda))
    complexity = np.sqrt(np.var(second_derivative) / np.var(derivative)) / mobility

    # Neurokit2: Complexity Calculations
    ap_entropy, _ = nk.entropy_approximate(eda)
    samp_entropy, _ = nk.entropy_sample(eda)
    shan_entropy, _ = nk.entropy_shannon(eda)
    fuzz_entropy, _ = 1, 1  # nk.entropy_fuzzy(eda)
    mse_entropy, _ = 1, 1  # nk.entropy_multiscale(eda)
    cmse_entropy, _ = 1, 1  # nk.entropy_multiscale(eda, method='CMSEn')
    rcmse_entropy, _ = 1, 1  # nk.entropy_multiscale(eda, method='RCMSEn')
    
    # Phasic
    rise_time = (peaks.squeeze() - onsets) / f_resampling
    peaks_count = peaks.shape[0]
    mean_peak_amplitude = np.mean(amps)
    mean_rise_time = np.mean(rise_time)
    sum_peak_amplitude = np.sum(amps)
    sum_rise_time = np.sum(rise_time)

    # Frequency
    sma = np.sum(np.abs(eda))
    energy = np.sum(np.abs(eda) ** 2)
    power_range = fh[xh < 1]
    band_power_idx, _ = find_peaks(power_range, height=0.01)
    matrix = np.array([power_range[band_power_idx], band_power_idx])
    sorted_matrix = matrix[:, matrix[0, :].argsort()[::-1]]
    sorted_fh = sorted_matrix[0, :]
    if len(sorted_fh) < 5:
        m = 5 - len(sorted_fh)
        padd = np.zeros(m)
        sorted_fh = np.hstack((sorted_fh, padd))
    band_power = sorted_fh[:5]
    var_spectral_power = np.var(band_power)

    # DWT Wavelets
    levels = 4
    dwav = pywt.Wavelet('db3')
    dwt_coeffs = pywt.wavedec(eda, wavelet=dwav, level=levels)
    detailed_coeff = dwt_coeffs[1:]

    energy_wavelet = np.array([energy_(detailed_coeff, i) for i in range(levels)])
    total_energy_wavelet = np.sum(energy_wavelet)
    distribution_energy = np.array([100 * energy_wavelet[i] / total_energy_wavelet for i in range(levels)])
    entropy_wavelet = np.array([entropy(energy_wavelet[i]) for i in range(levels)])

    # MFCC
    mfccs = feature.mfcc(y=eda, sr=f_resampling, n_mfcc=20, n_fft=2048, hop_length=256)
    mean_mfccs = np.mean(mfccs, axis=-1)
    std_mfccs = np.std(mfccs, axis=-1)
    median_mfccs = np.median(mfccs, axis=-1)
    kurt_mfccs = scipy.stats.kurtosis(mfccs, axis=-1)
    skew_mfccs = scipy.stats.skew(mfccs, axis=-1)

    temp = np.hstack(
        (mean_eda, std_eda, kurt_eda, skew_eda, mean_derivative, mean_negative_derivative, activity, mobility,
         complexity, ap_entropy, samp_entropy, shan_entropy, fuzz_entropy, mse_entropy, cmse_entropy, rcmse_entropy,
         peaks_count, mean_peak_amplitude, mean_rise_time, sum_peak_amplitude, sum_rise_time, sma, energy,
         var_spectral_power, energy_wavelet, total_energy_wavelet, distribution_energy, entropy_wavelet, mean_mfccs,
         std_mfccs, median_mfccs, kurt_mfccs, skew_mfccs)
    )

    columns = [
        'EDA_mean', 'EDA_std', 'EDA_kurt', 'EDA_skew', 'EDA_mean_der', 'EDA_mean_neg_der', 'EDA_activity',
        'EDA_mobility', 'EDA_complexity', 'EDA_ApEn', 'EDA_SampEn', 'EDA_ShanEn', 'EDA_FuzzEn', 'EDA_MSE', 'EDA_CMSE',
        'EDA_RCMSE', 'EDA_peaks_count', 'EDA_mean_peaks_ampl', 'EDA_mean_rise_time', 'EDA_sum_peak_ampl',
        'EDA_sum_rise_time', 'EDA_sma', 'EDA_energy', 'EDA_var_spectral_power', 'EDA_energy_wavelet_lv1',
        'EDA_energy_wavelet_lv2', 'EDA_energy_wavelet_lv3', 'EDA_energy_wavelet_lv4', 'EDA_tot_energy_wavelet',
        'EDA_energy_distribution_lv1', 'EDA_energy_distribution_lv2', 'EDA_energy_distribution_lv3',
        'EDA_energy_distribution_lv4', 'EDA_entropy_wavelet_lv1', 'EDA_entropy_wavelet_lv2', 'EDA_entropy_wavelet_lv3',
        'EDA_entropy_wavelet_lv4'
    ]
    strings_mean = ['EDA_mean_MFCCS_' + str(j) for j in range(1, 21)]
    strings_std = ['EDA_std_MFCCS_' + str(j) for j in range(1, 21)]
    strings_median = ['EDA_median_MFCCS_' + str(j) for j in range(1, 21)]
    strings_kurt = ['EDA_kurt_MFCCS_' + str(j) for j in range(1, 21)]
    strings_skew = ['EDA_skew_MFCCS_' + str(j) for j in range(1, 21)]

    columns = columns + strings_mean + strings_std + strings_median + strings_kurt + strings_skew

    features_df = pd.DataFrame(
        [temp],
        columns=columns
    )

    return features_df


def extract_eda_nk_features(data, fs, f_resampling=None, window=30, ):
    if data.ndim == 2:
        data = np.squeeze(data)

    if f_resampling is None:
        f_resampling = fs

    new_num_samples = f_resampling * window
    eda = resample(data, new_num_samples)

    feats = nk.eda_process(eda, sampling_rate=f_resampling)
    return feats


def extract_eeg_features(
    data: np.array,
    fs: int,
) -> tuple[pd.DataFrame, list, dict]:

    power = nk.eeg_power(data.T, sampling_rate=fs)
    if 'Hz_' in '\t'.join(power.columns.tolist()):
        power = power.rename(columns={'Hz_30_80': 'Gamma', 'Hz_13_30': 'Beta', 'Hz_8_13': 'Alpha',
                                      'Hz_4_8': 'Theta',  'Hz_1_4': 'Delta', })
    features = power.copy()
    ratios = [(features.query("Channel == 'EEG_4'")['Theta'].values /
               features.query("Channel == 'EEG_6'")['Alpha'].values)[0],
              (features.query("Channel == 'EEG_4'")['Theta'].values /
               features.query("Channel == 'EEG_7'")['Alpha'].values)[0],
              (features.query("Channel == 'EEG_5'")['Theta'].values /
               features.query("Channel == 'EEG_7'")['Alpha'].values)[0],
              (features.query("Channel == 'EEG_5'")['Theta'].values /
               features.query("Channel == 'EEG_6'")['Alpha'].values)[0]]
    for ch in range(data.shape[1]):
        sampen, _ = nk.entropy_sample(data[:, ch], delay=8, dimension=2)
        de, _ = nk.entropy_differential(data[:, ch])

        features.loc[ch, 'Samp_En'] = sampen
        features.loc[ch, 'Diff_En'] = de

    info = {'shape': power.shape}
    return features, ratios, info


def pearson_correlation(w: np.array) -> np.array:
    corr_matrix = np.ones((12, 12))
    for ii in range(w.shape[1]):
        for jj in range(ii + 1, w.shape[1]):
            cov = np.sum((w[:, ii] - np.mean(w[:, ii])) * (w[:, jj] - np.mean(w[:, jj])))
            std_product = np.sqrt(np.sum((w[:, ii] - np.mean(w[:, ii])) ** 2)) * \
                np.sqrt(np.sum((w[:, jj] - np.mean(w[:, jj])) ** 2))
            corr_matrix[ii, jj] = cov / std_product
            corr_matrix[jj, ii] = corr_matrix[ii, jj]

    return corr_matrix


def phase_locking_value(w: np.array) -> np.array:
    plv_matrix = np.ones((12, 12))
    for ii in range(w.shape[1]):
        for jj in range(ii + 1, w.shape[1]):
            delta_theta = np.unwrap(np.angle(signal.hilbert(w[:, ii]))) - \
                          np.unwrap(np.angle(signal.hilbert(w[:, jj])))
            plv_matrix[ii, jj] = np.abs(np.sum(np.exp((0 + 1j) * delta_theta))) / w.shape[0]
            plv_matrix[jj, ii] = plv_matrix[ii, jj]

    return plv_matrix


def extract_resp_features(
        w: np.ndarray,
        fs: int,
) -> pd.DataFrame:
    columns = ['RSP_Mean', 'RSP_Max', 'RSP_Min',
               'RRV_RMSSD', 'RRV_MeanBB', 'RRV_SDBB', 'RRV_SDSD', 'RRV_CVBB',
               'RRV_CVSD', 'RRV_MedianBB', 'RRV_MadBB', 'RRV_MCVBB', 'RRV_VLF',
               'RRV_LF', 'RRV_HF', 'RRV_LFHF', 'RRV_LFn', 'RRV_HFn', 'RRV_SD1',
               'RRV_SD2', 'RRV_SD2SD1', 'RRV_ApEn', 'RRV_SampEn', 'RAV_Mean', 'RAV_SD', 'RAV_RMSSD', 'RAV_CVSD',
               'RSP_Symmetry_PeakTrough_Mean', 'RSP_Symmetry_PeakTrough_Median', 'RSP_Symmetry_PeakTrough_Max',
               'RSP_Symmetry_PeakTrough_Min', 'RSP_Symmetry_PeakTrough_Std',
               'RSP_Symmetry_RiseDecay_Mean', 'RSP_Symmetry_RiseDecay_Median', 'RSP_Symmetry_RiseDecay_Max',
               'RSP_Symmetry_RiseDecay_Min', 'RSP_Symmetry_RiseDecay_Std']
    w = np.ravel(w)
    try:
        peaks_signal, info = nk.rsp_peaks(w, fs)
        rsp_rate = nk.rsp_rate(w, sampling_rate=fs, method='xcorr')
        rsp_rate_mean = np.mean(rsp_rate)
        rsp_rate_max = np.max(rsp_rate)
        rsp_rate_min = np.min(rsp_rate)

        rrv = nk.rsp_rrv(rsp_rate, info, sampling_rate=fs)[
            ['RRV_RMSSD', 'RRV_MeanBB', 'RRV_SDBB', 'RRV_SDSD', 'RRV_CVBB',
             'RRV_CVSD', 'RRV_MedianBB', 'RRV_MadBB', 'RRV_MCVBB', 'RRV_VLF',
             'RRV_LF', 'RRV_HF', 'RRV_LFHF', 'RRV_LFn', 'RRV_HFn', 'RRV_SD1',
             'RRV_SD2', 'RRV_SD2SD1', 'RRV_ApEn', 'RRV_SampEn']
        ]

        symmetry = nk.rsp_symmetry(w, peaks_signal)
        symmetry_pt_mean = np.mean(symmetry['RSP_Symmetry_PeakTrough'])
        symmetry_pt_median = np.median(symmetry['RSP_Symmetry_PeakTrough'])
        symmetry_pt_max = np.max(symmetry['RSP_Symmetry_PeakTrough'])
        symmetry_pt_min = np.min(symmetry['RSP_Symmetry_PeakTrough'])
        symmetry_pt_std = np.std(symmetry['RSP_Symmetry_PeakTrough'])

        symmetry_rd_mean = np.mean(symmetry['RSP_Symmetry_RiseDecay'])
        symmetry_rd_median = np.median(symmetry['RSP_Symmetry_RiseDecay'])
        symmetry_rd_max = np.max(symmetry['RSP_Symmetry_RiseDecay'])
        symmetry_rd_min = np.min(symmetry['RSP_Symmetry_RiseDecay'])
        symmetry_rd_std = np.std(symmetry['RSP_Symmetry_RiseDecay'])

        rav = nk.rsp_rav(w, peaks_signal)[['RAV_Mean', 'RAV_SD', 'RAV_RMSSD', 'RAV_CVSD']]
        feats = ([rsp_rate_mean, rsp_rate_max, rsp_rate_min] + rrv.values.tolist()[0] + rav.values.tolist()[0] +
                 [symmetry_pt_mean, symmetry_pt_median, symmetry_pt_max, symmetry_pt_min, symmetry_pt_std,
                  symmetry_rd_mean, symmetry_rd_median, symmetry_rd_max, symmetry_rd_min, symmetry_rd_std])
        df = pd.DataFrame([feats], columns=columns)
        return df
    except Exception as e:
        print(e)
        return pd.DataFrame(np.empty((1, len(columns))) * np.nan, columns=columns)


if __name__ == "__main__":
    pass
