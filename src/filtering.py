import numpy as np
from neurokit2.signal import signal_smooth
import neurokit2 as nk
import scipy.signal as signal


def ecg_filter(ecg_signal, fs):
    f1 = 0.05
    f2 = 40
    filtered = ecg_signal

    filtered = signal.savgol_filter(
        np.ravel(filtered),
        polyorder=1,
        deriv=0,
        delta=1,
        mode='interp',
        window_length=4,
        axis=-1,
    )

    b, a = signal.butter(N=1, Wn=[f1, f2], btype='band', analog=False, fs=fs)
    filtered = signal.lfilter(b, a, filtered)

    filtered = nk.signal_detrend(filtered, sampling_rate=fs, method='locreg')

    return filtered


def microv_to_micros(eda):
    r = 6.49e6 / ((3.3 / eda) - 1)
    return (1 / r) * 1e6


def eda_filter(eda_signal, fs):
    # Parameters
    # eda_signal = microv_to_micros(eda_signal)
    order = 4
    frequency = 5
    frequency = 2 * np.array(frequency) / fs  # Normalize frequency to Nyquist Frequency (Fs/2).

    # Filtering
    b, a = signal.butter(N=order, Wn=frequency, btype="lowpass", analog=False, output="ba")
    filtered = signal.filtfilt(b, a, np.ravel(eda_signal))

    # Smoothing
    eda_cleaned = signal_smooth(filtered, method="convolution", kernel="boxzen", size=int(0.75 * fs))

    return eda_cleaned


def resp_filter(resp_signal, fs):
    # Parameters
    order = 2
    f1 = 0.03
    f2 = 5

    # Filtering
    b, a = signal.butter(N=order, Wn=[f1, f2], btype='band', analog=False, fs=fs)
    filtered = signal.filtfilt(b, a, np.ravel(resp_signal))

    return filtered


def emg_filter(emg_signal, fs):
    # Parameters (SENIAM recommendations)
    f1 = 10
    f2 = fs // 2 - 1
    
    # Band-pass Butterworth's filter
    b, a = signal.butter(N=3, Wn=[f1, f2], btype='band', analog=False, fs=fs)
    filtered = signal.lfilter(b, a, np.ravel(emg_signal))

    # De-trending
    filtered = signal.detrend(filtered)
    
    return filtered


def eeg_filter(eeg_signal, fs):
    b_bp, a_bp = signal.butter(N=2, Wn=[0.5, 40], btype='bandpass', fs=fs)
    b_n, a_n = signal.butter(N=1, Wn=[49, 51], btype='bandstop', fs=fs)
    
    filtered = np.empty_like(eeg_signal)
    for ch in range(eeg_signal.shape[1]):
        foo = signal.filtfilt(b_n, a_n, x=eeg_signal[:, ch])
        filtered[:, ch] = signal.filtfilt(b_bp, a_bp, x=foo)
    
    return filtered


if __name__ == "__main__":
    pass
