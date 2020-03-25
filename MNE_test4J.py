import numpy as np

import mne
from mne.datasets import sample
from mne.preprocessing import create_ecg_epochs, create_eog_epochs

# getting some data ready
data_path = sample.data_path()
raw_fname = data_path + '/MEG/sample/sample_audvis_raw.fif'

raw = mne.io.read_raw_fif(raw_fname, preload=True)

tmin, tmax = 0, 30  # use the first 20s of data
raw.crop(tmin, tmax).load_data()
raw.info['bads'] = ['MEG 2443', 'EEG 053']  # bads + 2 more
fmin, fmax = 2, 300  # look at frequencies between 2 and 300Hz
n_fft = 2048  # the FFT size (n_fft). Ideally a power of 2


# Pick a subset of channels (here for speed reason)
selection = mne.read_selection('Left-temporal')
picks = mne.pick_types(raw.info, meg='mag', eeg=False, eog=False,
                       stim=False, exclude='bads', selection=selection)

# Let's first check out all channel types
raw.plot_psd(area_mode='range', tmax=10.0, picks=picks, average=False)

# band-pass filtering in the range 1 Hz - 50 Hz
raw.filter(1, 50., l_trans_bandwidth='auto', h_trans_bandwidth='auto',
           filter_length='auto', phase='zero')

raw.resample(150, npad="auto")  # set sampling frequency to 150Hz
raw.plot_psd(area_mode='range', tmax=10.0, picks=picks)