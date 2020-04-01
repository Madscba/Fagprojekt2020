"""
Clean funktions from david

Responsble Andreas
"""

import os, mne, torch,json
from collections import defaultdict
from datetime import datetime
from mne.io import read_raw_edf
import numpy as np
# from LoadFarrahTueData.loadData import jsonLoad, pathLoad
from loadData import jsonLoad
from scipy import signal
import matplotlib.pyplot as plt

# load from json to dict
def jsonLoad(path = False):
    if path is False:
        sys.exit("no path were given to load Json")
    else:
        with open(path, "r") as read_file:
            edfDefDict = json.load(read_file)
    print("\npaths found for loading")
    return edfDefDict

def readRawEdf(edfDict=None, read_raw_edf_param={'preload':True, 'stim_channel':'auto'}, tWindow=120, tStep=30):
    edfDict["rawData"] = read_raw_edf(saveDir+edfDict["path"][0], **read_raw_edf_param)
    tStart = edfDict["rawData"].annotations.orig_time - 60*60
    tLast = int((1+edfDict["rawData"].last_samp)/edfDict["rawData"].info["sfreq"])
    edfDict["t0"] = datetime.fromtimestamp(tStart)
    edfDict["tN"] = datetime.fromtimestamp(tStart + tLast),
    edfDict["tWindow"] = tWindow
    edfDict["tStep"] = tStep
    edfDict["fS"] = edfDict["rawData"].info["sfreq"]
    return edfDict

# pre-processing pipeline single file
def pipeline(EEGseries=None, lpfq=1, hpfq=40, notchfq=50):
    # EEGseries.plot
    EEGseries.set_montage(mne.channels.read_montage(kind='easycap-M1', ch_names=EEGseries.ch_names))
    # EEGseries.plot_psd()
    EEGseries.notch_filter(freqs=notchfq, notch_widths=5)
    # EEGseries.plot_psd()
    EEGseries.filter(lpfq, hpfq, fir_design='firwin')
    EEGseries.set_eeg_reference()
    # EEGseries.plot_sensors(show_names=True)
    return EEGseries

def spectrogramMake(EEGseries=None, t0=0, tWindow=120):
    edfFs = EEGseries.info["sfreq"]
    chWindows = EEGseries.get_data(start=int(t0), stop=int(t0+tWindow))
    _, _, Sxx = signal.spectrogram(chWindows, fs=edfFs)
    # fTemp, tTemp, Sxx = signal.spectrogram(chWindows, fs=edfFs)
    # plt.pcolormesh(tTemp, fTemp, np.log(Sxx))
    # plt.ylabel('Frequency [Hz]')
    # plt.xlabel('Time [sec]')
    # plt.title("channel spectrogram: "+EEGseries.ch_names[count])
    # plt.show()
    return torch.tensor(np.log(Sxx+np.finfo(float).eps)) # for np del torch.tensor