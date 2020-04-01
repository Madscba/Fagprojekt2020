'''
Class to make preprosseing easy

Responsble Andreas
'''
import os, mne, torch,json
from collections import defaultdict
from datetime import datetime
import numpy as np
from mne.io import read_raw_edf
from scipy import signal
import matplotlib.pyplot as plt

#Import David functions
from Davidclean import jsonLoad,readRawEdf

class preprossingPipeline:
    def __init__(self,BC_datapath= r"C:\Users\Andreas\Desktop\KID\Fagproject\Data\BC"):
        """
        To do make it posiple to change settings for now just init something that works 
        """
        Wdir=os.getcwd()
        self.dataDir =os.path.join(Wdir,BC_datapath)
        jsonDir =os.path.join(Wdir,r"Preprossering\edfFiles.json")
        self.edfDict = jsonLoad(jsonDir)


    def get_spectrogram(self,name):
        """
        Takes name of spectrogram
        returns spectrograms
        """
        dataDict=self.readRawEdf(self.edfDict[name])
        dataDict["cleanData"]=self.filter(dataDict["rawData"])
        #tensor=self.spectrogramMake(dataDict["cleanData"])
        specktogram=self.slidingWindow(dataDict, tN=dataDict["cleanData"].last_samp,
                     tStep=dataDict["tStep"]*dataDict["fS"],
                     )
        return spectrogram



    def readRawEdf(self,edfDict=None, read_raw_edf_param={'preload':True, 'stim_channel':'auto'}, tWindow=120, tStep=30):
        edfDict["rawData"] = read_raw_edf(os.path.join(self.dataDir,edfDict["path"][0]), **read_raw_edf_param)
        tStart = edfDict["rawData"].annotations.orig_time - 60*60
        tLast = int((1+edfDict["rawData"].last_samp)/edfDict["rawData"].info["sfreq"])
        edfDict["t0"] = datetime.fromtimestamp(tStart)
        edfDict["tN"] = datetime.fromtimestamp(tStart + tLast),
        edfDict["tWindow"] = tWindow
        edfDict["tStep"] = tStep
        edfDict["fS"] = edfDict["rawData"].info["sfreq"]
        return edfDict

    # pre-processing pipeline single file
    def filter(self,EEGseries=None, lpfq=1, hpfq=40, notchfq=50):
        """
        Credit david. 
        Original name pipeline
        """
        # EEGseries.plot
        EEGseries.set_montage(mne.channels.read_montage(kind='easycap-M1', ch_names=EEGseries.ch_names))
        # EEGseries.plot_psd()
        EEGseries.notch_filter(freqs=notchfq, notch_widths=5)
        # EEGseries.plot_psd()
        EEGseries.filter(lpfq, hpfq, fir_design='firwin')
        EEGseries.set_eeg_reference()
        # EEGseries.plot_sensors(show_names=True)
        return EEGseries

    def spectrogramMake(self,EEGseries=None, t0=0, tWindow=120):
        edfFs = EEGseries.info["sfreq"]
        chWindows = EEGseries.get_data(start=int(t0), stop=int(t0+tWindow))
        #t, f, Sxx = signal.spectrogram(chWindows, fs=edfFs)
        #plt.pcolormesh(f, t, Sxx)
        fTemp, tTemp, Sxx = signal.spectrogram(chWindows, fs=edfFs)
        plt.pcolormesh(tTemp, fTemp, np.log(Sxx[0]))
         #plt.ylabel('Frequency [Hz]')
         #plt.xlabel('Time [sec]')
         #plt.title("channel spectrogram: "+EEGseries.ch_names[count])
         #plt.show()
        pxx, freqs, bins, im = plt.specgram(chWindows, Fs = edfFs,cmap='gray')
        img = img.resize((224, 224))
        a = np.asarray(img)
        b = np.empty_like(a, dtype=float) # DIM: (224,224,3)
        min_max_scaler = preprocessing.MinMaxScaler() #Rescale values to interval 0-1
        for i in range(a.shape[2]):
            a_stand = min_max_scaler.fit_transform(a[:, :, i])
            b[:, :, i] = a_stand
        b = b.transpose((2, 0, 1))
        img2 = torch.from_numpy(b)
        return torch.tensor(np.log(Sxx+np.finfo(float).eps)) # for np del torch.tensor

    def slidingWindow(self,edfInfo=None, tN=0, tStep=60, localSave={"sliceSave":False, "saveDir":os.getcwd()}):
        windowEEG = defaultdict(list)
        sampleWindow = edfInfo["tWindow"]*edfInfo["fS"]
        for i in range(0, tN, int(tStep)):
            windowKey = "window_%i_%i" % (i, i+sampleWindow)
            windowEEG[windowKey] = self.spectrogramMake(edfInfo["rawData"], t0=i, tWindow=sampleWindow)
        if (1+tN) % int(tStep) != 0:
            windowKey = "window_%i_%i" % (int(tN-sampleWindow), int(tN))
            windowEEG[windowKey] = self.spectrogramMake(edfInfo["rawData"], t0 = int(tN-sampleWindow), tWindow = sampleWindow)
        if localSave["sliceSave"]:
            idDir = edfInfo["rawData"].filenames[0].split('\\')[-1].split('.')[0]
            if not os.path.exists(localSave["saveDir"]):
                os.mkdir(saveDir + "tempData\\")
            if not os.path.exists(saveDir + "tempData\\" + idDir):
                os.mkdir(saveDir + "tempData\\" + idDir)
            for k,v in windowEEG.items():
                torch.save(v, saveDir + "tempData\\%s\\%s.pt" % (idDir, k)) # for np del torch.tensor
        if not localSave["sliceSave"]:
            windowOut = windowEEG.copy()
        else:
            windowOut = None
        return windowOut

#Debugging
C=preprossingPipeline()
C.get_spectrogram("sbs2data_2018_09_01_08_04_51_328.edf")    