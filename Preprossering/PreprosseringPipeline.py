'''
Class to make preprosseing easy
test
Responsble Andreas
'''
import mne
import os
import torch
from collections import defaultdict
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from mne.io import read_raw_edf
from scipy import signal
from skimage.transform import resize

# Import David functions
from Preprossering.loadData import jsonLoad
import re


class preprossingPipeline:
    def __init__(self,mac=False,BC_datapath=r"/Users/villadsstokbro/Dokumenter/DTU/KID/3. "
                                            r"semester/Fagprojekt/BrainCapture/dataEEG"):
        """
        args BC datapath: your local path to bc dataset.
        """
        Wdir=os.getcwd()
        self.dataDir =BC_datapath
        if mac:
            jsonDir = os.path.join(Wdir, r"Preprossering/edfFiles.json")
            print(jsonDir)
            self.edfDict = jsonLoad(jsonDir)
            for key in self.edfDict.keys():
                self.edfDict[key]['path'][0]=re.sub(r'\\',r'/',self.edfDict[key]['path'][0])
        else:
            jsonDir = os.path.join(Wdir, r"Preprossering\edfFiles.json")
            self.edfDict = jsonLoad(jsonDir)




    def get_spectrogram(self,name):
        """
        Takes name of edf file.
        returns spectrograms
        """
        dataDict=self.readRawEdf(self.edfDict[name])
        dataDict["cleanData"]=self.filter(dataDict["rawData"])
        #tensor=self.spectrogramMake(dataDict["cleanData"])
        specktograms=self.slidingWindow(dataDict, tN=dataDict["cleanData"].last_samp,
                     tStep=dataDict["tStep"]*dataDict["fS"])
        specktograms["annotations"]=self.edfDict[name]["annotation"]
        return specktograms



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

    def spectrogramMake(self,EEGseries=None, t0=0, tWindow=120,resized=True):
        #Not debygged
        edfFs = EEGseries.info["sfreq"]
        chWindows = EEGseries.get_data(start=int(t0), stop=int(t0+tWindow))
        ch_dict=defaultdict()
        for i,ch in enumerate(EEGseries.ch_names):
            if resized:
                _, _, _, im = plt.specgram(chWindows[i], Fs = edfFs)
                image_resized = resize(im.get_array(), (224, 224), anti_aliasing = True)
                ch_dict[ch]=torch.tensor(image_resized)
            else:
                fTemp, tTemp, Sxx = signal.spectrogram(chWindows[i], fs=edfFs)
                ch_dict[ch]=torch.tensor(np.log(Sxx+np.finfo(float).eps)) # for np del torch.tensor

        return ch_dict


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

    def make_label(self,make_spectograms=False,make_from_names=None, quality=None, is_usable =None, max_files=10, path = '/Users/villadsstokbro/Dokumenter/DTU/KID/3. semester/Fagprojekt/spectograms_all_ch/',seed=0):
        i=0
        if quality is not None:
            label_dict = {key: str(int(self.edfDict[key]["annotation"]['Quality Of Eeg'])) for key in self.edfDict.keys()}
            fileNames = [key for key in self.edfDict.keys() if np.any(int(label_dict[key]) == np.array(quality))]
        elif is_usable is not None:
            label_dict = {key: is_usable for key in
                            self.edfDict.keys() if self.edfDict[key]["annotation"]["Is Eeg Usable For Clinical Purposes"] == is_usable }
            fileNames = list(label_dict.keys())
        elif make_from_names is not None:
            label_dict = {key:key for key in make_from_names}
            fileNames = make_from_names

        else:
            label_dict={key:key for key in self.edfDict.keys()}
            fileNames = list(self.edfDict.keys())
        np.random.seed(seed)
        np.random.shuffle(fileNames)
        filenames=[]
        for filename in fileNames:
            if i==max_files:
                break
            if not os.path.exists(path+filename + '.npy'):
                pass
            else:
                if i == 0:
                    spectogram = np.load(path+filename + '.npy')
                    if make_spectograms:
                        spectogram=spectogram.squeeze()
                    spectograms = spectogram
                    labels = spectogram.shape[0]*[label_dict[filename]]
                    filenames.append(filename)
                    i+=1
                else:
                    spectogram = np.load(path + filename + '.npy')
                    if make_spectograms:
                        spectogram=spectogram.squeeze()
                    spectograms = np.vstack((spectograms, spectogram))
                    label=spectogram.shape[0]*[label_dict[filename]]
                    labels=labels+label
                    filenames.append(filename)
                    i+= 1

        return spectograms, labels, filenames


def plot_spectrogram(windows,win_idx,ch_idx):
    win_name=windows.keys()[win_idx]
    ch_name=windows[win_name].keys()
    plt.plot(windows[win_name][ch_name])
    plt.show()
#Debugging

def getFeatureVecWholeFile(filePath):
    spectrogramDict = C.get_spectrogram(filePath)
    windowVec = []
    for windowName in spectrogramDict:
        windowValues = spectrogramDict[windowName]
        featureVec = []
        for channelSpectrogram in windowValues.values():
            #channelValue = windowValues[channelName]
            tempFeatureVec = model(channelSpectrogram.unsqueeze(0).unsqueeze(0).float())
            if len(featureVec)==0:
                featureVec = tempFeatureVec
            else:
                featureVec = torch.cat((featureVec, tempFeatureVec), 1)
        if len(windowVec) ==0:
            windowVec = featureVec
        else:
            windowVec = torch.cat((windowVec,featureVec),0)
                #windowFeatureVec.append(tempFeatureVec)
    return windowVec

def getFeatureVec(windowValues,model):
    featureVec = []
    for channelSpectrogram in windowValues.values():
        #channelValue = windowValues[channelName]
        tempFeatureVec = model(channelSpectrogram.unsqueeze(0).unsqueeze(0).float())
        if len(featureVec)==0:
            featureVec = tempFeatureVec
        else:
            featureVec = torch.cat((featureVec, tempFeatureVec), 1)
    return featureVec

