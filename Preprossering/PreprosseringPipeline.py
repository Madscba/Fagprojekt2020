'''
Class to make preprosseing easy

test
Responsble Andreas
'''
import mne
import os
import torch
from collections import defaultdict
from datetime import datetime,timedelta
from flashtorch.utils import load_image
import matplotlib.pyplot as plt
import numpy as np
from mne.io import read_raw_edf
from scipy import signal
from Villads.PCA_TSNE_classes import scale_data
from torchvision import transforms
import pickle

# Import David functions
from Preprossering.loadData import jsonLoad
import re
import io


class preprossingPipeline:
    def __init__(self,BC_datapath,resize=True,filters={"lpfq": 1, "hpfq": 40, "notchfq": 50},mac=False):
        """
        args BC datapath: your local path to bc dataset.
        mac: set true if you are using a mac
        args BC datapath: your local path to bc dataset: 
        resize: bool if true rezise spectrogram to 224*224
        filters: dict: with index "lpfq": , "hpfq":, "notchfq": if and idex is missing the filter will not be applied
        """
        Wdir=os.getcwd()
        self.dataDir =BC_datapath
        self.filters=filters
        self.resized=resize
        if mac:
            jsonDir = os.path.join(Wdir, r"Preprossering/edfFiles.json")
            print(jsonDir)
            self.edfDict = jsonLoad(jsonDir)
            for key in self.edfDict.keys():
                self.edfDict[key]['path'][0]=re.sub(r'\\',r'/',self.edfDict[key]['path'][0])
        else:
            jsonDir = os.path.join(Wdir, r"Preprossering/edfFiles.json")
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

        #comment this out to get meta data on recording time stamps, WARNING will given and error in python 3.7
        #tStart = edfDict["rawData"].annotations.orig_time-timedelta(hours=1)
        #tLast = int((1+edfDict["rawData"].last_samp)/edfDict["rawData"].info["sfreq"])
        #edfDict["t0"] = tStart
        #edfDict["tN"] = tStart + timedelta(seconds=tLast)

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
       # EEGseries.plot()
        EEGseries.set_montage(mne.channels.make_standard_montage(kind='easycap-M1', head_size=0.095))
        #EEGseries.plot_psd()
        EEGseries.notch_filter(freqs=notchfq, notch_widths=5)
        #EEGseries.plot_psd()
        EEGseries.filter(lpfq, hpfq, fir_design='firwin')
        EEGseries.set_eeg_reference()
        #EEGseries.plot_sensors(show_names=True)
        return EEGseries

    def spectrogramMake(self,EEGseries=None, t0=0, tWindow=120,resized=False):
        #Not debygged
        edfFs = EEGseries.info["sfreq"]
        chWindows = EEGseries.get_data(start=int(t0), stop=int(t0+tWindow))
        ch_dict=defaultdict()
        for i,ch in enumerate(EEGseries.ch_names):
            if resized:
                fTemp, tTemp, Sxx = signal.spectrogram(chWindows[i], fs=edfFs)
                #ch_dict[ch]=torch.tensor(image_resized)
                buf = io.BytesIO()
                plt.imsave(buf, np.log(Sxx+np.finfo(float).eps)[0:90], format='png')
                buf.seek(0)
                image = load_image(buf)
                img = apply_transforms_new(image)
                buf.close()
                ch_dict[ch] = img
            else:
                fTemp, tTemp, Sxx = signal.spectrogram(chWindows[i], fs=edfFs)
                ch_dict[ch]=torch.tensor(np.log(Sxx+np.finfo(float).eps)) # for np del torch.tensor
        return ch_dict

    def plot_window(self,name,win_idx,type="spec",plot=True):
        """
        ARGS:\\
        name: str name of file\\
        window_idx: int: index of window\\
        type: str: spec og EEG\\
        plot: bool if true plot, else returns values NOT implementet 
        """
        dataDict=self.readRawEdf(self.edfDict[name])
        EEGserie=self.filter(dataDict["rawData"])
        #tN=dataDict["cleanData"].last_samp,
        tStep=dataDict["tStep"]*dataDict["fS"]


        
        if type=="spec":
            sampleWindow = dataDict["tWindow"]*dataDict["fS"]
            t0=win_idx*int(tStep)
            spectrograms=self.spectrogramMake(EEGserie, t0=t0, tWindow=sampleWindow,resized=False)
            col=5 #Images per row
            row=np.ceil(len(spectrograms.keys())/col)
            for i,key in enumerate(spectrograms.keys()):
                if plot:
                    plt.subplot(row,col,i+1)
                    plt.imshow(spectrograms[key])
                    plt.xticks([])
                    plt.yticks([])
                    plt.title(key)
                else:
                    return spectrograms
            
            plt.show()
        elif type=="EEG":
            if plot:
                fig=EEGserie.plot(start=win_idx*30,duration=4,scalings ="auto")
                fig.show()
            else:
                EEGserie.crop(win_idx * 30, win_idx * 30 + 120)
                DF = EEGserie.to_data_frame()
                return DF


        else:
            raise ("type not specified, use Spec or EEG")


            
                
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


    def make_label(self, make_from_filenames=None, quality=None, is_usable=None, max_files=10, max_windows = 0,
                   path='/Users/villadsstokbro/Dokumenter/DTU/KID/3. semester/Fagprojekt/spectograms_all_ch/', seed=0):
        """
        Funktion til at retunere labels på udvalgte EEG-recordings, det kan både være på spectogrammer
        eller feature vectors.
        Args:
            make_from_filenames: if not none, list of strings including the names of the files
                            you want to include
            quality: if not none, list of integers including the quality_scores you want to include
            is_usable: if not none, either 'Yes' or 'No'
            max_files: maximum number of files to include
            path: local path to folder containing either spectograms of feature vectors
            seed: to make reproduction possible
        Output:
            Windows: array of windows, dimension_0=window, dimension_1=values for spectograms or
                    feature vectors
            labels: list of strings containg labels for each window
            filenames: the filenames included
            window_idx_full: list of tuples, with the first index corresponding to the first
                             window. Tuple contains name of the i'th window and the window idx for
                             for that recording
        """
        edfDict_keys=list(self.edfDict.keys())
        edfDict_keys.sort()
        i = 0
        if quality is not None:
            label_dict = {key: str(int(self.edfDict[key]["annotation"]['Quality Of Eeg'])) for key in edfDict_keys}
            fileNames = [key for key in edfDict_keys if np.any(int(label_dict[key]) == np.array(quality))]
        elif is_usable is not None:
            label_dict = {key: is_usable for key in
                          edfDict_keys if
                          self.edfDict[key]["annotation"]["Is Eeg Usable For Clinical Purposes"] == is_usable}
            fileNames = list(label_dict.keys())
        elif make_from_filenames is not None:
            label_dict = {key: key for key in make_from_filenames}
            fileNames = make_from_filenames

        else:
            label_dict = {key: key for key in edfDict_keys}
            fileNames = list(edfDict_keys)

        np.random.seed(seed)
        np.random.shuffle(fileNames)
        filenames = []
        for filename in fileNames:
            fv_path = os.path.join(path, filename + '.npy')
            if i == max_files:
                break
            if not os.path.exists(fv_path):
                print(f"Warning could't find {fv_path}")
                pass
            else:
                if i == 0:
                    window = np.load(fv_path).squeeze()
                    if max_windows > window.shape[0]:
                        integer = 1
                    else:
                        integer = int(round(window.shape[0]/max_windows))
                    window = window.squeeze()
                    window_new=window[::integer]
                    window_idx_full=[(filename,idx) for idx in range(0,window.shape[0],integer)]
                    labels = window_new.shape[0] * [label_dict[filename]]
                    windows = window_new
                    filenames.append(filename)
                    i += 1
                else:
                    window = np.load(fv_path).squeeze()
                    if max_windows > window.shape[0]:
                        integer = 1
                    else:
                        integer = int(round(window.shape[0] / max_windows))
                    window_new = window[::integer]
                    window_idx = [(filename, idx) for idx in range(0, window.shape[0], integer)]
                    window_idx_full = window_idx_full + window_idx
                    label = window_new.shape[0] * [label_dict[filename]]
                    labels = labels + label
                    windows = np.vstack((windows, window_new))
                    filenames.append(filename)
                    i += 1
        return windows, labels, filenames, window_idx_full

    def make_label_cnn(self, make_from_filenames=None, quality=None, is_usable=None, max_files=10, max_windows = 1000,
                   path=None, seed=0, ch_to_include=range(14)):

        edfDict_keys=list(self.edfDict.keys())
        edfDict_keys.sort()
        i = 0
        if quality is not None:
            label_dict = {key: str(int(self.edfDict[key]["annotation"]['Quality Of Eeg'])) for key in edfDict_keys}
            fileNames = [key for key in edfDict_keys if np.any(int(label_dict[key]) == np.array(quality))]
        elif is_usable is not None:
            label_dict = {key: is_usable for key in
                          edfDict_keys if
                          self.edfDict[key]["annotation"]["Is Eeg Usable For Clinical Purposes"] == is_usable}
            fileNames = list(label_dict.keys())
        elif make_from_filenames is not None:
            label_dict = {key: key for key in make_from_filenames}
            fileNames = make_from_filenames

        else:
            label_dict = {key: key for key in edfDict_keys}
            fileNames = list(edfDict_keys)

        np.random.seed(seed)
        np.random.shuffle(fileNames)
        filenames = []
        for filename in fileNames:
            fv_path = os.path.join(path, filename + '.pt')
            if i == max_files:
                break
            if not os.path.exists(fv_path):
                pass
            else:
                if i == 0:
                    window = torch.load(fv_path)[:,ch_to_include,:,:]
                    if max_windows > window.shape[0]:
                        integer = 1
                    else:
                        integer = int(round(window.shape[0]/max_windows))

                    window_new=window[::integer,ch_to_include,:,:]
                    window_idx_full=[[filename,idx,ch] for idx in range(0,window.shape[0],integer) for ch in ch_to_include]
                    labels = len(ch_to_include)*window_new.shape[0] * [label_dict[filename]]

                    filenames.append(filename)
                    window_new = window_new.reshape(-1, 3, 224, 224)
                    windows = window_new
                    i += 1
                else:
                    print("Reached:",fv_path)
                    window = torch.load(fv_path)[:, ch_to_include, :, :]
                    if max_windows>window.shape[0]:
                        integer=1
                    else:
                        integer = int(round(window.shape[0]/max_windows))
                    window_new = window[::integer, ch_to_include, :, :]
                    window_idx = [[filename, idx, ch] for idx in range(0,window.shape[0], integer) for ch in
                                       ch_to_include]
                    print(window_idx)
                    label = len(ch_to_include) * window_new.shape[0] * [label_dict[filename]]
                    labels = labels + label
                    window_idx_full=window_idx_full+window_idx
                    window_new = window_new.reshape(-1, 3, 224, 224)
                    windows = torch.cat((windows, window_new), dim=0)
                    filenames.append(filename)
                    i += 1

        return windows.detach().requires_grad_(requires_grad=True), labels, filenames, window_idx_full

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

def getNewFeatureVec(windowValues,model):
    featureVec = []
    for channelSpectrogram in windowValues.values():
        x = model.features(channelSpectrogram.unsqueeze(0).unsqueeze(0).float())
        x = model.avgpool(x)
        x = torch.flatten(x,1)
        tempFeatureVec = model.classifer[0:4](x)
        if len(featureVec)==0:
            featureVec = tempFeatureVec
        else:
            featureVec = torch.cat((featureVec, tempFeatureVec), 1)
    return featureVec.detach()

def make_pca(windows,make_spectograms=False):
    if make_spectograms:
        path_pca = os.path.join(os.getcwd(), 'Villads', 'PCA_spectograms.sav')
    else:
        path_pca = os.path.join(os.getcwd(), 'Villads', 'PCA_feature_vectors_1.sav')
    pca = pickle.load(open(path_pca, 'rb'))
    windows = scale_data(windows)
    windows = pca.transform(windows)
    return windows

def apply_transforms_new(image, size=(224,224)):


    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]

    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(means, stds)
    ])

    tensor = transform(image).unsqueeze(0)

    tensor.requires_grad = True

    return tensor




if __name__ == "__main__":
    c=preprossingPipeline(BC_datapath=r"C:\Users\Andre\Desktop\Fagproject\Data\BC\data_farrahtue_EEG")
    c.make_label(make_from_filenames=['sbs2data_2018_09_03_11_38_39_138.edf',])
