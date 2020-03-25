# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 21:08:39 2020

@author: Johannes
"""

import os
import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt
from PIL import Image
from random import uniform
from IPython.utils import io

os.chdir(r'C:\Users\johan\iCloudDrive\DTU\KID\4. semester\Fagprojekt\dataEEG.zip\dataEEG')
print(os.getcwd())
path_xlsx_file = r'\dataEEG'
file_name_xlsx = r'\MGH_File_Annotations.xlsx'
new_path = r'C:\Users\Mads-\Documents\Universitet\4. Semester\02466 Fagprojekt - Bachelor i kunstig intelligens og data\dataEEG\dataEEG\MGH_File_Annotations.xlsx'
print(os.path.join(path_xlsx_file + file_name_xlsx))
print(new_path)
annotation = pd.read_excel(new_path, sheet_name=3)
# annotation = pd.read_excel(path_xlsx_file+file_name_xlsx, sheet_name= 3)
# annotation.head()


annotation_path = annotation['Recording']
annotation_Quality = annotation['Quality Of Eeg']

file_names = []
quality = []

for i in range(len(annotation_path)):
    if type(annotation_Quality[i]) == int:
        file_names.append(annotation_path[i].split('/')[-1])
        quality.append(int(int(annotation_Quality[i]) > 3))

path = 'dataEEG/'
fileName_quality = []

idx = None

for folder, subfolder, filenames in os.walk(path):
    for edf_file in filenames:

        idx = None

        # search for filename index
        for i, file_name in enumerate(file_names):
            if file_name == edf_file:
                idx = i
                break

        # create list of tuples contaning the patg to the fil and gender
        if idx != None:
            fileName_quality.append((folder + '/' + edf_file, quality[idx]))
        else:
            continue

window_size = 30
count = 0
channel = 0

# Creating data
for path, q in fileName_quality:
    with io.capture_output() as captured:
        raw = mne.io.read_raw_edf(path);
        freq = raw.info['sfreq'];
        num_samples = raw.n_times
    lower_bound = list(range(0, int(num_samples / freq) - window_size, window_size - int(window_size / 2)))
    upper_bound = list(range(window_size, int(num_samples / freq), window_size - int(window_size / 2)))

    # for low, up in zip(lower_bound,upper_bound):
    #     # Copy the data so we stil have the full data file after crop
    #     with io.capture_output() as captured:
    #         raw_subset = raw.copy();

    #     # crop the copy
    #     with io.capture_output() as captured:
    #         raw_subset.crop(tmin = low, tmax = up).load_data();

    #     # filter the data
    #     with io.capture_output() as captured:
    #         raw_subset.filter(l_freq=0.4, h_freq=40);
    #     # extract the data values
    #     with io.capture_output() as captured:
    #         data = raw_subset.get_data();

    #     # Choose channel
    #     data = data[channel]
    #     #raw_subset.plot()
    #     # Create spectrogram
    #     pxx, freqs, bins, im = plt.specgram(data, Fs = freq,noverlap =200,NFFT = 201,scale= 'dB')
    #     pxx = np.log1p(pxx)
    #     #plt.yscale('log')
    #     #plt.axis('off')
    #     plt.show()

    #     # decide between test or train
    # test_or_train = uniform(0, 1)

    # save spectrogram into correct folder
    # if q == 1: # Tjek if current data is recorded on male
    #     Img_name = '3_or_above'+str(count)+'_'+str(channel)+'.jpg'
    #     if test_or_train < 0.9:
    #         plt.savefig(r'C:\Users\Mads-\Documents\Universitet\4. Semester\02466 Fagprojekt - Bachelor i kunstig intelligens og data\dataEEG\dataEEG\spectrograms\train\3_or_above'+Img_name, edgecolor = None, bbox_inches='tight',pad_inches = 0)
    #     else:
    #         plt.savefig(r'C:\Users\Mads-\Documents\Universitet\4. Semester\02466 Fagprojekt - Bachelor i kunstig intelligens og data\dataEEG\dataEEG\spectrograms\train\3_or_above'+Img_name, edgecolor = None, bbox_inches='tight',pad_inches = 0)
    # else:
    #     Img_name = 'below_3'+str(count)+'_'+str(channel)+'.jpg'
    #     if test_or_train < 0.9:
    #         plt.savefig(r'C:\Users\Mads-\Documents\Universitet\4. Semester\02466 Fagprojekt - Bachelor i kunstig intelligens og data\dataEEG\dataEEG\spectrograms\train\below_3'+Img_name, edgecolor = None, bbox_inches='tight',pad_inches = 0)
    #     else:
    #         plt.savefig(r'C:\Users\Mads-\Documents\Universitet\4. Semester\02466 Fagprojekt - Bachelor i kunstig intelligens og data\dataEEG\dataEEG\spectrograms\train\below_3'+Img_name, edgecolor = None, bbox_inches='tight',pad_inches = 0)
    # del raw_subset
    # plt.close()
    # count +=1
    del raw
    # breaks +=1
    # if breaks > 0:
    # break