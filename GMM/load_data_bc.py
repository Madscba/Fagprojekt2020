import os
import numpy as np
import mne
import matplotlib.pyplot as plt
import pandas as pd

#file_name_xlsx = 'MGH File Annotations.xlsx'
#path_xlsx_file = r'C:/Users/johan/iCloudDrive/DTU/KID/4. semester/Fagprojekt/Data/dataEEG/dataEEG/'

def read_excel(file_name_xlsx,path_xlsx_file):
    file_names = []
    quality = []

    #Only for the first time
    # =============================================================================
    for i in range(4):
        annotation=pd.read_excel(path_xlsx_file+file_name_xlsx, sheet_name=i+2)
        annotation_path = annotation['Recording']
        annotation_Quality = annotation['Quality Of Eeg']
        for i in range(len(annotation_path)):
             if type(annotation_Quality[i]) == int:
                 file_names.append(annotation_path[i].split('/')[-1])
                 quality.append(int(int(annotation_Quality[i])))

    # =============================================================================
    return file_names, quality

file_name_xlsx = 'MGH File Annotations.xlsx'
path_xlsx_file = r'C:/Users/johan/iCloudDrive/DTU/KID/4. semester/Fagprojekt/Data/dataEEG/'

file_names, quality = read_excel(file_name_xlsx,path_xlsx_file)

#def data_processing(file_names, path_xlsx_file, quality):
path = path_xlsx_file
fileName_quality = []

idx = None

for folder, subfolder, filenames in os.walk(path):

    for edf_file in filenames:
        idx = None

        # search for filename index
        for i, file_name in enumerate(file_names[0:500]):
            if file_name == edf_file:
                idx = i
                break
        # create list of tuples contaning the patg to the fil and gender
        if idx != None:
            fileName_quality.append((folder + '/' + edf_file, quality[idx]))
        else:
            continue

window_size=60
i = 0
for filename, quality in fileName_quality[0:500]:
    raw  = mne.io.read_raw_edf(filename)

    freq = raw.info['sfreq']
    num_samples = raw.n_times
    lower_bound = list(range(0, int(num_samples/freq)-window_size, window_size-int(window_size/2)))
    upper_bound = list(range(window_size, int(num_samples/freq), window_size-int(window_size/2)))
    if i==0:
        bound_list=np.array([0,len(lower_bound)])
    else:
         bound_list=np.append(bound_list,np.array([i,i+len(lower_bound)]))
    for low, up in zip(lower_bound,upper_bound):
        raw_subset = raw.copy()
        raw_subset.crop(tmin = low, tmax = up).load_data()
        raw_subset.filter(l_freq=4, h_freq=40)
        data = raw_subset.get_data()
        data = data[0]
        ps, _, _, _ =plt.specgram(data, Fs = freq)
        s = ps.shape
        ps = ps.reshape(1,s[0], s[1], 1)
        if i == 0:
            imgs = ps
        else:
            imgs = c = np.concatenate((imgs,ps), axis=0)
        if i==200:
            break
        i+=1