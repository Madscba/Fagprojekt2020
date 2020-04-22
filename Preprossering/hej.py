def make_label(self, make_spectograms=True, quality=None, is_usable=None, max_files=10,
               path='/Users/villadsstokbro/Dokumenter/DTU/KID/3. semester/Fagprojekt/spectograms_all_ch/', seed=0):
    i = 0
    if quality is not None:
        label_dict = {key: int(self.edfDict[key]["annotation"]['Quality Of Eeg']) for key in self.edfDict.keys()}
        fileNames = [key for key in self.edfDict.keys() if np.any(label_dict[key] == np.array(quality))]
    elif is_usable is not None:
        usable_category = {'No': 0, 'Yes': 1}
        label_dict = {key: usable_category[is_usable] for key in
                      self.edfDict.keys() if
                      self.edfDict[key]["annotation"]["Is Eeg Usable For Clinical Purposes"] == is_usable}
        fileNames = list(label_dict.keys())
    else:
        label_dict = {key: idx for idx, key in enumerate(self.edfDict.keys())}
        fileNames = list(self.edfDict.keys())
    np.random.seed(seed)
    np.random.shuffle(fileNames)
    for filename in fileNames:
        if i == max_files:
            break
        if not os.path.exists(path + filename + '.npy'):
            pass
        else:
            if i == 0:
                spectogram = np.load(path + filename + '.npy')
                if make_spectograms:
                    spectogram = spectogram.squeeze()
                spectograms = spectogram
                labels = np.ones(spectogram.shape[0]) * label_dict[filename]
            else:
                spectogram = np.load(path + filename + '.npy')
                if make_spectograms:
                    spectogram = spectogram.squeeze()
                spectograms = np.vstack((spectograms, spectogram))
                print(label_dict[filename])
                label = np.ones(spectogram.shape[0]) * label_dict[filename]
                labels = np.concatenate((labels, label))
        i += 1
    return spectograms, labels