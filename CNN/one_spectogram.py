
C=preprossingPipeline(BC_datapath=r"/Users/villadsstokbro/Dokumenter/DTU/KID/3. semester/Fagprojekt/BrainCapture/dataEEG",mac=True)
fileNames=C.edfDict.keys()
wdir="/Volumes/B"
k=0
for file in fileNames:
        spec = C.get_spectrogram(file)
        break
