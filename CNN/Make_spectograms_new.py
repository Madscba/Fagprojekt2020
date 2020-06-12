from Preprossering.PreprosseringPipeline import preprossingPipeline
import os
import torch




C=preprossingPipeline(BC_datapath=r"/Users/villadsstokbro/Dokumenter/DTU/KID/3. semester/Fagprojekt/BrainCapture/dataEEG",mac=True)
fileNames=C.edfDict.keys()
wdir="/Volumes/B"
k=0
for file in fileNames:
    if os.path.exists(wdir+r'/spectograms_rgb/'+file+".pt")==True:
        pass
    else:
        spec = C.get_spectrogram(file)
        j = 0
        for window_value in spec.keys():
            if window_value == 'annotations':
                break
            i = 0
            for channel in spec[window_value].values():
                if i == 0:
                    channels = channel.unsqueeze(0)

                else:
                    channels = torch.cat((channels,channel.unsqueeze(0)),axis=0)
                i+=1
            if j == 0:
                window_values=channels.resize(1,14,3,224,224)
            else:
                window_values=torch.cat((window_values,channels.resize(1,14,3,224,224)))
            j+=1

        filename=r'/spectograms_rgb/'+file+'.pt'
        torch.save(window_values,wdir+filename)
    k+=1
    print(k)