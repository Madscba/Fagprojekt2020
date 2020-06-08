import io
from Preprossering.PreprosseringPipeline import preprossingPipeline
import matplotlib.pyplot as plt
from flashtorch.utils import apply_transforms, load_image
import numpy as np
import os
import torch




C=preprossingPipeline(BC_datapath=r"/Users/villadsstokbro/Dokumenter/DTU/KID/3. semester/Fagprojekt/BrainCapture/dataEEG",mac=True)
fileNames=C.edfDict.keys()
wdir="/Users/villadsstokbro/Dokumenter/DTU/KID/3. semester/Fagprojekt"

for file in fileNames:
    k=0
    if os.path.exists(wdir+r'/spectograms_rgb/'+file+".pt")==True:
        pass
    else:
        spec = C.get_spectrogram(file)
        j = 0
        for window_value in spec.keys():
            if window_value == 'annotations':
                break
            i = 0
            for channel in spec[window_value].keys():
                if i == 0:
                    a = np.array(spec[window_value][channel])
                    buf = io.BytesIO()
                    plt.imsave(buf, a, format='jpg')
                    buf.seek(0)
                    image = load_image(buf)
                    img = apply_transforms(image)
                    imgs=img
                    buf.close()
                else:
                    a = np.array(spec[window_value][channel])
                    buf = io.BytesIO()
                    plt.imsave(buf, a, format='jpg')
                    buf.seek(0)
                    image = load_image(buf)
                    img = apply_transforms(image)
                    imgs = torch.cat((imgs,img),axis=0)
                    buf.close()
                i+=1
            if j == 0:
                window_values=imgs.resize(1,14,3,224,224)
            else:
                window_values=torch.cat((window_values,imgs.resize(1,14,3,224,224)))

        filename=r'/spectograms_rgb/'+file+'.pt'
        torch.save(window_values,wdir+filename)
    k+=1
    if k==20:
        break
    print(k)