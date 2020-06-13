from CNN.loadTrainedCNN import trained_model_RGB
from Preprossering.PreprosseringPipeline import preprossingPipeline,getNewFeatureVec
import numpy as np
import torch
import os
import gc


def window_vector_loop(windowVec, featureVec):
    if windowVec is 0:
        windowVec = featureVec
        print(i)
    else:
        windowVec = torch.cat((windowVec, featureVec), 0)
        print(i)
    return windowVec.detach()

def feature_vector_loop_inner(tensor_window,model):
    for i in range(14):
        if i==0:
            x = model.features(tensor_window[i].unsqueeze(0).float())
            x = model.avgpool(x)
            x = torch.flatten(x, 1)
            tempFeatureVec = model.classifer[0:4](x)
            #tempFeatureVec=tensor_window[i].reshape(1,-1)
            featureVec = tempFeatureVec
        else:
            x = model.features(tensor_window[i].unsqueeze(0).float())
            x = model.avgpool(x)
            x = torch.flatten(x, 1)
            tempFeatureVec = model.classifer[0:4](x)
            #tempFeatureVec = tensor_window[i].reshape(1,-1)
            featureVec = torch.cat((featureVec, tempFeatureVec), 1)
    return featureVec.detach()


C=preprossingPipeline(BC_datapath=r"C:\Users\Andre\Desktop\Fagproject\Data\BC",mac=False)
fileNames=C.edfDict.keys()
wdir=r"C:\Users\Andre\Desktop\Fagproject\Feture_vectors_new"
path_new=r'D:\spectograms_rgb'
i=0
model = trained_model_RGB
model.eval()
for file in fileNames:
    if os.path.exists(wdir+r'/spectograms/'+file)==True:
        pass
    else:
        #try:
            windowVec = 0
            tensor, _, _, _ = C.make_label_cnn(make_from_filenames=[file], path='/Volumes/B/spectograms_rgb')
            tensor.requires_grad_(requires_grad=False)
            print(file)
            for i in range(int(len(tensor) / 14)):
                feature_vector = feature_vector_loop_inner(tensor[0 + i * 14: 14 + i * 14],model=model)
                windowVec = window_vector_loop(windowVec, feature_vector)
                windowVec=windowVec.detach()
                print(i)
                del feature_vector
                gc.collect()
            i += 1
            print(i)
            filename = r'/feature_vectors/' + file
            np.save(wdir + filename, windowVec)
        #except:
            #print(file)