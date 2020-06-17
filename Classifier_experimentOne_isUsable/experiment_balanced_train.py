import sys
sys.path.append('/zhome/87/9/127623/Fagprojekt/Fagprojekt2020')
from Classifier_experimentOne_isUsable.Validation import *

hpc = True
if hpc:
    BC = r"/work3/s173934/Fagprojekt/dataEEG"
    F = r'/work3/s173934/Fagprojekt/FeatureVectors'
    S = r'/work3/s173934/Fagprojekt/Spektrograms'
    SP = r'/work3/s173934/Fagprojekt/spectograms_rgb'
else:
    BC = r"C:\Users\Andre\Desktop\Fagproject\Data\BC"
    F = r"C:\Users\Andre\Desktop\Fagproject\Feture_vectors_new"
    S = r"C:\Users\Andre\Desktop\Fagproject\Spektrograms"
Kfold_path = r"Preprossering//K-stratified_is_useble_shuffle.json"

n=10

for i in range(n):
    Kfold_path = f"Preprossering//K-stratified_is_useble_shuffle{i}.json"
    CV=classifier_validation(Bc_path=BC, feture_path=F, speck_path=S,Kfold_path=Kfold_path, logfile_path="ClassifierTestLogs",max_windows=40,Balance_train=True)
    CV.test(classifyers=["SVM","LDA"],folds=None, type="fetures", logname=f"ex_balanced_train_fea{i}",confusion_matrix=True)
    CV.test(classifyers=["RF"],folds=None,type="spectrograms",logname=f"ex_balanced_train_spec{i}",confusion_matrix=True)