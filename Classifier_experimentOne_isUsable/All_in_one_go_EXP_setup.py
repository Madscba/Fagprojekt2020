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
    F = r"C:\Users\Andre\Desktop\Fagproject\Feature_vector4"
    S = r"C:\Users\Andre\Desktop\Fagproject\Spektrograms"
Kfold_path = r"Preprossering//K-stratified_is_useble_shuffle.json"

max_test=40 #20
max_train=20 #20

# unbalanced_setup=classifier_validation(Bc_path=BC, feture_path=F, speck_path=S,Kfold_path=Kfold_path, logfile_path=r"C:\Users\Andre\OneDrive - Danmarks Tekniske Universitet\Experiment_result",max_windows_test=max_test,max_windows_train=max_train)
# unbalanced_setup.test(classifyers=["RF","SVM","LDA","GNB"],folds=None, type="fetures", logname=f"Feature_unbal_final1",confusion_matrix=True)
# unbalanced_setup.test(classifyers=["RF","SVM","LDA","GNB"],folds=None,type="spectrograms",logname=f"Spectrograms_unbal_final1",confusion_matrix=True)

balanced_setup=classifier_validation(Bc_path=BC, feture_path=F, speck_path=S,Kfold_path=Kfold_path, logfile_path=r"ClassifierTestLogs",max_windows_test=max_test,max_windows_train=max_train,Balance_train=True)
balanced_setup.test(classifyers=["RF","SVM","LDA","GNB"],folds=None,type="spectrograms",logname=f"Spectrograms_bal_final1",confusion_matrix=True)
balanced_setup.test(classifyers=["RF","SVM","LDA","GNB"],folds=None, type="fetures", logname=f"Feture_bal_final1",confusion_matrix=True)
