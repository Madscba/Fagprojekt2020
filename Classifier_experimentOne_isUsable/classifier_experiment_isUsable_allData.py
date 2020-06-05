##Mads Christian
import numpy as np
import pandas as pd
from sklearn import model_selection
from Preprossering.PreprosseringPipeline import preprossingPipeline
from Classifier_experimentOne_isUsable.trainTestValidateClassifiers import getClassifierAccuracies,tryNewDiv
import random
import os

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

random.seed(42)

class classifier_validation():
    def __init__(self,Bc_path,feture_path,speck_path,inner_splits=5,outer_splits=5,max_files=None):
        self.innerK=inner_splits
        self.outerK=outer_splits
        self.max_files=max_files
        text_file = open("filenames.txt", "r")
        x = text_file.read().splitlines() #Read 126 files, first 100 is usable, the next 26 is not usable
        text_file.close()

        text_file = open("isUsable.txt", "r")
        y = text_file.read().splitlines() #Read 126 files, first 100 is usable, the next 26 is not usable
        text_file.close()

        self.prepros = preprossingPipeline(mac=False, BC_datapath=Bc_path)
        self.feture_path=feture_path
        self.speck_path=speck_path

        self.outerloop(x,y)

    def get_spectrogram(self,x):
        spectrograms, spectrogram_labels, _, _ = self.prepros.make_label(make_from_filenames=x, quality=None,
                                                                        is_usable=None, max_files=self.max_files,
                                                                        path=self.speck_path)  # 18 files = 1926
        feature_vectors_labels=[self.prepros.edfDict[lable]["annotation"]["Is Eeg Usable For Clinical Purposes"] for lable in spectrogram_labels]

        return spectrograms,spectrogram_labels

    def get_feturevectors(self,x):
        feature_vectors, feature_vectors_labels, _, _ = self.prepros.make_label(make_from_filenames=x,
                                                                                 quality=None, is_usable=None,
                                                                                 max_files=self.max_files,
                                                                                 path=self.feture_path)  # 18 files = 2144
        feature_vectors_labels=[self.prepros.edfDict[lable]["annotation"]["Is Eeg Usable For Clinical Purposes"] for lable in feature_vectors_labels]
        return feature_vectors, feature_vectors_labels

    def outerloop(self,x,y):
        CV = model_selection.KFold(n_splits=self.outerK, shuffle=True)

        for (train_index, test_index) in CV.split(x,y):
            x_train = list(x[i] for i in train_index)
            x_test = list(x[i] for i in test_index)
            y_train = list(y[i] for i in train_index)
            y_test = list(y[i] for i in test_index)


            self.inner_loop(x_train,y_train)

    def inner_loop(self,x,y):
        """
        :param x:
        :param y:
        :return pandas df for feture vectors and spectrograms:
        """
        CV = model_selection.KFold(n_splits=self.outerK, shuffle=True)

        Spec_AC_matrix = []
        Feture_AC_matrix=[]
        for train_index, test_index in CV.split(x,y):
            x_train = list(x[i] for i in train_index)
            x_test = list(x[i] for i in test_index)

            #Test feturevectors
            Fx_train,Fy_train=self.get_feturevectors(x_train)
            Fx_test, Fy_test = self.get_feturevectors(x_test)

            Feture_AC_matrix.append(self.predict_all(Fx_train,Fy_train,Fx_test,Fy_test))

        Feture_AC_matrix = pd.DataFrame(Feture_AC_matrix,columns=["svm_predict", "LDA_predict", "DecisionTree_predict", "RF_predict"])
        Spec_AC_matrix = pd.DataFrame(Spec_AC_matrix,columns=["svm_predict", "LDA_predict", "DecisionTree_predict", "RF_predict"])
        return Feture_AC_matrix, Spec_AC_matrix

    def predict_all(self,x,y,x_test,y_test):
        """""
        test alle models
        """""
        svm_predict = np.array([])
        LDA_predict = np.array([])
        GNB_predict = np.array([])
        clf_predict = np.array([])
        DecisionTree_predict = np.array([])
        RF_predict = np.array([])

        y_true = np.array([])

        x_train = x

        y_train = y

        y_true = np.append(y_true, y_test)

        LDA = LinearDiscriminantAnalysis()
        LDA.fit(x, y)
        LDA_predict = np.append(LDA_predict, LDA.predict(x_test))
        print("Lda done", np.mean(y_true == LDA_predict))
        # support vector machine
        m_svm = SVC(gamma="auto", kernel="linear")
        m_svm.fit(x_train, y_train)
        svm_predict = np.append(svm_predict, m_svm.predict(x_test))
        print("SVM done", np.mean(y_true == svm_predict))
        m_DecisionTree = DecisionTreeClassifier()
        m_DecisionTree.fit(x_train, y_train)
        DecisionTree_predict = np.append(DecisionTree_predict, m_DecisionTree.predict(x_test))
        print("DT done", np.mean(y_true == DecisionTree_predict))
        # clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(20, 20, 20, 10), random_state=1)
        # clf.fit(x_train, y_train)
        # clf_predict = np.append(clf_predict, clf.predict(x_test))
        # print("neural done",np.mean(y_true == clf_predict))
        m_gaus = GaussianNB()
        m_gaus.fit(x_train, y_train)
        GNB_predict = np.append(GNB_predict, m_gaus.predict(x_test))
        print("gaus done", np.mean(y_true == GNB_predict))
        ranFor = RandomForestClassifier(n_estimators=100, criterion='gini')
        ranFor.fit(x_train, y_train)
        RF_predict = np.append(RF_predict, ranFor.predict(x_test))
        print("RF done", np.mean(y_true == RF_predict))

        print("svm_acc:", np.mean(y_true == svm_predict))
        print("LDA:", np.mean(y_true == LDA_predict))
        print("Dec:", np.mean(y_true == DecisionTree_predict))
        # print("MLP:", np.mean(y_true == clf_predict))
        print("GNB:", np.mean(y_true == GNB_predict))
        print("RandForest:", np.mean(y_true == RF_predict))
        return [np.mean(y_true == svm_predict), np.mean(y_true == LDA_predict),
                np.mean(y_true == DecisionTree_predict), np.mean(y_true == RF_predict)]
    # np.save(r'C:\Users\Mads-\OneDrive\Dokumenter\Universitet\4. Semester\02466 Fagprojekt - Bachelor i kunstig intelligens og data\Fagprojekt2020\Classifier_experimentOne_isUsable',results_spec,True)
    # np.save(r'C:\Users\Mads-\OneDrive\Dokumenter\Universitet\4. Semester\02466 Fagprojekt - Bachelor i kunstig intelligens og data\Fagprojekt2020\Classifier_experimentOne_isUsable', results_feature, True)

# spec_is_usable_train = np.array([])
# spec_not_usable_train = np.array([])
# spec_is_usable_test = np.array([])
# spec_not_usable_test = np.array([])
# path_spec = r'C:\Users\Mads-_uop20qq\Documents\fagprojekt\Spektrograms'
#
# used_usable_files = np.unique([x[0] for x in window_idx_full_is_usable[:]])
# used_not_usable_files = np.unique([x[0] for x in window_idx_full_not_usable[:]])
#
# for i in range(N):
#     if (i==0):
#         spec_is_usable_train = np.load(os.path.join(path_spec, used_usable_files[i] + '.npy')).squeeze(0)
#         spec_not_usable_train = np.load(os.path.join(path_spec, used_not_usable_files[i] + '.npy')).squeeze(0)
#     elif (i <18):
#         spec_is_usable_train = np.vstack( (spec_is_usable_train, np.load(os.path.join(path_spec, used_usable_files[i] + '.npy')).squeeze(0)) )
#         spec_not_usable_train = np.vstack( (spec_not_usable_train, np.load(os.path.join(path_spec, used_not_usable_files[i] + '.npy')).squeeze(0)) )
#     elif (i == 18):
#         spec_is_usable_test = np.load(os.path.join(path_spec, used_usable_files[i] + '.npy')).squeeze(0)
#         spec_not_usable_test = np.load(os.path.join(path_spec, used_not_usable_files[i] + '.npy')).squeeze(0)
#     else:
#         spec_is_usable_test = np.vstack((spec_is_usable_test, np.load(os.path.join(path_spec, used_usable_files[i] + '.npy')).squeeze(0)))
#         spec_not_usable_test = np.vstack((spec_not_usable_test, np.load(os.path.join(path_spec, used_not_usable_files[i] + '.npy')).squeeze(0)))
#
# spec_train = np.vstack((spec_is_usable_train,spec_not_usable_train))
# spec_test = np.vstack((spec_is_usable_test,spec_not_usable_test))




#getClassifierAccuracies(feature_vectors,labels,5,x_test,y_test)
#test = np.load(r'C:\Users\Mads-_uop20qq\Documents\fagprojekt\wetransfer-2bf20e\PCA_TSNE\pca_features.npy')
pass



##'sbs2data_2018_09_01_10_51_53_336.edf', 'sbs2data_2018_09_01_10_52_06_332.edf', 'sbs2data_2018_09_01_11_15_15_345DJP.edf', 'sbs2data_2018_09_01_11_33_40_98.edf', 'sbs2data_2018_09_01_11_38_26_333.edf', 'sbs2data_2018_09_01_11_56_10_342.edf', 'sbs2data_2018_09_01_12_24_19_105.edf', 'sbs2data_2018_09_01_12_30_52_341.edf', 'sbs2data_2018_09_01_12_43_34_344.edf



#####################################################
#############Get labels (Is usable)##################
#####################################################

##Setup models

##Divide data into train/test/val (set seed) (perhaps stratified)

##Get accuracies
if __name__ == '__main__':
    classifier_validation(Bc_path=r"C:\Users\Andre\Desktop\Fagproject\Data\BC",feture_path=r'C:\Users\Andre\Desktop\Fagproject\feature_vectors',speck_path=r'C:\Users\Andre\Desktop\Fagproject\Spektrograms',max_files=4)