import numpy as np
import pandas as pd
from sklearn import model_selection
import sys
sys.path.append('/zhome/87/9/127623/Fagprojekt/Fagprojekt2020')
from Preprossering.PreprosseringPipeline import preprossingPipeline
from Classifier_experimentOne_isUsable.trainTestValidateClassifiers import getClassifierAccuracies,tryNewDiv
import random
import os
import json

from datetime import datetime
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
    def __init__(self, Bc_path, feture_path, speck_path, Kfold_path, max_windows_test=None,max_windows_train=None , logfile_path=None,Balance_test=False,Balance_train=False):
        """

        :param Bc_path:
        :param feture_path: feture vektors dataset
        :param speck_path: spectrograms dataset
        :param Kfold_path: path to K-fold.json fille
        :param max_windows:  limet numbers of files for debugging purpes. WARNING the program would still show to numbers of files in your fold
        :param logfile_path: path to folder where log files is to be saved
        """
        self.max_windows_test=max_windows_test
        self.max_windows_train=max_windows_train
        self.logfile_path=logfile_path

        self.prepros = preprossingPipeline(mac=False, BC_datapath=Bc_path)
        self.feture_path=feture_path
        self.speck_path=speck_path
        self.Balance_test=Balance_test
        self.Balance_train = Balance_train

        with open(os.path.join(os.getcwd(),Kfold_path) , "r") as read_file:
            self.Kfold = json.load(read_file)




    def get_spectrogram(self,x,max_windows):
        #OLD
        spectrograms, spectrogram_labels, _, idx = self.prepros.make_label(make_from_filenames=x, quality=None,max_files=None,
                                                                        is_usable=None,max_windows=max_windows,
                                                                        path=self.speck_path)  # 18 files = 1926
        spectrogram_labels=[self.prepros.edfDict[lable]["annotation"]["Is Eeg Usable For Clinical Purposes"] for lable in spectrogram_labels]

        return spectrograms,spectrogram_labels,idx

    def get_balanced(self,x,path_s,max_windows):
        #OLD
        is_useble=[]
        Not_useble=[]
        for name in x:
            lable=self.prepros.edfDict[name]["annotation"]["Is Eeg Usable For Clinical Purposes"]
            if lable=="Yes":
                is_useble.append(name)
            else:
                Not_useble.append(name)

        windows1, labels1, filenames1, window_idx_full1 = self.prepros.make_label(make_from_filenames=is_useble, quality=None,max_files=None,
                                                                        is_usable=None,max_windows=int(max_windows*(5/8)),
                                                                        path=path_s)
        windows2, labels2, filenames2, window_idx_full2 = self.prepros.make_label(make_from_filenames=Not_useble, quality=None,max_files=None,
                                                                        is_usable=None,max_windows=max_windows*2.5,
                                                                        path=path_s)
        labels=labels1+labels2
        idx=window_idx_full1+window_idx_full2
        windows=np.vstack([windows1,windows2])
        labels = [self.prepros.edfDict[lable]["annotation"]["Is Eeg Usable For Clinical Purposes"] for lable
                              in labels]
        return windows,labels, idx

    def get_feturevectors(self,x,max_windows):
        #OLD
        feature_vectors, feature_vectors_labels, _, idx = self.prepros.make_label(make_from_filenames=x,max_files=None,
                                                                                 quality=None, is_usable=None,
                                                                                 max_windows=max_windows,
                                                                                 path=self.feture_path)  # 18 files = 2144
        feature_vectors_labels=[self.prepros.edfDict[lable]["annotation"]["Is Eeg Usable For Clinical Purposes"] for lable in feature_vectors_labels]
        return feature_vectors, feature_vectors_labels,idx


    def test(self,type,folds=None,classifyers=["GNB","SVM", "LDA", "DecisionTree", "RF","Random"],logname="test.json",confusion_matrix=False):
        """

        :param type: fetures or spectrograms
        :param folds: if None use fold outer fold of fold whos path was given, if dict  opbjet use that as.
        if list use that subfold eg: [1,2] for subfold 2 in subfold 1.
        :param classifyers: list of clasifyers to test
        :param logname: Name of json file if none don't make a log
        :return: AC matrix
        """

        if isinstance(folds, dict):
            folddic=folds
        else:
            folddic=self.Kfold

            if folds!=None:
                for i in folds:
                    try:
                        folddic=folddic[f"subfolds_{i}"]
                    except:
                        print(f"Invalid fold index {i} max fold is {folddic['N_folds']}")

        Nfold=folddic['N_folds']
        AC_collumns=np.append(classifyers,["N_TestFiles","N_TestWindows","N_TrainFiles","N_TrainWindows"])
        AC_matrix = pd.DataFrame(index=np.arange(0, Nfold), columns=AC_collumns)
        resultDict={}
        debuglog={}
        for n in range(Nfold):

            trainNames=folddic[f"train_{n}"]
            testNames=folddic[f"test_{n}"]
            resultDict[f"fold_{n}"]={}
            now=datetime.now()
            debuglog[f"fold_{n}"]={"tstat": now.strftime("%m/%d/%Y, %H:%M:%S")}
            if type=="fetures":
                #Test feturevectors
                if self.Balance_train:
                    x_train,y_train, idx_train=self.get_balanced(trainNames,path_s=self.feture_path,max_windows=self.max_windows_train)
                else:
                    x_train, y_train, idx_train = self.get_feturevectors(trainNames,max_windows=self.max_windows_train)

                if self.Balance_test:
                    x_test, y_test, idx_test = self.get_balanced(testNames, path_s=self.feture_path,max_windows=self.max_windows_test)
                else:
                    x_test, y_test, idx_test = self.get_feturevectors(testNames,max_windows=self.max_windows_test)


            elif type=="spectrograms":
                #Test spectrograms
                if self.Balance_train:
                    x_train,y_train, idx_train=self.get_balanced(trainNames,path_s=self.speck_path,max_windows=self.max_windows_train)
                else:
                    x_train, y_train, idx_train = self.get_spectrogram(trainNames,max_windows=self.max_windows_train)

                if self.Balance_test:
                    x_test, y_test, idx_test = self.get_balanced(testNames, path_s=self.speck_path,max_windows=self.max_windows_test)
                else:
                    x_test, y_test, idx_test = self.get_spectrogram(testNames,max_windows=self.max_windows_test)

            else:
                raise Exception("wrong type try fetures or spectrograms")

            for C in classifyers:
                if C=="SVM":
                    predict,prob=self.predict_svm(x_train,y_train,x_test,y_test)
                if C=="LDA":
                    predict,prob=self.predict_LDA(x_train,y_train,x_test,y_test)

                if C=="GNB":
                    predict,prob=self.predict_GNB(x_train,y_train,x_test,y_test)

                if C=="DecisionTree":
                    predict,prob=self.predict_DissionTree(x_train,y_train,x_test,y_test)

                if C=="RF":
                    predict,prob= self.predict_RF(x_train, y_train, x_test, y_test)

                if C=="Random":
                    predict,prob=self.predict_random(x_train, y_train, x_test, y_test)

                AC_matrix.loc[n,C]=np.mean(y_test == predict)
                for lable1 in set(y_test):
                    base1=np.full(len(y_test),lable1)
                    AC_matrix.loc[n,f"Baseline {lable1}"]=np.sum([y_test==base1])/len(y_test)
                    if confusion_matrix==True:
                        for lable2 in set(y_test):
                            base2=np.full(len(y_test),lable2)
                            AC_matrix.loc[n,f"Predicted {lable1} True {lable2}"]=np.sum(np.logical_and(predict==base1,y_test==base2))

                resultDict[f"fold_{n}"][f"{C}_predict"]=list(predict)
                resultDict[f"fold_{n}"][f"{C}_prob"] = prob.tolist()
            debuglog[f"fold_{n}"]["train_idx"]=list(idx_train)
            debuglog[f"fold_{n}"]["test_idx"] = list(idx_test)
            debuglog[f"fold_{n}"]["trian_lengt"]=len(x_train)
            debuglog[f"fold_{n}"]["test_lengt"]=len(x_test)
            #Ad baseline to debyg log:
            for lable in set(y_test):
                base_test=np.full(len(y_test), lable)
                base_train=np.full(len(y_train), lable)
                debuglog[f"fold_{n}"][f"train {lable}"]=str(np.sum([y_train==base_train]))
                debuglog[f"fold_{n}"][f"test {lable}"] = str(np.sum([y_test == base_test]))

            AC_matrix.loc[n,"N_TestFiles"]=len(testNames)
            AC_matrix.loc[n, "N_TestWindows"] = len(x_test)
            AC_matrix.loc[n, "N_TrainFiles"] = len(trainNames)
            resultDict[f"fold_{n}"]["index"] = list(idx_test)
            resultDict[f"fold_{n}"]["True"]=list(y_test)
            # clear data
            del x_test
            del y_test
            del x_train
            del y_train


            print(f"fold {n} completet")


            #print(AC_matrix)
        if logname is not None:
            AC_matrix.to_csv(os.path.join(os.getcwd(),self.logfile_path,f"{logname}_AC"))
            with open(os.path.join(os.getcwd(),self.logfile_path,f"{logname}_predict.json"), 'w') as fp:
                json.dump(resultDict, fp, indent=3)
            with open(os.path.join(os.getcwd(),self.logfile_path,f"{logname}_debuglog.json"), 'w') as fp:
                json.dump(debuglog, fp, indent=3)

        return AC_matrix

    def two_layes(self,type,classifyers=["SVM", "LDA", "DecisionTree","GNB", "RF"],EXP_name=None):
        """
        Implement two layers cross validation as spesified in 02450book algoritme 6 page 175
        :param type: spetrograms of fetures
        :param classifyers: list of classifyers to test defult ["SVM", "LDA", "DecisionTree", "RF"]
        :param EXP_name: name of experiment, all experiment files will be saved with that name
        :return: if EXP_name== None return experiment log and log of genneralisation erro. both pd.DF opjects
        """

        folddic = self.Kfold
        Nfold=folddic['N_folds']
        Best_log=pd.DataFrame(index=np.arange(0, Nfold), columns=["Best","Best_AC","N_TestFiles","N_TestWindows","N_TrainFiles","N_TrainWindows"])
        Gen_error=pd.DataFrame(index=np.arange(0, Nfold), columns=classifyers)
        for n in range(Nfold):

            if EXP_name !=None:
                logname=f"{EXP_name}_Inderfold_{n}.json"
            else:
                logname=None
            print(logname)
            Innerresult=self.test(folds=[n],classifyers=classifyers, type=type, logname=logname)
            totalDatapoint=Innerresult.N_TestWindows.sum()

            Egen_s=[np.sum([(Innerresult.N_TestWindows.iloc[j]/totalDatapoint)*Innerresult.loc[j,s]
                            for j in range(len(Innerresult.index))]) for s in classifyers]
            best = classifyers[np.argmax(Egen_s)]
            print(f"Best classifyer {best}")
            testdict={"train_0": folddic[f"train_{n}"], "test_0": folddic[f"test_{n}"],"N_folds": 1}
            Outerresult=self.test(folds=testdict,classifyers=[best], type=type, logname=None,confusion_matrix=True)
            Gen_error.iloc[n] = Egen_s

            Best_log.loc[n,"Best"]=best
            Best_log.loc[n,"Best_AC"]=Outerresult.iloc[0,0]  #Insert best value
            for col in Outerresult.columns[1:]:
                Best_log.loc[n, col] = Outerresult.loc[0, col] #Update meta data

        totalDatapoint = Best_log.N_TestWindows.sum()
        Egen_s = np.sum([(Best_log.N_TestWindows.iloc[j] / totalDatapoint) * Best_log.loc[j,"Best_AC"]
                          for j in range(len(Best_log.index))])
        Best_log.loc["Final", "Best"] = "Combined"
        Best_log.loc["Final","Best_AC"]=Egen_s
        Best_log.loc["Final", "N_TestWindows"] = totalDatapoint
        Best_log.loc["Final", "N_TestFiles"]= Best_log.N_TestFiles.sum()
        if EXP_name != None:

            Gen_error.to_csv(os.path.join(os.getcwd(),self.logfile_path,f"{EXP_name}_Genneralisation_errors.json"))
            Best_log.to_csv(os.path.join(os.getcwd(),self.logfile_path,f"{EXP_name}_Outerloop.json"))


        return Best_log,Gen_error






        #Feture_AC_matrix = pd.DataFrame(Feture_AC_matrix,columns=["svm_predict", "LDA_predict", "DecisionTree_predict", "RF_predict"])
        #Spec_AC_matrix = pd.DataFrame(Spec_AC_matrix,columns=["svm_predict", "LDA_predict", "DecisionTree_predict", "RF_predict"])
        #return Feture_AC_matrix, Spec_AC_matrix

    def predict_LDA(self,x,y,x_test,y_test):
        LDA_predict = np.array([])
        LDA = LinearDiscriminantAnalysis()
        LDA.fit(x, y)
        LDA_predict = np.append(LDA_predict, LDA.predict(x_test))
        p=LDA.predict_proba(x_test)
        print("Lda done", np.mean(y_test == LDA_predict))
        return LDA_predict,p

    def predict_svm(self,x,y,x_test,y_test):
        svm_predict = np.array([])
        # support vector machine
        m_svm = SVC(gamma="auto", kernel="linear",probability=True)
        m_svm.fit(x, y)
        svm_predict = np.append(svm_predict, m_svm.predict(x_test))
        p=m_svm.predict_proba(x_test)
        print("SVM done", np.mean(y_test == svm_predict))

        return svm_predict,p

    def predict_GNB(self,x_train,y_train,x_test,y_test):
        GNB_predict = np.array([])
        m_gaus = GaussianNB()
        m_gaus.fit(x_train, y_train)
        GNB_predict = np.append(GNB_predict, m_gaus.predict(x_test))
        p=m_gaus.predict_proba(x_test)
        print("gaus done", np.mean(y_test == GNB_predict))
        return  GNB_predict,p


    def predict_DissionTree(self,x_train,y_train,x_test,y_test):
        DecisionTree_predict =np.array([])
        m_DecisionTree = DecisionTreeClassifier()
        m_DecisionTree.fit(x_train, y_train)
        DecisionTree_predict = np.append(DecisionTree_predict, m_DecisionTree.predict(x_test))
        p=m_DecisionTree.predict_proba(x_test)
        print("DT done", np.mean(y_test == DecisionTree_predict))
        return DecisionTree_predict,p

    def predict_RF(self,x_train,y_train,x_test,y_test):
        p = np.array([])
        RF_predict = np.array([])

        ranFor = RandomForestClassifier(n_estimators=100, criterion='gini',random_state=42)
        ranFor.fit(x_train, y_train)
        RF_predict = np.append(RF_predict, ranFor.predict(x_test))
        p=ranFor.predict_proba(x_test)
        print("RF done", np.mean(y_test == RF_predict))
        return RF_predict,p

    def predict_random(self,x_train,y_train,x_test,y_test):
        classes=["Yes","No"]
        prob=[0.8,0.2]
        predict=np.random.choice(classes, size=len(x_test), p=prob)
        print("Random predict", np.mean(y_test == predict))
        return predict,np.array([0,0,0])



    def predict_clf(self,x,y,x_test,y_test):
        pass
        #Denne function var ikke skævet færdigt.
    # clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(20, 20, 20, 10), random_state=1)
    # clf.fit(x_train, y_train)
    # clf_predict = np.append(clf_predict, clf.predict(x_test))
    # print("neural done",np.mean(y_true == clf_predict))

def dict_to_pd(predict_dict):
    print(predict_dict)

if __name__ == '__main__':
    hpc=False
    if hpc:
        BC=r"/work3/s173934/Fagprojekt/dataEEG"
        F=r'/work3/s173934/Fagprojekt/FeatureVectors'
        S=r'/work3/s173934/Fagprojekt/Spektrograms'
        SP=r'/work3/s173934/Fagprojekt/spectograms_rgb'
    else:
        BC=r"C:\Users\Andre\Desktop\Fagproject\Data\BC"
        F=r"C:\Users\Andre\Desktop\Fagproject\Feture_vectors_new"
        S=r"C:\Users\Andre\Desktop\Fagproject\Spektrograms"
        BC = r'C:\Users\Mads-\OneDrive\Dokumenter\Universitet\4. Semester\02466 Fagprojekt - Bachelor i kunstig intelligens og data\dataEEG'

    Kfold_path=r"Preprossering//K-stratified_is_useble_shuffle.json"
    CV=classifier_validation(Bc_path=BC, feture_path=F, speck_path=S,Kfold_path=Kfold_path, logfile_path="ClassifierTestLogs",max_windows_test=10,max_windows_train=15)
   # CV.test(classifyers=["Random"],folds=None, type="fetures", logname=None,confusion_matrix=False)
    CV.test(classifyers=["Random"],folds=None,type="spectrograms",logname=None,confusion_matrix=False)
    CV.test(classifyers=["Random"], folds=None, type="spectrograms", logname=None, confusion_matrix=False)
    CV.test(classifyers=["Random"], folds=None, type="spectrograms", logname=None, confusion_matrix=False)
    # CV.two_layes(type="spectrograms", EXP_name="Spec_twofoldsrat_fulldataset")
    #CV.two_layes(type="spectrograms",EXP_name="test")

