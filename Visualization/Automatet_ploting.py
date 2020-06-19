import numpy as np
import matplotlib.pyplot as plt
import plotly
from Preprossering.PreprosseringPipeline import preprossingPipeline
import pickle
from Villads.PCA_TSNE_classes import scale_data
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os, json
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from Visualization.plot_inteactive_functions import plot_pca_interactiv
from sklearn import preprocessing

class plot_auto():
    def __init__(self,feature_path,spectrograms_path,BC_datapath,Kfold_path,type,annotation="Is Eeg Usable For Clinical Purposes"):
        """"
        annotation="Quality Of Eeg" or "Is Eeg Usable For Clinical Purposes"
        """

        self.get_data = preprossingPipeline(BC_datapath=BC_datapath)


        self.LDA=None #LDA model
        self.PCA=None #PCA model
        self.TSNE=None #TSNE model
        self.max_windows=30
        self.max_files=10
        self.le = preprocessing.LabelEncoder()
        with open(os.path.join(os.getcwd(),Kfold_path) , "r") as read_file:
            Kfold = json.load(read_file)

        trainNames = Kfold[f"train_0"]
        testNames = Kfold[f"test_0"]

        self.type=type

        if type=="Feture_test":
            self.X_train,self.Y_train,self.filenames_train, self.train_id=self.get_vectors(trainNames,feature_path,annotation)
            self.X_test, self.Y_test, self.filenames_test, self.test_id = self.get_vectors(testNames, feature_path,annotation)
            self.le.fit(np.append(self.filenames_train,self.filenames_test))
            self.X_train = scale_data(self.X_train)
            self.X_test = scale_data(self.X_test)

        if type=="Spectrograms_test":
            self.X_train,self.Y_train,self.filenames_train,self.train_id=self.get_vectors(trainNames,spectrograms_path,annotation)
            self.X_test, self.Y_test, self.filenames_test,self.test_id = self.get_vectors(testNames, spectrograms_path,annotation)
            self.le.fit(np.append(self.filenames_train,self.filenames_test))
            self.X_train = scale_data(self.X_train)
            self.X_test = scale_data(self.X_test)

        if type=="Spectrograms_balance":
            self.X_train,self.Y_train,self.filenames_train,self.train_id=self.get_balance(5,spectrograms_path,annotation)
            self.le.fit(self.filenames_train)
        if type=="Feature_balance":
            self.X_train, self.Y_train, self.filenames_train, self.train_id = self.get_balance(5,feature_path,annotation)
            self.le.fit(self.filenames_train)
    def get_balance(self,n,path,annotation):
        windows1, labels1, filenames1, window_idx_full1 = self.get_data.make_label(quality=None,max_files=n,
                                                                        is_usable="Yes",max_windows=30,
                                                                        path=path)
        windows2, labels2, filenames2, window_idx_full2 = self.get_data.make_label(quality=None,max_files=n,
                                                                        is_usable="No",max_windows=30,
                                                                        path=path)
        idx=window_idx_full1+window_idx_full2
        filenames = filenames1 + filenames2
        feature_vectors_labels=labels1+labels2
        feature_vectors = np.vstack((windows1, windows2))

        return feature_vectors, feature_vectors_labels, filenames, idx

    def get_vectors(self, x, path,annotation):
        feature_vectors, feature_vectors_labels, filenames, idx = self.get_data.make_label(make_from_filenames=x, max_files=self.max_files,
                                                                                quality=None, is_usable=None,
                                                                                max_windows=self.max_windows,
                                                                                path=path)  # 18 files = 2144

        feature_vectors_labels = [self.get_data.edfDict[lable]["annotation"][annotation] for
                                  lable in feature_vectors_labels]
        return feature_vectors, feature_vectors_labels,filenames,idx



    def plot_all(self,idx,N_chanel=None,test=False):
        if test:
            self.window_id=self.test_id
        else:
            self.window_id = self.train_id

        spectrogram=self.get_data.plot_window(self.window_id[idx][0], self.window_id[idx][1], type="spec",plot=False)
        EEG=self.get_data.plot_window(self.window_id[idx][0], self.window_id[idx][1], type="EEG",plot=False)
        ch="TP10"
        fig = make_subplots(rows=1, cols=2)
        EEG.time=EEG.time / 1000
        if N_chanel==None:
            N_chanel=len(EEG.columns)-1
        fig = make_subplots(rows=N_chanel, cols=2)
        for i,ch in enumerate(EEG.columns[1:N_chanel+1]):

            fig1 = px.imshow(spectrogram[ch])

            fig2 = px.line(x=EEG.time, y=EEG[ch])
            trace1 = fig1['data'][0]
            trace2 = fig2['data'][0]

            fig.add_trace(
                trace1,
                row=i+1, col=1
            )

            fig.add_trace(
                go.Scatter(x=EEG.time, y=EEG[ch], mode="lines"),
                row=i+1, col=2,
            )
        fig.show()
    def plot_EEG(self,idx,N_chanel=None,test=False):
        if test:
            self.window_id=self.test_id
        else:
            self.window_id = self.train_id
        EEG = self.get_data.plot_window(self.window_id[idx][0], self.window_id[idx][1], type="EEG", plot=False)
        EEG.time = EEG.time / 1000
        if N_chanel==None:
            N_chanel=len(EEG.columns)-1
        fig = make_subplots(rows=int(np.ceil(N_chanel/3)), cols=3,subplot_titles=EEG.columns[1:N_chanel+1])
        for i,ch in enumerate(EEG.columns[1:N_chanel+1]):

            fig.add_trace(
                go.Scatter(x=EEG.time, y=EEG[ch], mode="lines"),
                row=int(i/3)+1, col=i%3+1)
        fig.update_layout(title=f"File: {self.window_id[idx][0]} window {self.window_id[idx][1]}")
        fig.show()

    def plot_Spec(self,idx,N_chanel=None,test=False):
        if test:
            self.window_id=self.test_id
        else:
            self.window_id = self.train_id
        spectrogram=self.get_data.plot_window(self.window_id[idx][0], self.window_id[idx][1], type="spec",plot=False)
        if N_chanel==None:
            N_chanel=len(list(spectrogram.keys()))
        fig = make_subplots(rows=int(np.ceil(N_chanel/4)), cols=4,subplot_titles=list(spectrogram.keys())[:N_chanel])
        for i,ch in enumerate(list(spectrogram.keys())[:N_chanel]):
            pix = px.imshow(spectrogram[ch])
            trace1 = pix['data'][0]
            fig.add_trace(
                trace1,
                row=int(i/4)+1, col=i%4+1)
        #fig.update_layout(title=f"File: {self.window_id[idx][0]} window {self.window_id[idx][1]}",showlegend: False)
        fig.show()

    def plot_space(self,space,test=False,point=None):
        if test:
            data=self.X_test
            l=self.Y_test
            i=self.test_id
            ln = [a[0] for a in self.test_id]
        else:
            data=self.X_train
            l=self.Y_train
            i=self.train_id
            ln=[a[0] for a in self.train_id]


        if space=="LDA":
            if self.LDA==None:
                self.LDA_trian()
            vectors=self.LDA.transform(data)

        if space=="PCA":
            if self.PCA==None:
                self.PCA_trian()
            vectors=self.PCA.transform(data)

        if space=="TSNE":
            if self.TSNE==None:
                self.TSNE_train()
            vectors=self.TSNE.fit_transform(data)

        if test:
            testssting="test set"
        else:
            testssting = "train set"

        plot_pca_interactiv(vectors, l, i, model=None,index=point)
        #plot_pca_interactiv(vectors, self.le.transform(ln).astype(str), i, model=None)

    def LDA_trian(self):
        print("Fitting LDA")
        LDA=LinearDiscriminantAnalysis()
        LDA.fit(self.X_train,self.Y_train)
        self.LDA=LDA

    def PCA_trian(self):
        print("Fitting PCA")
        PCA_model=PCA()
        PCA_model.fit(self.X_train)
        self.PCA=PCA_model

    def TSNE_train(self):
        print("Fitting TSNE")
        TSNE_model=TSNE()
        TSNE_model.fit(self.X_train)
        self.TSNE=TSNE_model


if __name__ == '__main__':
    feature_path=r"C:\Users\Andre\Desktop\Fagproject\Feture_vectors_new"

    BC_datapath=r"C:\Users\Andre\Desktop\Fagproject\Data\BC"
    K_path=r"Preprossering//K-stratified_is_useble_shuffle.json"
    spectrograms_path=r"C:\Users\Andre\Desktop\Fagproject\Spektrograms"

    ploterfature = plot_auto(feature_path=feature_path, spectrograms_path=spectrograms_path, BC_datapath=BC_datapath,
                         Kfold_path=K_path,type="Feature_balance")
    ploterfature.plot_space("PCA",test=False,point=211)
    #ploterfature.plot_space("PCA", test=True)
    #ploterfature.plot_EEG(224,test=False)
    #ploterfature.plot_Spec(224, test=False)
    #ploterfature.plot_all(72,test=False,N_chanel=2)
    #ploterfature.plot_space("LDA",test=False)
    #ploterfature.plot_space("LDA", test=True)
    ploterfature.plot_space("TSNE",test=False)
    #ploterfature.plot_space("TSNE", test=True)

    ploterspec = plot_auto(feature_path=feature_path, spectrograms_path=spectrograms_path, BC_datapath=BC_datapath,
                             Kfold_path=K_path, type="Spectrograms_balance")
    ploterspec.plot_space("PCA",test=False)
    #ploterspec.plot_space("LDA",test=False)
    #loterspec.plot_space("LDA", test=True)
    ploterspec.plot_space("TSNE",test=False)


    dummy=1

