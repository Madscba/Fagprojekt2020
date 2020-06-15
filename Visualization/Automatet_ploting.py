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
        with open(os.path.join(os.getcwd(),Kfold_path) , "r") as read_file:
            Kfold = json.load(read_file)

        trainNames = Kfold[f"train_0"]
        testNames = Kfold[f"test_0"]

        self.type=type
        if type=="Feture":
            self.X_train,self.Y_train,self.filenames_train, self.train_id=self.get_vectors(trainNames,feature_path,annotation)
            self.X_test, self.Y_test, self.filenames_test, self.test_id = self.get_vectors(testNames, feature_path,annotation)

        if type=="Spectrograms":
            self.X_train,self.Y_train,self.filenames_train,self.train_id=self.get_vectors(trainNames,spectrograms_path,annotation)
            self.X_test, self.Y_test, self.filenames_test,self.test_id = self.get_vectors(testNames, spectrograms_path,annotation)


    def get_vectors(self, x, path,annotation):
        feature_vectors, feature_vectors_labels, filenames, idx = self.get_data.make_label(make_from_filenames=x, max_files=self.max_files,
                                                                                quality=None, is_usable=None,
                                                                                max_windows=self.max_windows,
                                                                                path=path)  # 18 files = 2144

        feature_vectors_labels = [self.get_data.edfDict[lable]["annotation"][annotation] for
                                  lable in feature_vectors_labels]
        return feature_vectors, feature_vectors_labels,filenames,idx



    def plot_all(self,idx):
        spectrogram=self.get_data.plot_window(self.window_id[idx][0], self.window_id[idx][1], type="spec",plot=False)
        EEG=self.get_data.plot_window(self.window_id[idx][0], self.window_id[idx][1], type="EEG",plot=False)
        ch="TP10"
        fig = make_subplots(rows=1, cols=2)
        EEG.time=EEG.time / 1000
        fig1 = px.imshow(spectrogram[ch])
        fig2 = px.line(x=EEG.time, y=EEG[ch])
        trace1 = fig1['data'][0]
        trace2 = fig2['data'][0]

        fig.add_trace(
            trace1,
            row=1, col=1
        )

        fig.add_trace(
            trace2,
            row=1, col=2
        )
        fig.show()

    def plot_space(self,space,test=False):
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

        plot_pca_interactiv(vectors, l, i, model=f"{space} {testssting} {self.type}")
        plot_pca_interactiv(vectors, ln, i, model=f"{space} {testssting} {self.type}")

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
    ploterfature=plot_auto(feature_path=feature_path,spectrograms_path=spectrograms_path,BC_datapath=BC_datapath,Kfold_path=K_path,type="Feture")
    #ploter.plot_all(3)

    ploterfature.plot_space("PCA",test=False)
    #ploterfature.plot_space("PCA", test=True)
    ploterfature.plot_space("LDA",test=False)
    ploterfature.plot_space("LDA", test=True)
    ploterfature.plot_space("TSNE",test=False)
    #ploterfature.plot_space("TSNE", test=True)

    ploterspec = plot_auto(feature_path=feature_path, spectrograms_path=spectrograms_path, BC_datapath=BC_datapath,
                             Kfold_path=K_path, type="Spectrograms")
    ploterspec.plot_space("PCA",test=False)
    #ploterfature.plot_space("PCA", test=True)
    ploterspec.plot_space("LDA",test=False)
    ploterspec.plot_space("LDA", test=True)
    ploterspec.plot_space("TSNE",test=False)


    dummy=1

