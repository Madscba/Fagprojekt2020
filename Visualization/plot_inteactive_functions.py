
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from Preprossering.PreprosseringPipeline import preprossingPipeline
import pickle
from Villads.PCA_TSNE_classes import scale_data
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.tools as tls
from plotly.subplots import make_subplots
"""
Functions for interactive plots made by Andreas 

"""
#data = np.load(r'C:\Users\Mads-_uop20qq\Documents\fagprojekt\wetransfer-2bf20e\PCA_TSNE\pca_features.npy')
def plot_pca_interactiv(pca_vectors,labels,window_id,pca1=0,pca2=1,model="PCA",plot=True):
    """

    :param pca_vectors:
    :param labels:
    :param window_id:
    :param pca1:
    :param pca2:
    :param model:
    :param plot: if true plot else returns trace
    :return:
    """
    #make data frame
    df=(pd.DataFrame(window_id,columns=["file","window"]))
    df[int(pca1)]=pca_vectors[:,int(pca1)]
    try:
        df[int(pca2)]=pca_vectors[:,int(pca2)]
    except:
        print("second component not found insert 0")
        df[int(pca2)]=0
    df["label"]=labels
    df["index"]=df.index
    fig=px.scatter(df,x=int(pca1),y=int(pca2),color="label",hover_data=["file","window","index"])
    fig.update_layout(
        title=model,
        xaxis_title=f"Component {pca1}",
        yaxis_title=f"Component {pca2}"
    )
    if plot:
        fig.show()
    else:
        return fig['data'][0]

def plot_comand(feature_path,path_pca,BC_datapath,newplot=True):
    pca = pickle.load(open(path_pca, 'rb'))
    get_data=preprossingPipeline(BC_datapath=BC_datapath)
    feature_vectors,labels,filenames,window_id= get_data.make_label( quality=None, is_usable=None, max_files=10,path = feature_path)
    while True:
        print("What you want to plot, options pca or")
        command=input()
        if command== "pca":
            feature_vectors,labels,filenames,window_id= get_data.make_label( quality=None, is_usable=None, max_files=10,path = feature_path)
            scaled_feature_vectors=scale_data(feature_vectors)
            pca_vectors=pca.transform(scaled_feature_vectors)
            print("what components ex 0 1")
            components=input()
            components=components.split(" ")
            plot_pca_interactiv(pca_vectors,labels,window_id,components[0],components[1])

        if command== "break":
            break

        if command== "spec" or command=="EEG":
            print("Insert index")
            idx=int(input())
            if newplot==False:
                get_data.plot_window(window_id[idx][0],window_id[idx][1],type=command)
            else:
                data,time,chanels=get_data.plot_window(window_id[idx][0],window_id[idx][1],type=command,plot=True)
                data=np.array(data).T
                df=pd.DataFrame(data,columns=chanels,index=time)
                fig = make_subplots(rows=len(chanels), cols=1)
                for i,ch in enumerate(chanels):
                    fig.add_trace(
                        px.line(df,y=ch,x=df.index),
                        row=(i+1),col=1)
                fig=px.line(df,y=chanels[0],x=df.index)
                fig.show()

if __name__ == "__main__":
    feature_path=r"C:\Users\Andre\Desktop\Fagproject\feature_vectors"
    path_pca=r'C:\Users\Andre\Desktop\Fagproject\PCA_TSNE\PCA.sav'

    plot_comand(feature_path,path_pca,newplot=False,BC_datapath=r"C:\Users\Andre\Desktop\Fagproject\Data\BC")