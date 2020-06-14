import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
"""
File for visualisation of the outer loop of experiment. aka plotting confusione matrix
Responsble Andreas
"""

path=r"ClassifierTestLogs/Spec_twofoldsrat_fulldataset_Outerloop.json"

lables=["Yes","No"]

data=pd.read_csv(os.path.join(os.getcwd(),path))

title=f"Spectrograms representation: Genneralisation AC={data.iloc[-1,2]}"

fig, axis=plt.subplots( ncols=max(data.index),figsize=(25,5))
fig.subplots_adjust(top=0.85)
for row in range(0,max(data.index)):

    #predict=[f"Predicted {l}" for l in lables]
    #true=[f"True {l}" for l in lables]
    MX=pd.DataFrame(dtype="int_")

    for l in lables:
        for j in lables:
            MX.loc[f"True {j}",f"Predicted {l}"]=np.int(data.loc[row,f"Predicted {l} True {j}"])

    MX.loc[:,"N windows"]=MX.sum(axis=1)
    MX.loc["N windows", :] = MX.sum(axis=0)
    sns.heatmap(MX, annot=True,fmt=".0f",cbar=False,ax=axis[row])
    axis[row].set_title(f"Fold {row+1}: best classifir {data.loc[row,'Best']} with {np.round(data.loc[row,'Best_AC'],3)} Accuracy")
print(title)
#fig.constrained_layout()

fig.show()
    #print(MX)





