#Til Andreas:

import os
import json
import pandas as pd
datadir=r"C:\Users\Andre\Desktop\Fagproject\Data\BC\data_farrahtue_EEG"
file_name_xlsx=r"MGH_File_Annotations.xlsx"

with open('Preprossering/edfFiles.json') as json3_file:
    edfDict = json.load(json3_file)

annotation = pd.read_excel(os.path.join(datadir,file_name_xlsx), sheet_name=[0])
annotation=annotation[0]
annotation.index=annotation.Path
MetaData=pd.DataFrame()
for idx, path in enumerate(annotation['Path']):
    name = path.split("/")[-1]
    if name in edfDict:  # Is file in recordins
        MetaData.loc[:,idx]=annotation.loc[path]
        MetaData.loc["Useble", idx]=edfDict[name]["annotation"]["Is Eeg Usable For Clinical Purposes"]
MetaData=MetaData.transpose()
MetaData['Length [seconds]']=MetaData['Length [seconds]']/60
print(f"Age mean={MetaData['Age'].mean()} std={MetaData['Age'].std()} max={MetaData['Age'].max()} min={MetaData['Age'].min()} ")
print(MetaData['Age'].quantile([0,0.25,0.5,0.75,1],interpolation="midpoint"))
print(f"Below the age of 2={MetaData[MetaData['Age']<2]['Age'].count()}")
print(f"Lengt in S mean={MetaData['Length [seconds]'].mean()} std={MetaData['Length [seconds]'].std()} max={MetaData['Length [seconds]'].max()} min={MetaData['Length [seconds]'].min()} ")
print(MetaData['Length [seconds]'].quantile([0,0.25,0.5,0.75,1],interpolation="midpoint"))
print(f"Male ={MetaData[MetaData['Sex']=='Male']['Sex'].count()} female={MetaData[MetaData['Sex']=='Female']['Sex'].count()}")
print(f"total= {MetaData['Age'].count()}")

yesData=MetaData[MetaData['Useble']=="Yes"]
print("Yes")
print(f"Age mean={yesData['Age'].mean()} std={yesData['Age'].std()} max={yesData['Age'].max()} min={yesData['Age'].min()} ")
print(yesData['Age'].quantile([0,0.25,0.5,0.75,1],interpolation="midpoint"))
print(f"Below the age of 2={yesData[yesData['Age']<2]['Age'].count()}")
print(f"Lengt in S mean={yesData['Length [seconds]'].mean()} std={yesData['Length [seconds]'].std()} max={yesData['Length [seconds]'].max()} min={yesData['Length [seconds]'].min()} ")
print(yesData['Length [seconds]'].quantile([0,0.25,0.5,0.75,1],interpolation="midpoint"))
print(f"Male ={yesData[yesData['Sex']=='Male']['Sex'].count()} female={yesData[yesData['Sex']=='Female']['Sex'].count()}")
print(f"total= {yesData['Age'].count()}")

noData=MetaData[MetaData['Useble']=="No"]
print("No")
print(f"Age mean={noData['Age'].mean()} std={noData['Age'].std()} max={noData['Age'].max()} min={noData['Age'].min()} ")
print(noData['Age'].quantile([0,0.25,0.5,0.75,1],interpolation='midpoint'))
print(f"Below the age of 2={noData[noData['Age']<2]['Age'].count()}")
print(f"Lengt in S mean={noData['Length [seconds]'].mean()} std={noData['Length [seconds]'].std()} max={noData['Length [seconds]'].max()} min={yesData['Length [seconds]'].min()} ")
print(noData['Length [seconds]'].quantile([0,0.25,0.5,0.75,1],interpolation='midpoint'))
print(f"Male ={noData[noData['Sex']=='Male']['Sex'].count()} female={noData[noData['Sex']=='Female']['Sex'].count()}")
print(f"total= {noData['Age'].count()}")

# with open("filenames.txt", "r") as fh:
#     filenames = fh.read().splitlines()
# length_usable,count_usable = [], []
# length_non_usable, count_non_usable = [], []
# for idx,i in enumerate(filenames):
#     if i in loader.no_file:
#         if loader.no_file[i]['annotation']['Is Eeg Usable For Clinical Purposes'] == "Yes":
#             length_usable.append(loader.no_file[i]['annotation']['Recording Length [seconds]'])
#             count_usable += 1
#         else:
#             length_non_usable.append(loader.no_file[i]['annotation']['Recording Length [seconds]'])
#             count_non_usable += 1


# with open('Preprossering/edfFiles.json') as json3_file:
#     usefulData = json.load(json3_file)
#
# with open(r'C:\Users\Mads-\Downloads/compromised_files.json') as json4_file:
#     compromisedData = json.load(json4_file)
# pass
# reasons = []
# files = []
# for i in compromisedData:
#     reasons.append(compromisedData[i]['reson'])

# pass