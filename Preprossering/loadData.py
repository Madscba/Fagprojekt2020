import os, re, glob, json, sys
import pandas as pd
import numpy as np
from collections import defaultdict


# load from json to dict
def jsonLoad(path = False):
    if path is False:
        sys.exit("no path were given to load Json")
    else:
        with open(path, "r") as read_file:
            edfDefDict = json.load(read_file)
    print("\npaths found for loading")
    return edfDefDict

class json_maker:
    """
    farrahData: farrahData full path
    """
    def __init__(self):
        pass

        
    def jsonSave(self,jsonName,saveDir=""):
        """
        Args: 
        jsonName name of jeson file.json:
        saveDir subdiratory to save file working dir is automatikly the root:
        """
        edfNonComp = {ID: v for (ID, v) in self.edfDefDict.items() if self.edfDefDict[ID]["deathFlag"] is False}
        path=os.path.join(os.getcwd(),saveDir,jsonName)
        with open(path, 'w') as fp:
            json.dump(edfNonComp, fp, indent=4)

    # .edf paths in a non-comp and comp json as logging file
    def debugLog(self,jsonNames=["compromised_files.json","files_anotatet_but_not_found.json"],saveDir=""):
        """
        jsonNames list of 2 ellements: namea of jesons files  file.json:
        saveDir subdiratory to save file working dir is automatikly the root:
        """
        edfComp = {ID: v for (ID, v) in self.edfDefDict.items() if self.edfDefDict[ID]["deathFlag"] is True}

        path=os.path.join(os.getcwd(),saveDir,jsonNames[0])
        with open(path, 'w') as fp:
            json.dump(edfComp, fp, indent=4)
        with open(path, 'w') as fp:
            json.dump(edfComp, fp, indent=4)
        path=os.path.join(os.getcwd(),saveDir,jsonNames[1])
        with open(path, 'w') as fp:
            json.dump(self.no_file, fp, indent=4)

    def find_edf(self,farrahDataDir):
        # find all .edf files
        pathRootInt = len(farrahDataDir.split('\\'))-3
        farrahPaths = ['\\'.join(fDir.split('\\')[pathRootInt:]) for fDir in glob.glob(farrahDataDir + "**/*.edf", recursive=True)]
        # construct defaultDict for data setting
        self.edfDefDict = defaultdict(dict)
        for path in farrahPaths:
            file = path.split('\\')[-1]
            if file in self.edfDefDict.keys():
                self.edfDefDict[file]["path"].append(path)
                self.edfDefDict[file]["deathFlag"] = True
                self.edfDefDict[file]["reson"] = "already existing"
            else:
                self.edfDefDict[file]["path"] = []
                self.edfDefDict[file]["deathFlag"] = False
                self.edfDefDict[file]["path"].append(path)
            self.edfDefDict[file]["Files named %s" % file] = len(self.edfDefDict[file]["path"])


    def add_annotations(self,path_xlsx_file,file_name_xlsx='MGH_File_Annotations.xlsx',sheet_names=[2,3,4],atributes=["Quality Of Eeg","Is Eeg Usable For Clinical Purposes","Reader","Recording Length [seconds]"],sort=False):
        """
        Args: path_xlsx_file:
        file_name_xlsx:
        sheet_names: list of the sheets to loop over
        sort bool: if true files with missing attributes will be deathFlagged deflualt=false 
        atributes: atribute that you wish to use
        #Note if time atributes is added expect bug in make json
        """
        for sheet in sheet_names:
            annotation = pd.read_excel(os.path.join(path_xlsx_file,file_name_xlsx), sheet_name=sheet)
            self.no_file=defaultdict(dict)
            for idx,path in enumerate(annotation['Recording']):
                name=path.split("/")[-1]
                if name in self.edfDefDict: #Is file in recordins 
                    self.edfDefDict[name]["annotation"]=annotation.iloc[idx].loc[atributes].to_dict()
                    if sort:
                        for an in atributes: #deathFlag missing values 
                            if isinstance(self.edfDefDict[name]["annotation"][an],str)==False:
                                if np.isnan(self.edfDefDict[name]["annotation"][an]):
                                    self.edfDefDict[name]["deathFlag"]=True
                                    self.edfDefDict[name]["reson"]="atribute one or more atributes is missing"
                else:
                    #If we have anotations without files
                    self.no_file[name]={"annotation": annotation.iloc[idx].loc[atributes].to_dict()}
                
                #Check if we have files without anotation
        if sort:
            for id in self.edfDefDict.keys():
                if ("annotation" in self.edfDefDict[id])==False: 
                    self.edfDefDict[id]["deathFlag"]=True
                    self.edfDefDict[id]["reson"]="annotaiton missing"

def getNumberOfAnnotators():
    datadir=r"C:\Users\Mads-\OneDrive\Dokumenter\Universitet\4. Semester\02466 Fagprojekt - Bachelor i kunstig intelligens og data\dataEEG\data_farrahtue_EEG"
    loader=loadData.json_maker()
    loader.find_edf(datadir)
    loader.add_annotations(datadir,sheet_names=[2,3,4],sort=True)
    pass
    flist = []
    with open("filenames.txt", "r") as fh:
        filenames = fh.read().splitlines()

    for i in filenames:
        if i in loader.no_file:
            flist.append(loader.no_file[i]['annotation']['Reader'])
#Debugging
# import loadData
# import os
# datadir=r"C:\Users\Mads-\OneDrive\Dokumenter\Universitet\4. Semester\02466 Fagprojekt - Bachelor i kunstig intelligens og data\dataEEG\data_farrahtue_EEG"
# loader=loadData.json_maker()
# loader.find_edf(datadir)
# loader.add_annotations(datadir,sheet_names=[2,3,4],sort=True)
# pass
# flist = []
# with open("filenames.txt", "r") as fh:
#     filenames = fh.read().splitlines()
#
# for i in filenames:
#     if i in loader.no_file:
#         flist.append(loader.no_file[i]['annotation']['Recording Length [seconds]'])
# pass
