"""
Program for intial plots to report

Createt by Andreas 16/04
"""
import os
import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt
from PIL import Image
from random import uniform
from IPython.utils import io

def get_data()
    os.chdir(r'C:\Users\Mads-\Documents\Universitet\4. Semester\02466 Fagprojekt - Bachelor i kunstig intelligens og data\dataEEG')
    print(os. getcwd())
    path_xlsx_file = r'\dataEEG'
    file_name_xlsx = r'\MGH_File_Annotations.xlsx'
    new_path = r'C:\Users\Mads-\Documents\Universitet\4. Semester\02466 Fagprojekt - Bachelor i kunstig intelligens og data\dataEEG\dataEEG\MGH_File_Annotations.xlsx'
    print(os.path.join(path_xlsx_file+file_name_xlsx))
    print(new_path)
    annotation = pd.read_excel(new_path, sheet_name= 3)

