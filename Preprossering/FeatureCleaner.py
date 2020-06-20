
from Preprossering.loadData import jsonLoad
import numpy as np
import os

Json_path=r"feature_vectors_remove.json"
raw_folder=r"C:\Users\Andre\Desktop\Fagproject\Feature_vector3"
destination_folder=r"C:\Users\Andre\Desktop\Fagproject\Feature_vector4"

dict=jsonLoad(Json_path)

for file in dict.keys():
    if os.path.exists(os.path.join(destination_folder,f"{file}.npy"))==True:
        print(f"{file} already excisting")
    elif    os.path.exists(os.path.join(raw_folder,f"{file}.npy"))==False:
        print(f"{file} not made yet")
    else:
        vectorold=np.load(os.path.join(raw_folder,f"{file}.npy"))
        vectornew=np.delete(vectorold, dict[file], axis=0)
        np.save(os.path.join(destination_folder,f"{file}.npy"),vectornew)




