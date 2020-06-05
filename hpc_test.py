import numpy as np
import subprocess

l1 = np.append(np.repeat('Yes',40),np.repeat('No',40))


with open("test.txt", "w+") as myfile:
    for ele in l1:
        myfile.write(ele+'\n')

