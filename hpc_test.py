import numpy as np

print("Hello World!")

l1 = np.append(np.repeat('Yes',40),np.repeat('No',40))
with open("test.txt", "w") as myfile:
    for ele in l1:
        myfile.write(ele+'\n')