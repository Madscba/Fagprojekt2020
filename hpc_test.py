import numpy as np

print("Hello World!")

l1 = np.append(np.repeat('Yes',40),np.repeat('No',40))
f=open('testResults.txt','w')
for ele in l1:
    f.write(ele+'\n')
f.close()