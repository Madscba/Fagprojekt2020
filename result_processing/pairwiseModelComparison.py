import scipy.stats
import numpy as np
import scipy.stats as st
import json, os

#FUNCTION COPIED FROM 02450 COURSE MATERIALS- AUTHOR TUE HERLAU
def jeffrey_interval(y, yhat, alpha=0.05):
    m = sum(y - yhat == 0)
    n = y.size
    a = m+.5
    b = n-m + .5
    CI = scipy.stats.beta.interval(1-alpha, a=a, b=b)
    thetahat = a/(a+b)
    return thetahat, CI

#FUNCTION COPIED FROM 02450 COURSE MATERIALS- AUTHOR TUE HERLAU
def mcnemar(y_true, yhatA, yhatB, alpha=0.05):
    # perform McNemars test
    nn = np.zeros((2,2))
    c1 = yhatA - y_true == 0
    c2 = yhatB - y_true == 0

    nn[0,0] = sum(c1 & c2)
    nn[0,1] = sum(c1 & ~c2)
    nn[1,0] = sum(~c1 & c2)
    nn[1,1] = sum(~c1 & ~c2)

    n = sum(nn.flat);
    n12 = nn[0,1]
    n21 = nn[1,0]

    thetahat = (n12-n21)/n
    Etheta = thetahat

    Q = n**2 * (n+1) * (Etheta+1) * (1-Etheta) / ( (n*(n12+n21) - (n12-n21)**2) )

    p = (Etheta + 1) * (Q-1)
    q = (1-Etheta) * (Q-1)

    CI = tuple(lm * 2 - 1 for lm in scipy.stats.beta.interval(1-alpha, a=p, b=q) )

    p = 2*scipy.stats.binom.cdf(min([n12,n21]), n=n12+n21, p=0.5)
    print("Result of McNemars test using alpha=", alpha)
    print("Comparison matrix n")
    print(nn)
    if n12+n21 <= 10:
        print("Warning, n12+n21 is low: n12+n21=",(n12+n21))

    print("Approximate 1-alpha confidence interval of theta: [thetaL,thetaU] = ", CI)
    print("p-value for two-sided test A and B have same accuracy (exact binomial test): p=", p)

    return thetahat, CI, p

#Get labels y_true and predictions y_hat
def extractLabelsAndModelPredictions(path,filename,classifiers):
    y_true,y_hat = np.array([]),np.array([])
    filepath = os.path.join(path, filename)
    with open(filepath) as json_file:
        tempData = json.load(json_file)
        for c in classifiers:
            for i in range(5):
                if c==classifiers[0]:
                    y_true = np.append(y_true,tempData[f'fold_{i}']['meta']['True'])
                y_hat = np.append(y_hat, tempData[f'fold_{i}'][c])
    # tempData{''}
    y_hat = np.where(y_hat == 'Yes', 1, 0)
    y_true = np.where(y_true == 'Yes', 1, 0)
    return y_true,y_hat

        # ex_balanced_train_fea0
        # ex_unbalanced_train_spec0
def extractInformationFromFile():
    pass
    return

def computeJeffreyIntervals(path = r"C:\Users\Mads-\OneDrive\Dokumenter\Universitet\4. Semester\downloads from hpc\ClassifierTestLogs. Ex_10_split_bal_unbal",classifiers=['SVM'],files=""):
    if files == "":
        raise Exception("No files given")
    n = 10
    alpha = 0.05
    for i in files:
        computeJeffreyInveralFromFile()
def computeJeffreyInveralFromFile(file):
    y_true,y_hat = extractLabelsAndModelPredictions(path, f'ex_balanced_train_fea{i}_predict.json',classifiers=classifiers)
    for i in range(len(classifiers)):
        [thetahatA, CIA] = jeffrey_interval(y_true, y_hat[0+len(y_true)*i:len(y_true)*(1+i)], alpha=alpha)
        print("Model ",classifiers[i],"in file ",i,"Theta point estimate", thetahatA, " CI: ", CIA)


if __name__ == '__main__':
    files = [f'ex_balanced_train_fea{i}_predict.json',f'ex_balanced_train_spec{i}_predict.json']
    computeJeffreyIntervals(classifiers = ['SVM', 'LDA'],files=files)