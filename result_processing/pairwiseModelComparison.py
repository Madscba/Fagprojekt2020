import scipy.stats
import numpy as np
import scipy.stats as st
import json, os, itertools
import pandas as pd

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
def extractLabelsAndModelPredictionsJson(path, filename, classifiers,baseline=False):
    y_true,y_hat = np.array([]),np.array([])
    filepath = os.path.join(path, filename)
    with open(filepath) as json_file:
        tempData = json.load(json_file)
        for c in classifiers:
            for i in range(5):
                if c==classifiers[0]:
                    y_true = np.append(y_true,tempData[f'fold_{i}']['True'])
                y_hat = np.append(y_hat, tempData[f'fold_{i}'][f'{c}_predict'])
    # tempData{''}
    y_hat = np.where(y_hat == 'Yes', 1, 0)
    y_true = np.where(y_true == 'Yes', 1, 0)
    # if baseline == True:
    #     data_baseline = pd.read_csv(filepath.replace("_predict.json",""))
    base_results = np.repeat(1,len(y_true))
    return y_true,y_hat,base_results

        # ex_balanced_train_fea0
        # ex_unbalanced_train_spec0
def extractInformationTxtFile():
    pass
    return

def computeJeffreyIntervals(path = r"C:\Users\Mads-\OneDrive\Dokumenter\Universitet\4. Semester\downloads from hpc\ClassifierTestLogs. Ex_10_split_bal_unbal",classifiers=['SVM'],files="",n_splits=10):
    if files == "":
        raise Exception("No files given")
    for idx, file in enumerate(files):
        file_results= computeJeffreyInveralFromFile(file,path=path,classifiers=classifiers[idx],n_splits=n_splits)
        if idx ==0:
            jeff_interval=file_results
        else:
            jeff_interval = pd.concat([jeff_interval,file_results])
    return jeff_interval
def computeJeffreyInveralFromFile(file,path="",classifiers=[],n_splits=10):
    alpha = 0.05
    data = []
    for k in range(n_splits):
        file_path =file.replace('{i}',f'{k}')
        y_true,y_hat,_ = extractLabelsAndModelPredictionsJson(path, file_path, classifiers=classifiers)
        for i in range(len(classifiers)):
            [thetahatA, CIA] = jeffrey_interval(y_true, y_hat[0+len(y_true)*i:len(y_true)*(1+i)], alpha=alpha)
            data.append([classifiers[i],thetahatA,CIA,file_path])
    classifier_results = pd.DataFrame(data, columns=['Model', 'ThetaA', 'Confidence interval', 'File'])
    return classifier_results


def computeMcNemarComparisons(path = r"C:\Users\Mads-\OneDrive\Dokumenter\Universitet\4. Semester\downloads from hpc\ClassifierTestLogs. Ex_10_split_bal_unbal",classifiers=['SVM'],files="",n_splits=10):
    if files == "":
        raise Exception("No files given")
    for idx, file in enumerate(files):
        bool = True if idx ==0 else False
        mcNemar_pred,model= extractMcNemarFromFile(file,path=path,classifiers=classifiers[idx],n_splits=n_splits,baseline= bool)
        if idx ==0:
            mcNemar_pred_all = mcNemar_pred
            model_all = model
        else:
            mcNemar_pred_all = np.vstack((mcNemar_pred_all,mcNemar_pred[:,0:5080]))
            model_all = np.hstack((model_all,model))

    pairs = list(itertools.combinations(['SVM','LDA','RF' ,'True Label'], 2))
    data = []
    for n,m in pairs:
        [thetahatA, CIA,p] = mcnemar(mcNemar_pred_all[np.where(model_all=='True Label')[0][0],:], mcNemar_pred_all[np.where(model_all==n)[0][0],:],mcNemar_pred_all[np.where(model_all==m)[0][0],:],alpha=0.05)
        data.append([(n,m), thetahatA, CIA])

    return mcNemar_pred_all,model_all
def extractMcNemarFromFile(file="",path="",classifiers=["SVM"],n_splits=10,baseline=True):
    for k in range(n_splits):
        file_path =file.replace('{i}',f'{k}')
        y_true,y_hat, baseline= extractLabelsAndModelPredictionsJson(path, file_path, classifiers=classifiers,baseline=baseline)
    model_name = np.append(classifiers,np.array(['True Label']))
    model_pred = y_true
    for i in range(len(classifiers)):
        model_pred = np.vstack((model_pred,y_hat[0+i*len(y_true):(1+i)*len(y_true)]))
    return model_pred, model_name



if __name__ == '__main__':
    with open('file:///C:/Users/Andre/Downloads/ex_fea_bal0_predict.json') as json_file:
        tempData = json.load(json_file)
    with open('file:///C:/Users/Andre/Downloads/ex_fea_unbal0_predict.json') as json_file1:
        tempData = json.load(json_file1)
    #     tempData1 = json.load(json_file1)
    # pass
    # files_bal = ['ex_balanced_train_fea{i}_predict.json','ex_balanced_train_spec{i}_predict.json']
    files_bal = ['ex_fea_bal{i}_predict.json','ex_spec_bal{i}_predict.json']
    classifiers_bal = [['SVM', 'LDA'],['RF']]
    # jeff_bal = computeJeffreyIntervals(classifiers = classifiers_bal,files=files_bal,n_splits=1)
    pass
    # files_unbal = ['ex_unbalanced_train_fea{i}_predict.json','ex_unbalanced_train_spec{i}_predict.json']
    files_unbal = ['ex_fea_unbal{i}_predict.json', 'ex_spec_unbal{i}_predict.json']
    classifiers_unbal = [['SVM', 'LDA'],['RF']]
    # jeff_unbal = computeJeffreyIntervals(classifiers=classifiers_unbal, files=files_unbal,n_splits=1)
    pass

    # baseline,SVM, LDA, RF
    #10 runs, 5 splits pr. run. Each run has spec and feature
    computeMcNemarComparisons(files=files_bal,classifiers = classifiers_bal,n_splits=1)