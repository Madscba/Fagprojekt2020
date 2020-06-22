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
def extractLabelsAndModelPredictionsJson(path, filename, classifiers,baseline=False,previous_testidx=None):
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

    data_baseline = pd.read_csv(filepath.replace("_predict.json",""))
    base_results = np.repeat(1,len(y_true))
    # with open(r'C:\Users\Mads-\Downloads\Feature_unbal_final1_debuglog.json') as json0_file:
    #     tempData0= json.load(json0_file)
    with open(r'C:\Users\Mads-\Downloads\Feture_bal_final1_debuglog.json') as json1_file:
        tempData1 = json.load(json1_file)
    with open(r'C:\Users\Mads-\Downloads\Spectrograms_bal_final1_debuglog.json') as json2_file:
        tempData2 = json.load(json2_file)
    # with open(r'C:\Users\Mads-\Downloads\Spectrograms_unbal_final1_debuglog.json') as json3_file:
    #     tempData3 = json.load(json3_file)
    if type(previous_testidx)!=np.ndarray:
        previous_testidx_testidx = np.array([])
        for i in range(5):
            previous_testidx = np.append(previous_testidx, tempData[f'fold_{i}']['index'])
    else:
        new_testidx=np.array([])
        for i in range(5):
            new_testidx = np.append(new_testidx, tempData[f'fold_{i}']['index'])
        allign(previous_testidx,new_testidx,y_hat,y_true)
    return y_true,y_hat,base_results,previous_testidx

        # ex_balanced_train_fea0
        # ex_unbalanced_train_spec0
def allign(previous_testidx,new_testidx,y_hat,y_true):
    alligned_y_hat, alligned_y_true = np.array([]),np.array([])
    validation_idx = np.array([])
    if previous_testidx[0]==None:
        previous_testidx=previous_testidx[1:]

    for idx,i in enumerate(previous_testidx):
        if idx%2==0:
            matches = np.where(i == new_testidx)
            for j in matches[0]:
                if previous_testidx[idx+1] == j:
                    alligned_y_hat = np.append(alligned_y_hat,[y_hat[matches[0][j]/2],y_hat[matches[0][j]+1]/2+1])
                    alligned_y_true = np.append(alligned_y_true,[y_true[matches[0][j]/2],y_true[matches[0][j]/2+1]])
                    validation_idx = np.append(validation_idx,[0])
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
        file_path =file.replace('{i}',f'{k+1}')
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
        if bool:
            mcNemar_pred,model,test_idx= extractMcNemarFromFile(file,path=path,classifiers=classifiers[idx],n_splits=n_splits,baseline= bool)
        else:
            mcNemar_pred, model,_ = extractMcNemarFromFile(file, path=path, classifiers=classifiers[idx],n_splits=n_splits, baseline=bool,previous_testidx = test_idx)
        if idx ==0:
            mcNemar_pred_all = mcNemar_pred
            model_all = model
        else:
            mcNemar_pred_all = np.vstack((mcNemar_pred_all,mcNemar_pred))
            model_all = np.hstack((model_all,model))

    pairs = list(itertools.combinations(['True Label','SVM','LDA','RF','GNB'], 2))
    # pairs = list(itertools.combinations(['True Label','SVM','LDA','RF','GNB'], 2))
    Baseline_SMVF_SVMS_get33matrix(mcNemar_pred_all,model_all)
    measurements = []
    data = []
    # for n,m in pairs:
    #     [thetahatA, CIA,p] = mcnemar(mcNemar_pred_all[np.where(model_all=='True Label')[0][0],:], mcNemar_pred_all[np.where(model_all==n)[0][0],:],mcNemar_pred_all[np.where(model_all==m)[0][0],:],alpha=0.05)
    #     measurements.append([thetahatA, CIA])
    #     data.append([(n,m), thetahatA, CIA])

    return data,pairs, measurements
def Baseline_SMVF_SVMS_get33matrix(mcNemar_pred_all,model_all):
    model = ['',"SVM(F)","Baseline",'','SVM(S)']
    pairs = list(itertools.combinations([2,1,4], 2))
    data, measurements = [], []
    for i,j in pairs:
        [thetahatA, CIA, p] = mcnemar(mcNemar_pred_all[np.where(model_all == 'True Label')[0][0], :],
                                      mcNemar_pred_all[i, :],
                                      mcNemar_pred_all[j, :], alpha=0.05)
        measurements.append([thetahatA, CIA])
        data.append([(model[i], model[j]), thetahatA, CIA])
    pass

def extractMcNemarFromFile(file="",path="",classifiers=["SVM"],n_splits=10,baseline=True,previous_testidx=None):
    for k in range(n_splits):
        file_path =file.replace('{i}',f'{k+1}')
        if type(previous_testidx)!=np.ndarray:
            y_true,y_hat, y_baseline,previous_testidx= extractLabelsAndModelPredictionsJson(path, file_path, classifiers=classifiers,baseline=baseline)
        else:
            y_true, y_hat, y_baseline,_ = extractLabelsAndModelPredictionsJson(path, file_path, classifiers=classifiers,baseline=baseline,previous_testidx=previous_testidx)

    model_name = np.append(np.array(['True Label']),classifiers)
    model_pred = y_true
    for i in range(len(classifiers)):
        model_pred = np.vstack((model_pred,y_hat[0+i*len(y_true):(1+i)*len(y_true)]))
    if baseline:
        model_name = np.append(model_name,"Baseline")
        model_pred = np.vstack((model_pred,y_baseline))
    return model_pred, model_name,previous_testidx



def formatMcNemar(model, values):
    data = []
    column = []
    for i,j in model:
        column.append(str(i)+","+str(j))
    for j in values:
        data.append(str(np.round(j[0],2))+" "+str(np.round(j[1],2)))
    pass

    s = ""

    for i in range(len(data)):
        s = s+ model[i][0] + "," + model[i][1]+ " &" + str(np.round(values[i][0],2)) + " &"+ str(np.round(values[i][1],2)) +"//"
    s = s.replace("True Label","Baseline")

    return s


if __name__ == '__main__':
    files_bal = ['Feture_bal_final{i}_predict.json','Spectrograms_bal_final{i}_predict.json']
    # files_bal = ['ex_fea_bal{i}_predict.json','ex_spec_bal{i}_predict.json']
    # classifiers_bal = [["RF","SVM","LDA","GNB"],["RF","SVM","LDA","GNB"]]
    classifiers_bal = [["SVM"],["SVM"]]

    # jeff_bal = computeJeffreyIntervals(path=r'C:\Users\Mads-\Downloads',classifiers = classifiers_bal,files=files_bal,n_splits=1)

    files_unbal = ['Feature_unbal_final{i}_predict.json','Spectrograms_unbal_final{i}_predict.json']
    # files_unbal = ['ex_fea_unbal{i}_predict.json', 'ex_spec_unbal{i}_predict.json']
    classifiers_unbal = [["RF"],["RF"]]
    # jeff_unbal = computeJeffreyIntervals(path=r'C:\Users\Mads-\Downloads',classifiers=classifiers_unbal, files=files_unbal,n_splits=1)
    pass
    # pd.concat([jeff_unbal.round(4)[['Model', 'ThetaA', 'Confidence interval']][0:4],
    #            jeff_unbal.round(4)[['ThetaA', 'Confidence interval']][4:8]], axis=1).to_latex()
    # pd.concat([jeff_bal.round(4)[['Model', 'ThetaA', 'Confidence interval']][0:4],
    #            jeff_bal.round(4)[['ThetaA', 'Confidence interval']][4:8]], axis=1).to_latex()
    # baseline,SVM, LDA, RF
    #10 runs, 5 splits pr. run. Each run has spec and feature
    complete_mcnemar_bal, model_pairs_bal, values_bal  = computeMcNemarComparisons(path=r'C:\Users\Mads-\Downloads',files=files_bal,classifiers = classifiers_bal,n_splits=1)
    complete_mcnemar_unbal, model_pairs_unbal, values_unbal =computeMcNemarComparisons(path=r'C:\Users\Mads-\Downloads',files=files_unbal,classifiers = classifiers_unbal,n_splits=1)
    s1 = formatMcNemar(model_pairs_bal,values_bal)
    s2 = formatMcNemar(model_pairs_unbal,values_unbal)
    pd.DataFrame(np.vstack((data, column))).to_latex()
    pass
