import numpy as np
from sklearn import model_selection
from sklearn import preprocessing
from scipy import stats

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

##Mads Christian
def getClassifierAccuracies(x,y,k,x_test,y_test):
    #stadardizing x
    x = preprocessing.scale(x)
    #x = np.round(x,4)

    K = 5
    CV = model_selection.KFold(n_splits=K,shuffle=True)
    #random_state = 12

    svm_predict = np.array([])
    LDA_predict = np.array([])
    KNN_predict = np.array([])
    GNB_predict = np.array([])
    clf_predict = np.array([])
    DecisionTree_predict = np.array([])
    RF_predict = np.array([])
    LR_predict = np.zeros(50).reshape(1,50)

    opt_lambda = np.array([])

    y_true = np.array([])

    c = 0
    measurements_svm = np.array([])
    measurements_dct = np.array([])
    measurements_nnet = np.array([])
    measurements_GNB = np.array([])
    measurements_RF = np.array([])

    # for i in range(10):
    for (train_index, test_index) in CV.split(x,y):
        x_train = x[train_index]
        x_test = x[test_index]

        y_train = y[train_index]
        y_test = y[test_index]
        y_true = np.append(y_true,y_test)

        LDA = LinearDiscriminantAnalysis()
        LDA.fit(x, y)
        LDA_predict = np.append(LDA_predict,LDA.predict(x_test))

        #support vector machine
        m_svm = SVC(gamma = "auto",kernel = "linear")
        m_svm.fit(x_train,y_train)
        svm_predict = np.append(svm_predict,m_svm.predict(x_test))

        m_DecisionTree = DecisionTreeClassifier()
        m_DecisionTree.fit(x_train,y_train)
        DecisionTree_predict = np.append(DecisionTree_predict,m_DecisionTree.predict(x_test))


        clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(20,20,20, 10), random_state=1)
        clf.fit(x_train,y_train)
        clf_predict = np.append(clf_predict,clf.predict(x_test))

        m_gaus = GaussianNB()
        m_gaus.fit(x_train,y_train)
        GNB_predict = np.append(GNB_predict,m_gaus.predict(x_test))

        ranFor = RandomForestClassifier(n_estimators=100, criterion='gini')
        ranFor.fit(x_train,y_train)
        RF_predict = np.append(RF_predict,ranFor.predict(x_test))

    print("svm_acc:", np.mean(y_true == svm_predict))
    print("LDA:", np.mean(y_true == LDA_predict))
    print("Dec:", np.mean(y_true == DecisionTree_predict))
    print("MLP:", np.mean(y_true==clf_predict))
    print("GNB:", np.mean(y_true==GNB_predict))
    print("RandForest:", np.mean(y_true==RF_predict))


    for i in range(10):
        #print("svm_acc:", np.mean(y_true[i*10:i*10+10] == svm_predict[i*10:i*10+10]))
        measurements_svm = np.append(measurements_svm,np.mean(y_true[i*10:i*10+10] == svm_predict[i*10:i*10+10]))
        #print("Dec:", np.mean(y_true[i*10:i*10+10] == DecisionTree_predict[i*10:i*10+10]))
        measurements_dct = np.append(measurements_dct,np.mean(y_true[i*10:i*10+10] == DecisionTree_predict[i*10:i*10+10]))
        #print("MLP:", np.mean(y_true[i*10:i*10+10]==clf_predict[i*10:i*10+10]))
        measurements_nnet = np.append(measurements_nnet,np.mean(y_true[i*10:i*10+10]==clf_predict[i*10:i*10+10]))
        measurements_GNB = np.append(measurements_GNB,np.mean(y_true[i*10:i*10+10]==GNB_predict[i*10:i*10+10]))
        measurements_RF = np.append(measurements_RF,np.mean(y_true[i*10:i*10+10]==RF_predict[i*10:i*10+10]))

        print("svm_acc:", np.mean(y_true == svm_predict))
        print("Dec:", np.mean(y_true == DecisionTree_predict))
        print("MLP:", np.mean(y_true==clf_predict))
        print("GNB:", np.mean(y_true==GNB_predict))
        print("RF:", np.mean(y_true==GNB_predict))

        print(np.std(y_true == svm_predict),np.std(y_true == DecisionTree_predict),np.std(y_true==clf_predict),np.std(y_true==GNB_predict))
#TODO make double cross
def tryNewDiv(x,y,k,x_test,y_test):


    svm_predict = np.array([])
    LDA_predict = np.array([])
    GNB_predict = np.array([])
    clf_predict = np.array([])
    DecisionTree_predict = np.array([])
    RF_predict = np.array([])


    y_true = np.array([])

    x_train = x

    y_train = y

    y_true = np.append(y_true, y_test)

    LDA = LinearDiscriminantAnalysis()
    LDA.fit(x, y)
    LDA_predict = np.append(LDA_predict, LDA.predict(x_test))
    print("Lda done",np.mean(y_true == LDA_predict))
    # support vector machine
    m_svm = SVC(gamma="auto", kernel="linear")
    m_svm.fit(x_train, y_train)
    svm_predict = np.append(svm_predict, m_svm.predict(x_test))
    print("SVM done", np.mean(y_true == svm_predict))
    m_DecisionTree = DecisionTreeClassifier()
    m_DecisionTree.fit(x_train, y_train)
    DecisionTree_predict = np.append(DecisionTree_predict, m_DecisionTree.predict(x_test))
    print("DT done",np.mean(y_true == DecisionTree_predict))
    # clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(20, 20, 20, 10), random_state=1)
    # clf.fit(x_train, y_train)
    # clf_predict = np.append(clf_predict, clf.predict(x_test))
    # print("neural done",np.mean(y_true == clf_predict))
    m_gaus = GaussianNB()
    m_gaus.fit(x_train, y_train)
    GNB_predict = np.append(GNB_predict, m_gaus.predict(x_test))
    print("gaus done",np.mean(y_true == GNB_predict))
    ranFor = RandomForestClassifier(n_estimators=100, criterion='gini')
    ranFor.fit(x_train, y_train)
    RF_predict = np.append(RF_predict, ranFor.predict(x_test))
    print("RF done",np.mean(y_true == RF_predict))

    print("svm_acc:", np.mean(y_true == svm_predict))
    print("LDA:", np.mean(y_true == LDA_predict))
    print("Dec:", np.mean(y_true == DecisionTree_predict))
    # print("MLP:", np.mean(y_true == clf_predict))
    print("GNB:", np.mean(y_true == GNB_predict))
    print("RandForest:", np.mean(y_true == RF_predict))
    return [np.mean(y_true == svm_predict),np.mean(y_true == LDA_predict),np.mean(y_true == DecisionTree_predict),np.mean(y_true == RF_predict)]