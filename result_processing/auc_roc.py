import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import os
from result_processing.sklearn_roc import plot_roc_curve
from sklearn.metrics import auc
import json

def kfold():
    def kfold_roc(y_true, y_pred, model):
        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)

        fig, ax = plt.subplots()

        for i in range(3):
            viz = plot_roc_curve(y=np.asarray(y_true_feat_b[:int(test_size_feat_b[i])], dtype='float64'),
                                 y_pred=np.asarray(SVM_prob_b[:int(test_size_feat_b[i])], dtype='float64'),
                                 name='ROC fold {}'.format(i), ax=ax)
            interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(viz.roc_auc)

        ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
                label='Chance', alpha=.8)

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        ax.plot(mean_fpr, mean_tpr, color='b',
                label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
                lw=2, alpha=.8)

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                        label=r'$\pm$ 1 std. dev.')

        ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
               title=model)
        ax.legend(loc="lower right")
        plt.show()

def kfold_roc(y_true,y_pred, test_sizes, title):
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    fig, ax = plt.subplots()

    for i in range(5):
        if i == 0:
            viz = plot_roc_curve(y_true[:int(test_sizes[i])], y_pred[:int(test_sizes[i])],
                                 name='ROC fold {}'.format(i), ax=ax)
        else:
            viz = plot_roc_curve(y_true[int(test_sizes[i-1]):int(test_sizes[i])], y_pred[int(test_sizes[i-1]):int(test_sizes[i])],
                                 name='ROC fold {}'.format(i), ax=ax)
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)

    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
            label='Chance', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(mean_fpr, mean_tpr, color='b',
            label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
            lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                    label=r'$\pm$ 1 std. dev.')

    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
           title=title)
    ax.legend(loc="lower right")
    plt.show()

def roc_plot(p, y):
    '''
    Plots the receiver operating characteristic (ROC) curve and
    calculates the area under the curve (AUC).
    Input:
         p: Estimated probability of class 1. (Between 0 and 1.)
         y: True class indices. (Equal to 0 or 1.)
    Output:
        AUC: The area under the ROC curve
        TPR: True positive rate
        FPR: False positive rate
    '''
    fpr, tpr, thresholds = roc_curve(y, p)
    AUC = roc_auc_score(y, p)
    plt.plot(fpr, tpr, 'r', [0, 1], [0, 1], 'k')
    plt.grid()
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xticks(np.arange(0, 1.1, .1))
    plt.yticks(np.arange(0, 1.1, .1))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC \n AUC={:.3f}'.format(AUC))
    plt.show()

    return AUC, tpr, fpr

def load_prob(path,filename,classifiers,representation='spec'):
    filepath = os.path.join(path, filename)
    test_size = np.array([])
    RF_prob = np.array([])
    SVM_prob = np.array([])
    LDA_prob = np.array([])
    y_true = np.array([])
    test_size = np.array([])
    with open(filepath) as json_file:
        tempData = json.load(json_file)
        for i in range(5):
            fold_labels = tempData[f'fold_{i}']['True']
            y_true = np.append(y_true, fold_labels)
            test_size = np.append(test_size, np.shape(fold_labels)[0])
            for c in classifiers:
                if representation == 'spec':
                    if c == 'RF':
                        A = tempData[f'fold_{i}'][f'{c}_prob']
                        A = np.array(A)
                        RF_prob = np.append(RF_prob, A[:,1].T)
                else:
                    if c == 'SVM':
                        A = tempData[f'fold_{i}'][f'{c}_prob']
                        A = np.array(A)
                        SVM_prob = np.append(SVM_prob, A[:,1].T)
                    elif c == 'LDA':
                        A = tempData[f'fold_{i}'][f'{c}_prob']
                        A = np.array(A)
                        LDA_prob = np.append(LDA_prob, A[:,1].T)
    y_true = np.where(y_true == 'Yes', 1, 0)
    return y_true, test_size, RF_prob, SVM_prob, LDA_prob

if __name__ == '__main__':
    path = r'D:\Johannes results\Classifier_results_with_probabilities'
    files = ['ex_bal_proba_fea0_predict.json','ex_bal_proba_spec0_predict.json','ex_unbal_proba_fea0_predict.json','ex_unbal_proba_spec0_predict.json']
    y_true_feat_b, test_size_feat_b, _, SVM_prob_b, LDA_prob_b = load_prob(path,files[0],['RF','SVM','LDA'],representation='feature')
    y_true_spec_b, test_size_spec_b, RF_prob_b, _, _ = load_prob(path,files[1],['RF','SVM','LDA'])
    y_true_feat_ub, test_size_feat_ub, _, SVM_prob_ub, LDA_prob_ub = load_prob(path,files[2],['RF','SVM','LDA'],representation='feature')
    y_true_spec_ub, test_size_spec_ub, RF_prob_ub, _, _ = load_prob(path,files[3],['RF','SVM','LDA'])
    kfold_roc(y_true=np.asarray(y_true_feat_b, dtype='float64'),
              y_pred=np.asarray(SVM_prob_b, dtype='float64'),
              test_sizes=test_size_feat_b,
              title='ROC for SVM on feature vectors (balanced training set)')
    kfold_roc(y_true=np.asarray(y_true_feat_b, dtype='float64'),
              y_pred=np.asarray(LDA_prob_b, dtype='float64'),
              test_sizes=test_size_feat_b,
              title='ROC for LDA on feature vectors (balanced training set)')
    kfold_roc(y_true=np.asarray(y_true_feat_ub, dtype='float64'),
              y_pred=np.asarray(SVM_prob_ub, dtype='float64'),
              test_sizes=test_size_feat_ub,
              title='ROC for SVM on feature vectors (imbalanced training set)')
    kfold_roc(y_true=np.asarray(y_true_feat_ub, dtype='float64'),
              y_pred=np.asarray(LDA_prob_ub, dtype='float64'),
              test_sizes=test_size_feat_ub,
              title='ROC for LDA on feature vectors (imbalanced training set)')
    kfold_roc(y_true=np.asarray(y_true_spec_b, dtype='float64'),
              y_pred=np.asarray(RF_prob_b, dtype='float64'),
              test_sizes=test_size_spec_b,
              title='ROC for RF on spectrograms (balanced training set)')
    kfold_roc(y_true=np.asarray(y_true_spec_ub, dtype='float64'),
              y_pred=np.asarray(RF_prob_ub, dtype='float64'),
              test_sizes=test_size_spec_ub,
              title='ROC for RF on spectrograms (imbalanced training set)')
#AUC, tpr, fpr = roc_plot(np.asarray(SVM_prob_b[:int(test_size_feat_b[0])], dtype='float64'),
 #                            np.asarray(y_true_feat_b[:int(test_size_feat_b[0])], dtype='float64'))



