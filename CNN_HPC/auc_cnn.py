import numpy as np
import matplotlib.pyplot as plt
import torch
from result_processing.sklearn_roc import plot_roc_curve
from sklearn.metrics import auc
from CNN.modifyCNN import VGG16



def kfold_roc(y_true,y_pred, test_sizes, title):
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    fig, ax = plt.subplots()

    for i in range(5):
        if i == 0:
            viz = plot_roc_curve(y_true[:int(test_sizes[i])], y_pred[:int(test_sizes[i])],
                                 name='ROC fold {}'.format(i+1), ax=ax)
        else:
            viz = plot_roc_curve(y_true[int(test_sizes[i-1]):int(test_sizes[i])], y_pred[int(test_sizes[i-1]):int(test_sizes[i])],
                                 name='ROC fold {}'.format(i+1), ax=ax)
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

def load_prob_cnn(paths,X_test,Y_test):
    models = []
    for j in range(5):
        model1 = VGG16()
        models.append(model1.load_state_dict(torch.load(paths[j])))
    new_size = np.array([])
    test_size = np.array([])
    models_prob = np.array([])
    for i in range(5):
        new_size = np.append(new_size, Y_test)
        test_size = np.append(test_size, np.shape(new_size)[0])
        cur_model = models[i]
        cur_model.eval()
        A = cur_model(X_test.float())
        models_prob = np.append(models_prob, A[:,1].T)
    return test_size, models_prob

if __name__ == '__main__':
    model_0_ub_paths = []
    model_1_ub_paths = []
    model_0_b_paths = []
    model_1_b_paths = []
    X_test = torch.load(r'D:\Results Johannes\test_set.pt')
    Y_test = np.load(r'D:\Results Johannes\test_labels.npy')
    test_sizes_0_ub, models_prob_0_ub = load_prob_cnn(paths=model_0_ub_paths,X_test=X_test,Y_test=Y_test)
    test_sizes_1_ub, models_prob_1_ub = load_prob_cnn(paths=model_1_ub_paths,X_test=X_test,Y_test=Y_test)
    test_sizes_0_b, models_prob_0_b = load_prob_cnn(paths=model_0_b_paths,X_test=X_test,Y_test=Y_test)
    test_sizes_1_b, models_prob_1_b = load_prob_cnn(paths=model_1_b_paths,X_test=X_test,Y_test=Y_test)
    kfold_roc(y_true=Y_test,
              y_pred=models_prob_0_ub,
              test_sizes=test_sizes_0_ub,
              title='ROC for CNN1 (imbalanced training set)')
    kfold_roc(y_true=Y_test,
              y_pred=models_prob_1_ub,
              test_sizes=test_sizes_1_ub,
              title='ROC for CNN2 (imbalanced training set)')
    kfold_roc(y_true=Y_test,
              y_pred=models_prob_0_b,
              test_sizes=test_sizes_0_b,
              title='ROC for CNN1 (balanced training set)')
    kfold_roc(y_true=Y_test,
              y_pred=models_prob_1_b,
              test_sizes=test_sizes_1_b,
              title='ROC for CNN2 (balanced training set)')




