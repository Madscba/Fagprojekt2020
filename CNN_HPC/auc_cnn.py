import numpy as np
import matplotlib.pyplot as plt
import torch
from result_processing.sklearn_roc import plot_roc_curve
import sklearn
from sklearn.metrics import auc, roc_curve
from CNN.modifyCNN import VGG16



def kfold_roc(y_true,y_pred, test_sizes, title, lr=np.array([0.0001,0.0005,0.001,0.01,0.02])):
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    fig, ax = plt.subplots()

    for i in range(len(lr)):
        if i == 0:
            viz = plot_roc_curve(y_true[:int(test_sizes[i])], y_pred[:int(test_sizes[i])],
                                 name='ROC lr = {}'.format(lr[i]), ax=ax)
        else:
            viz = plot_roc_curve(y_true[int(test_sizes[i-1]):int(test_sizes[i])], y_pred[int(test_sizes[i-1]):int(test_sizes[i])],
                                 name='ROC lr = {}'.format(lr[i]), ax=ax)
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
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(y, p)
    AUC = sklearn.metrics.roc_auc_score(y, p)
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

def load_prob_cnn(paths,X_test,Y_test,num=100):
    softmax = torch.nn.Softmax(dim=0)
    model1 = VGG16()
    model1.load_state_dict(torch.load(paths))
    print("loaded model!")

    new_size = np.array([])
    test_size = np.array([])
    models_prob = np.zeros((2, num*77))
    for i in range(len(paths)):
        new_size = np.append(new_size, Y_test)
        test_size = np.append(test_size, np.shape(new_size)[0])
        cur_model = model1
        cur_model.eval()
        print("\n now evaluating")
        for j in range(num):
            A = cur_model(X_test[j*77:(j+1)*77,:,:,:].float())
            B = A.detach().numpy()
            models_prob[:,j*77:(j+1)*77] = B.T
    mp = torch.from_numpy(models_prob)
    out = softmax(mp)
    output = out.numpy()
    return test_size, np.around(output[1,:],decimals=2)

if __name__ == '__main__':
    model_0_ub_paths = []
    model_1_ub_paths = [r'C:\Users\johan\iCloudDrive\DTU\KID\4. semester\Fagprojekt\Resultater\CNN_ub\model1_l5_ub.pt']
    model_0_b_paths = []
    model_1_b_paths = [r'C:\Users\johan\iCloudDrive\DTU\KID\4. semester\Fagprojekt\Resultater\CNN_b\model1_l5_b.pt']
    #X_test_b = torch.load(r'C:\Users\johan\iCloudDrive\DTU\KID\4. semester\Fagprojekt\Resultater\CNN_b\test_set_b.pt')
    #Y_test_b = np.load(r'C:\Users\johan\iCloudDrive\DTU\KID\4. semester\Fagprojekt\Resultater\CNN_b\test_labels_b.npy')
    X_test_ub = torch.load(r'C:\Users\johan\iCloudDrive\DTU\KID\4. semester\Fagprojekt\Resultater\CNN_ub\test_set_ub.pt')
    Y_test_ub = np.load(r'C:\Users\johan\iCloudDrive\DTU\KID\4. semester\Fagprojekt\Resultater\CNN_ub\test_labels_ub.npy')
    y_true_ub = np.where(Y_test_ub == 'Yes', 1, 0)
    #test_sizes_0_ub, models_prob_0_ub = load_prob_cnn(paths=model_0_ub_paths,X_test=X_test,Y_test=Y_test)
    test_sizes_1_ub, models_prob_1_ub = load_prob_cnn(paths=model_1_ub_paths,X_test=X_test_ub,Y_test=Y_test_ub,num=100)
    #test_sizes_0_b, models_prob_0_b = load_prob_cnn(paths=model_0_b_paths,X_test=X_test,Y_test=Y_test)
    #test_sizes_1_b, models_prob_1_b = load_prob_cnn(paths=model_1_b_paths,X_test=X_test_b,Y_test=Y_test_b)
    AUC_ub, _,_ = roc_plot(p=models_prob_1_ub,y=y_true_ub[:2*77])





