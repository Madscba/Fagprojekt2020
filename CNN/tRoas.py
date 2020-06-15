import sys
sys.path.append('/zhome/87/9/127623/Fagprojekt/Fagprojekt2020')
from CNN.modifyCNN import VGG16, freeze_parameters, grad_parameters, list_of_features, check_grad
import torch.optim as optim
import torch
from skimage.transform import resize
from sklearn import preprocessing
from sklearn.utils import shuffle
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
from Preprossering.PreprosseringPipeline import preprossingPipeline
from sklearn.linear_model import LogisticRegression
np.random.seed(42)

def new_classifier(model, train_data, labels, channels):
    X = np.zeros((int(len(labels)/channels),channels))
    y = np.zeros((int(len(labels)/channels)))
    for i in range(int(len(labels)/channels)):
        X[i,:] = np.argmax(model(train_data[i*channels:(i+1)*channels,:,:,:]).data.numpy(), axis=-1)
        y[i] = labels[channels*i]
    clf = LogisticRegression().fit(X, y)
    return clf
def classify(output, balance):
    softmax = torch.softmax(output,dim=1).data.numpy()
    new_balance = 1- balance
    new_preds = np.zeros((np.shape(softmax)))
    for i in range(np.shape(softmax)[0]):
        new_preds[i,0] = new_balance*softmax[i,0]
        new_preds[i,1] = balance*softmax[i,1]
    return new_preds

def test_CNN(model,X_train,y_train,X_valid,y_valid,w_id,batch_size,num_epochs,preprocessed=False):
    num_samples = X_train.shape[0]
    num_batches = int(np.ceil(num_samples / float(batch_size)))
    l1 = preprocessing.LabelEncoder()
    t1 = l1.fit_transform(y_train)
    l2 = preprocessing.LabelEncoder()
    t2 = l2.fit_transform(y_valid)

    num_test_samples = X_valid.shape[0]
    num_test_batches = int(np.ceil(num_test_samples / float(batch_size)))

    # setting up lists for handling loss/accuracy
    train_loss, val_loss = [], []
    train_cost, val_cost = [], []
    for epoch in range(num_epochs):
        # Forward -> Backprob -> Update params
        ## Train
        correct = 0
        model.train()
        for i in range(num_batches):
            if i % 10 == 0:
                print("\n {}, still training...".format(i), end='')
            idx = range(i * batch_size, np.minimum((i + 1) * batch_size, num_samples))
            index = idx[-1]-idx[0]+1
            if preprocessed==False:
                batch_image = np.zeros((index,224,224))
                for j in range(index):
                    image_resized = resize(X_train[idx[j]], (224, 224), anti_aliasing=True)
                    batch_image[j,:,:] = image_resized
                X_batch_tr = Variable(torch.from_numpy(batch_image))
                y_batch_tr = Variable(torch.from_numpy(t1[idx]).long())
                optimizer.zero_grad()
                output = model(X_batch_tr.unsqueeze(1).float())
            else:
                X_batch_tr = X_train[idx,:,:,:]
                y_batch_tr = Variable(torch.from_numpy(t1[idx]).long())
                optimizer.zero_grad()
                output = model(X_batch_tr.float())

            batch_loss = criterion(output, y_batch_tr)
            train_loss.append(batch_loss.data.numpy())

            batch_loss.backward()
            optimizer.step()

            preds = np.argmax(output.data.numpy(), axis=-1)
            correct += np.sum(y_batch_tr.data.numpy() == preds)

        train_acc = correct / float(num_samples)
        train_cost.append(np.mean(train_loss))

        correct2 = 0
        model.eval()
        wrong_guesses = []
        wrong_predictions = []
        for i in range(num_test_batches):
            if i % 10 == 0:
                print("\n {}, now validation...".format(i), end='')
            idx = range(i * batch_size, np.minimum((i + 1) * batch_size, num_test_samples))
            index = idx[-1] - idx[0] + 1
            if preprocessed==False:
                batch_image = np.zeros((index,224,224))
                for j in range(index):
                    image_resized = resize(X_valid[idx[j]], (224, 224), anti_aliasing=True)
                    batch_image[j,:,:] = image_resized
                X_batch_v = Variable(torch.from_numpy(batch_image))
                y_batch_v = Variable(torch.from_numpy(t2[idx]).long())
                output = model(X_batch_v.unsqueeze(1).float())
            else:
                X_batch_v = X_valid[idx,:,:,:]
                y_batch_v = Variable(torch.from_numpy(t2[idx]).long())
                output = model(X_batch_v.float())

            batch_loss = criterion(output, y_batch_v)

            val_loss.append(batch_loss.data.numpy())
            preds = np.argmax(output.data.numpy(), axis=-1)
            eval_preds = y_batch_v.data.numpy() == preds
            for k in range(index):
                if eval_preds[k] == False:
                    wrong_guesses.append(w_id[idx[k]])
                    wrong_predictions.append(preds[k])
                else:
                    correct2 += 1

        val_acc = correct2 / float(num_test_samples)
        val_cost.append(np.mean(val_loss))

        if epoch % 10 == 0:
            print("\n Epoch %2i : Train Loss %f , Train acc %f, Valid acc %f" % (
                epoch + 1, train_cost[-1], train_acc, val_acc))

    return train_acc,train_cost,val_acc,val_cost, wrong_guesses, wrong_predictions, model

def split_dataset(C,path,N,train_split,max_windows,num_channels):
    """ Input: Data and training split (in %)
        Output: Training and test set """
    windows1, labels1, filenames1, window_idx_full1 = C.make_label_cnn(make_from_filenames=None, quality=None,
                                                                       is_usable='Yes', max_files=N, max_windows=max_windows,
                                                                       path=path_s, seed=0, ch_to_include=range(num_channels))
    windows2, labels2, filenames2, window_idx_full2 = C.make_label_cnn(make_from_filenames=None, quality=None,
                                                                       is_usable='No', max_files=N, max_windows=max_windows,
                                                                       path=path_s, seed=0, ch_to_include=range(num_channels))
    n_train_files1 = int(len(filenames1) / 10 * (train_split/10))
    n_train_files2 = int(len(filenames2) / 10 * (train_split/10))
    a = np.array(window_idx_full1)
    b = np.array(window_idx_full2)
    j1 = np.unique(a[:, 0], return_counts=True)
    j2 = np.unique(b[:,0], return_counts=True)
    j1 = np.array(j1)
    j2 = np.array(j2)
    j1 = np.asarray(j1[1, :], dtype='int64')
    j2 = np.asarray(j2[1,:], dtype='int64')
    n1, n2 = 0, 0
    for i in range(n_train_files1):
        n1 += j1[i]
    for i in range(n_train_files2):
        n2 += j1[i]
    wt1 = windows1[:n1,:,:,:]
    wt2 = windows2[:n2,:,:,:]
    w1 = torch.cat((wt1,wt2))
    ww1 = windows1[n1:,:,:,:]
    ww2 = windows2[n2:,:,:,:]
    w2 = torch.cat((ww1,ww2))
    l1 = labels1[:n1]+labels2[:n2]
    l2 = labels1[n1:]+labels2[n2:]
    wid = window_idx_full1[n1:]+window_idx_full2[n2:]
    train_windows, train_labels = shuffle(w1,l1)
    test_windows, test_labels, test_id = shuffle(w2,l2,wid)
    return train_windows.detach(), test_windows.detach(), train_labels, test_labels, test_id


C=preprossingPipeline(BC_datapath=r"C:\Users\johan\iCloudDrive\DTU\KID\4. semester\Fagprojekt\Data\dataEEG")
path_s = r'D:\spectograms_rgb'
criterion = nn.CrossEntropyLoss()
X_train, X_valid, Y_train, Y_valid,windows_id = split_dataset(C,path_s,N=100,train_split=80,max_windows=30,num_channels=2)
modelA = VGG16()
freeze_parameters(modelA,feature_extracting=True)
list2 = np.array(list_of_features(modelA))
modelB = VGG16()
freeze_parameters(modelB,feature_extracting=True)
for i in range(2):
    PATH = '/zhome/87/9/127623/Fagprojekt'
    if i == 0:
        activation_list = np.array([26, 27, 28, 29, 30, 31])
        grad_parameters(modelA, list(list2[activation_list]))
        optimizer = optim.Adam(modelA.parameters(), lr=0.005)
        train_accA, train_lossA, val_accA, val_lossA, wrong_guessesA, wrong_predictionsA, modelA = test_CNN(modelA, X_train, Y_train, X_valid,
                                                                                  Y_valid, windows_id, batch_size=64,
                                                                                  num_epochs=2, preprocessed=True)
        torch.save(modelA.state_dict(), 'modelA_pc')
        print("\n Model A accuracy: ", val_accA)

    else:
        activation_list = np.array([24,25,26, 27, 28, 29, 30, 31])
        grad_parameters(modelB, list(list2[activation_list]))
        optimizer = optim.Adam(modelB.parameters(), lr=0.005)
        train_accB, train_lossB, val_accB, val_lossB, wrong_guessesB, wrong_predictionsB, modelB = test_CNN(modelB, X_train, Y_train, X_valid,
                                                                                  Y_valid, windows_id, batch_size=64,
                                                                                  num_epochs=2, preprocessed=True)
        torch.save(modelB.state_dict(), 'modelB_pc')
        print("Model B accuracy: ", val_accB)

    #train_acc_data = np.asarray(train_acc)
    #np.save((f'train_acc_pc_{i}.npy'), train_acc_data)
    print("reached:")
    #train_loss_data = np.asarray(train_loss)
    #np.save((f'train_loss_pc_{i}.npy'), train_loss_data)
    #valid_acc_data = np.asarray(val_acc)
    #np.save((f'valid_acc_pc_{i}.npy'), valid_acc_data)
    #valid_loss_data = np.asarray(val_loss)
    #np.save((f'valid_loss_pc_{i}.npy'), valid_loss_data)
    #wrong_guesses_data = np.asarray(wrong_guesses)
    #np.save((f'wrong_guesses_pc_{i}.npy'), wrong_guesses_data)