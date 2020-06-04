from CNN.modifyCNN import model, list
import torch.optim as optim
import torch.nn as nn
import numpy as np
from Preprossering.PreprosseringPipeline import preprossingPipeline

def train_CNN(model,batch_size,num_epochs,x_train,x_valid):
    optimizer = optim.Adam(model.parameters)
    criterion = nn.CrossEntropyLoss()

    num_samples_train = x_train.shape[0]
    num_batches_train = num_samples_train // batch_size
    num_samples_valid = x_valid.shape[0]
    num_batches_valid = num_samples_valid // batch_size

    # setting up lists for handling loss/accuracy
    train_acc, train_loss = [], []
    valid_acc, valid_loss = [], []
    test_acc, test_loss = [], []
    cur_loss = 0
    losses = []

    get_slice = lambda i, size: range(i * size, (i + 1) * size)
    for epoch in range(num_epochs):
        # Forward -> Backprob -> Update params
        ## Train
        cur_loss = 0
        model.train()
        for i in range(num_batches_train):
            slce = get_slice(i, batch_size)
            x_batch = Variable(torch.from_numpy(x_train[slce]))
            output = net(x_batch)

            # compute gradients given loss
            target_batch = Variable(torch.from_numpy(targets_train[slce]).long())
            batch_loss = criterion(output, target_batch)
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            cur_loss += batch_loss
        losses.append(cur_loss / batch_size)

        model.eval()
        ### Evaluate training
        train_preds, train_targs = [], []
        for i in range(num_batches_train):
            slce = get_slice(i, batch_size)
            x_batch = Variable(torch.from_numpy(x_train[slce]))

            output = model(x_batch)
            preds = torch.max(output, 1)[1]

            train_targs += list(targets_train[slce])
            train_preds += list(preds.data.numpy())

        ### Evaluate validation
        val_preds, val_targs = [], []
        for i in range(num_batches_valid):
            slce = get_slice(i, batch_size)
            x_batch = Variable(torch.from_numpy(x_valid[slce]))

            output = model(x_batch)
            preds = torch.max(output, 1)[1]
            val_preds += list(preds.data.numpy())
            val_targs += list(targets_valid[slce])

        train_acc_cur = accuracy_score(train_targs, train_preds)
        valid_acc_cur = accuracy_score(val_targs, val_preds)

        train_acc.append(train_acc_cur)
        valid_acc.append(valid_acc_cur)

        if epoch % 10 == 0:
            print("Epoch %2i : Train Loss %f , Train acc %f, Valid acc %f" % (
                epoch + 1, losses[-1], train_acc_cur, valid_acc_cur))
        return train_acc, valid_acc

C = preprossingPipeline(
    BC_datapath=r"C:\Users\johan\iCloudDrive\DTU\KID\4. semester\Fagprojekt\Data\dataEEG")

path_spec = r'C:\Users\johan\iCloudDrive\DTU\KID\4. semester\Fagprojekt\Spektrograms'
N=10
spectrogram_is_usable,labels__is_usable_spec,_,_= C.make_label(make_from_filenames=False,quality=None,is_usable="Yes",max_files=N,path = path_spec) #18 files = 2074
spectrogram_not_usable,labels_not_usable_spec,_,_= C.make_label(make_from_filenames=False,quality=None,is_usable='No',max_files=N ,path = path_spec) #18 files = 1926


x_train = np.vstack((spectrogram_is_usable[:2074,:],spectrogram_not_usable[:1926,:]))
y_train = np.hstack((labels__is_usable_spec[:2074],labels_not_usable_spec[:1926]))

x_valid = np.vstack((spectrogram_is_usable[2074:,:],spectrogram_not_usable[1926:,:]))
y_test = np.hstack((labels__is_usable_spec[2074:],labels_not_usable_spec[1926:]))

batch_size = 10
num_epochs = 10