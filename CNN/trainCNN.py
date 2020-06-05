from CNN.modifyCNN import model, list
import torch.optim as optim
import torch
from skimage.transform import resize
from sklearn import preprocessing
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
from Preprossering.PreprosseringPipeline import preprossingPipeline


def test_CNN(model,X_train,y_train,X_valid,y_valid,batch_size):
    num_samples = X_train.shape[0]
    num_batches = int(np.ceil(num_samples / float(batch_size)))
    correct = 0
    l1 = preprocessing.LabelEncoder()
    t1 = l1.fit_transform(y_train)
    l2 = preprocessing.LabelEncoder()
    t2 = l2.fit_transform(y_valid)

    num_test_samples = X_valid.shape[0]
    num_test_batches = int(np.ceil(num_test_samples / float(batch_size)))

    # setting up lists for handling loss/accuracy
    train_loss, val_loss = [], []

    model.train()
    for i in range(num_batches):
        if i % 10 == 0:
            print("\n {}, still training...".format(i), end='')
        idx = range(i * batch_size, np.minimum((i + 1) * batch_size, num_samples))
        index = idx[-1]-idx[0]+1
        batch_image = np.zeros((index,224,224))
        for j in range(index):
            image_resized = resize(X_train[idx[j]], (224, 224), anti_aliasing=True)
            batch_image[j,:,:] = image_resized
        X_batch_tr = Variable(torch.from_numpy(batch_image))
        y_batch_tr = Variable(torch.from_numpy(t1[idx]).long())

        optimizer.zero_grad()
        output = model(X_batch_tr.unsqueeze(1).float())
        batch_loss = criterion(output, y_batch_tr)
        train_loss.append(batch_loss.data.numpy())

        batch_loss.backward()
        optimizer.step()

        preds = np.argmax(output.data.numpy(), axis=-1)
        correct += np.sum(y_batch_tr.data.numpy() == preds)

    train_acc = correct / float(num_samples)
    train_cost = np.mean(train_loss)

    correct2 = 0
    model.eval()
    for i in range(num_test_batches):
        if i % 10 == 0:
            print("\n {}, now validation...".format(i), end='')
        idx = range(i * batch_size, np.minimum((i + 1) * batch_size, num_test_samples))
        index = idx[-1] - idx[0] + 1
        batch_image = np.zeros((index,224,224))
        for j in range(index):
            image_resized = resize(X_valid[idx[j]], (224, 224), anti_aliasing=True)
            batch_image[j,:,:] = image_resized
        X_batch_v = Variable(torch.from_numpy(batch_image))
        y_batch_v = Variable(torch.from_numpy(t2[idx]).long())

        output = model(X_batch_v.unsqueeze(1).float())
        batch_loss = criterion(output, y_batch_v)

        val_loss.append(batch_loss.data.numpy())
        preds = np.argmax(output.data.numpy(), axis=-1)
        correct2 += np.sum(y_batch_v.data.numpy() == preds)

    val_acc = correct2 / float(num_test_samples)
    validation_cost = np.mean(val_loss)

    return train_acc,train_cost,val_acc,validation_cost

C = preprossingPipeline(
    BC_datapath=r"C:\Users\johan\iCloudDrive\DTU\KID\4. semester\Fagprojekt\Data\dataEEG")

path_spec = r'C:\Users\johan\iCloudDrive\DTU\KID\4. semester\Fagprojekt\Spektrograms'
N=60
spectrogram_is_usable,labels__is_usable_spec,_,_= C.make_label(make_from_filenames=False,quality=None,is_usable="Yes",max_files=N,path = path_spec) #18 files = 2074
spectrogram_not_usable,labels_not_usable_spec,_,_= C.make_label(make_from_filenames=False,quality=None,is_usable='No',max_files=N ,path = path_spec) #18 files = 1926

x_train = np.vstack((spectrogram_is_usable[:900,:],spectrogram_not_usable[:900,:]))
y_train = np.hstack((labels__is_usable_spec[:900],labels_not_usable_spec[:900]))

x_valid = np.vstack((spectrogram_is_usable[900:,:],spectrogram_not_usable[900:,:]))
y_valid = np.hstack((labels__is_usable_spec[900:],labels_not_usable_spec[900:]))

batch_size = 20
num_epochs = 10

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

train_acc, train_loss, val_acc, val_loss = test_CNN(model,x_train,y_train,x_valid,y_valid,batch_size)
print("\n Training accuracy: ", train_acc)
print("\n Validation accuracy: ", val_acc)