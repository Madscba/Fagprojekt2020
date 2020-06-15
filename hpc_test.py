import numpy as np


l1 = np.append(np.repeat('Yes',40),np.repeat('No',40))
#
#
# with open("test.txt", "w+") as myfile:
#     for ele in l1:
#         myfile.write(ele+'\n')
i=2
np.save((f'train_acc_{i}.npy'), l1)
print("\n reached:")
# train_loss_data = np.asarray(train_loss)
# np.save(('train_loss_%i.npy' % i), train_loss_data)
# valid_acc_data = np.asarray(val_acc)
# np.save(('valid_acc_%i.npy' % i), valid_acc_data)
# valid_loss_data = np.asarray(val_loss)
# np.save(('valid_loss_%i.npy' % i), valid_loss_data)
# wrong_guesses_data = np.asarray(wrong_guesses)
# np.save(('wrong_guesses_%i.npy' % i), wrong_guesses_data)
# wrong_pred_data = np.asarray(wrong_predictions)
# np.save(('wrong_guesses_class_%i.npy' % i), wrong_pred_data)

