from CNN.trainCNN2 import *
from flashtorch.saliency import Backprop
import matplotlib.pyplot as plt
from flashtorch.utils import apply_transforms, denormalize, format_for_plotting, visualize


x1 = x_train[0,:]
l1 = preprocessing.LabelEncoder()
t1 = l1.fit_transform(y_train)
batch_image = np.zeros((1,224,224))
image_resized = resize(x1, (224, 224), anti_aliasing=True)
plt.imshow(image_resized)
plt.show()
batch_image[0,:,:] = image_resized
X_image = Variable(torch.from_numpy(batch_image))
backprop = Backprop(model)
output = model(X_image.unsqueeze(1).float())
gradients = backprop.calculate_gradients(X_image.unsqueeze(1).float(),t1[0])
max_gradients = backprop.calculate_gradients(X_image.unsqueeze(1).float(), t1[0], take_max=True)
visualize(X_image,gradients, max_gradients)


