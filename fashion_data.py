import mnist_reader
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import matplotlib

X_train, y_train = mnist_reader.load_mnist('C:/Users/korea/Desktop/fashion-mnist-master', kind='train')
X_test, y_test = mnist_reader.load_mnist('C:/Users/korea/Desktop/fashion-mnist-master', kind='t10k')

print(X_train[0:10])
print(y_train[0:10])
X_0 = X_train[0]
X_0 = X_0.reshape(28, 28)
plt.imshow(X_0, cmap = matplotlib.cm.binary,
           interpolation="nearest")
plt.show()

# show the image
def plot_digits(instances, images_per_row=10, **options):
    size = 28
    images_per_row = min(len(instances), images_per_row)
    images = [instance.reshape(size,size) for instance in instances]
    n_rows = (len(instances) - 1) // images_per_row + 1
    row_images = []
    n_empty = n_rows * images_per_row - len(instances)
    images.append(np.zeros((size, size * n_empty)))
    for row in range(n_rows):
        rimages = images[row * images_per_row : (row + 1) * images_per_row]
        row_images.append(np.concatenate(rimages, axis=1))
    image = np.concatenate(row_images, axis=0)
    plt.imshow(image, cmap = matplotlib.cm.binary, **options)

plt.figure(figsize=(9,9))
example_images = np.r_[X_train[:12000:600], X_train[13000:30600:600], X_train[30600:60000:590]]
plot_digits(example_images, images_per_row=10)
plt.show()

