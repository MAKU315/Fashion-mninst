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




def create_sprite_image(images):
    """Returns a sprite image consisting of images passed as argument. Images should be count x width x height"""
    if isinstance(images, list):
        images = np.array(images)
    img_h = images.shape[1]
    img_w = images.shape[2]
    n_plots = int(np.ceil(np.sqrt(images.shape[0])))

    spriteimage = np.ones((img_h * n_plots, img_w * n_plots))

    for i in range(n_plots):
        for j in range(n_plots):
            this_filter = i * n_plots + j
            if this_filter < images.shape[0]:
                this_img = images[this_filter]
                spriteimage[i * img_h:(i + 1) * img_h,
                j * img_w:(j + 1) * img_w] = this_img

    return spriteimage



def vector_to_matrix_mnist(mnist_digits):
    import numpy as np
    """Reshapes normal mnist digit (batch,28*28) to matrix (batch,28,28)"""
    return np.reshape(mnist_digits, (-1, 28, 28))

def invert_grayscale(mnist_digits):
    """ Makes black white, and white black """
    return 255 - mnist_digits

def get_sprite_image(to_visualise, do_invert=True):
    to_visualise = vector_to_matrix_mnist(to_visualise)
    if do_invert:
        to_visualise = invert_grayscale(to_visualise)
    return create_sprite_image(to_visualise)
