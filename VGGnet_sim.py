import mnist_reader
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import matplotlib

import codecs, cv2, datetime, glob, itertools, keras, os, pickle
import re, sklearn, string, sys, tensorflow, time
from random import randint
from keras import backend as K, regularizers, optimizers
from keras.models import load_model, Sequential
from keras.layers import Activation, Dense, Conv2D, MaxPool2D, InputLayer, Flatten, Dropout
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
#from keras.layers import Con
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import np_utils

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

PROJECT_ROOT_DIR = "C:/Users/korea/Desktop/fashion-mnist-master"
def save_fig(fig_id, tight_layout=True):
    path = os.path.join(PROJECT_ROOT_DIR, "images", fig_id + "3.png")
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)

print('Keras version: \t\t%s' % keras.__version__)
print('OpenCV version: \t%s' % cv2.__version__)
print('Scikit version: \t%s' % sklearn.__version__)
print('TensorFlow version: \t%s' % tensorflow.__version__)
# The MNIST dataset has 10 classes, representing the digits 0 through 9.


K.set_image_dim_ordering('tf')

#######################
# Dimension of images #
#######################
img_width  = 28
img_height = 28
channels   = 1

######################
# Parms for learning #
######################
batch_size = 1000
num_epochs = 10
iterations = 1            # Number of iterations / models
# number_of_augmentation = 2 # defines the amount of additional augmentation images of one image
early_stopping = EarlyStopping(monitor='val_loss', patience=5) # Early stopping on val loss - not used

####################
#       Data       #
####################
train_data_dir      = 'C:/Users/korea/Desktop/fashion-mnist-master'
test_data_dir       = 'C:/Users/korea/Desktop/fashion-mnist-master'
classes             = {0: 'T-shirt/top',
                       1: 'Trouser',
                       2: 'Pullover',
                       3: 'Dress',
                       4: 'Coat',
                       5: 'Sandal',
                       6: 'Shirt',
                       7: 'Sneaker',
                       8: 'Bag',
                       9: 'Ankle boot'
                      }
num_classes         = len(classes)
classes_fashion     = ['T-shirt/top','Trouser','Pullover','Dress','Coat',
                       'Sandal','Shirt','Sneaker','Bag','Ankle boot']


def create_model():
    '''
    Creates a sequential model
    '''

    cnn = keras.Sequential()
    cnn.add(InputLayer(input_shape=(img_height, img_width, channels)))

    # Normalization

    # Conv + Maxpooling
    # 28
    cnn.add(Conv2D(32, (3, 1), padding='same',kernel_initializer='he_normal'))
    cnn.add(BatchNormalization(axis=3))
    cnn.add(Activation('selu'))
    cnn.add(Conv2D(32, (1, 3), padding='same',kernel_initializer='he_normal'))
    cnn.add(Activation('selu'))


    cnn.add(Conv2D(32, (3, 1), padding='same',kernel_initializer='he_normal'))
    cnn.add(Activation('selu'))
    cnn.add(Conv2D(32, (1, 3), padding='same',kernel_initializer='he_normal'))
    cnn.add(Activation('selu'))

    cnn.add(Conv2D(32, (3, 1), padding='same',kernel_initializer='he_normal'))
    cnn.add(Activation('selu'))
    cnn.add(Conv2D(32, (1, 3), padding='same',kernel_initializer='he_normal'))
    cnn.add(Activation('selu'))

    cnn.add(MaxPool2D(pool_size=(2, 2)))
    # 14
    cnn.add(Conv2D(64, (3, 3), padding='same',kernel_initializer='he_normal'))
    cnn.add(BatchNormalization(axis=3))
    cnn.add(Activation('selu'))

    cnn.add(Conv2D(64, (3, 3), padding='same',kernel_initializer='he_normal'))
    cnn.add(Activation('selu'))

    cnn.add(MaxPool2D(pool_size=(2, 2)))

    # 7
    cnn.add(Conv2D(128, (7, 1), padding='same',kernel_initializer='he_normal'))
    cnn.add(BatchNormalization(axis=3))
    cnn.add(Activation('selu'))

    cnn.add(Conv2D(128, (1, 7), padding='same',kernel_initializer='he_normal'))
    cnn.add(Activation('selu'))

    #
    cnn.add(Conv2D(256, (1, 1), padding='same',kernel_initializer='he_normal'))
    cnn.add(Activation('selu'))
    cnn.add(Dropout(0.2))

    cnn.add(Conv2D(128, (1, 1), padding='same',kernel_initializer='he_normal'))
    cnn.add(Activation('selu'))

    # Converting 3D feature to 1D feature Vektor
    cnn.add(Flatten())

    # Fully Connected Layer
    cnn.add(Dense(64, activation='relu',kernel_initializer='he_normal'))
    # Dropout
    cnn.add(Dropout(0.2))
    # Fully Connected Layer
    cnn.add(Dense(64, activation='relu'))
    # Normalization
    cnn.add(BatchNormalization())

    cnn.add(Dense(num_classes, activation='softmax'))
    cnn.compile(loss='categorical_crossentropy',
                optimizer=optimizers.Adam(),
                metrics=['accuracy'])

    return cnn


create_model().summary()

# Defines the options for augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    horizontal_flip=True,
    fill_mode='nearest'
)

def image_augmentation(image):
    '''
    Generates new images bei augmentation
    image : raw image reading in by cv2.imread()
    images: array with new images
    '''
    images = []
    image = image.reshape(1, img_height, img_width, channels)
    i = 0
    for x_batch in datagen.flow(image, batch_size=1):
        images.append(x_batch)
        i += 1
        if i >= number_of_augmentation:
            break
    return images

def load_data(path, use_augmentation=True):
    X = []
    y = []
    print('-- Reading path: {} --'.format(path))
    for j in range(num_classes):
        print('  Load folder {}...'.format(j))
        sub_path = os.path.join(path, str(j), '*g')
        files = glob.glob(sub_path)
        for image in files:

            if channels == 1:
                img = cv2.imread(image, 0)
            else:
                img = cv2.imread(image)

            # DATA AUGMENTATION
            if use_augmentation:
                argu_img = image_augmentation(img)
                for a in argu_img:
                    X.append(a.reshape(img_height, img_width))
                    y.append(j)

            X.append(img)
            y.append(j)
    print('*Reading complete: %i samples\n' % len(X))
    return X, y


train_data, train_target = mnist_reader.load_mnist('C:/Users/korea/Desktop/fashion-mnist-master', kind='train')
test_data, test_target   = mnist_reader.load_mnist('C:/Users/korea/Desktop/fashion-mnist-master', kind='t10k')

def shaping(data, target):
    data = np.array(data, dtype=np.uint8)
    target = np.array(target, dtype=np.uint8)
    data = data.reshape(data.shape[0], img_height, img_width, channels)
    target = np_utils.to_categorical(target, num_classes)
    data = data.astype('float32')
    return data, target


train_data_shaped, train_target_shaped  = shaping(train_data, train_target)
print(train_target)
print(train_target_shaped)
test_data_shaped, test_target_shaped    = shaping(test_data, test_target)
histories = []

for i in range(0, iterations):
    print('Running iteration: %i' % i)

    # Saving the best checkpoint for each iteration
    filepath = "C:/Users/korea/Desktop/fashion-mnist-master/log/fashion_mnist3-%i.hdf5" % i
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1,
                                 save_best_only=True, mode='min')

    X_train, X_val, y_train, y_val = train_test_split(train_data_shaped, train_target_shaped,
                                                      test_size=0.2, random_state=42)

    cnn = create_model()
    history = cnn.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        epochs=num_epochs,
        verbose=1,
        validation_data=(X_val, y_val),
        callbacks=[
            # early_stopping,
            checkpoint
        ]
    )

    histories.append(history.history)

with open('C:/Users/korea/Desktop/fashion-mnist-master/log/fashion_mnist3-history.pkl', 'wb') as f:
    pickle.dump(histories, f)

histories = pickle.load(open('C:/Users/korea/Desktop/fashion-mnist-master/log/fashion_mnist3-history.pkl', 'rb'))


def get_avg(histories, his_key):
    tmp = []
    for history in histories:
        tmp.append(history[his_key][np.argmin(history['val_loss'])])
    return np.mean(tmp)


print('Training: \t%0.8f loss / %0.8f acc' % (get_avg(histories, 'loss'),
                                              get_avg(histories, 'acc')))
print('Validation: \t%0.8f loss / %0.8f acc' % (get_avg(histories, 'val_loss'),
                                                get_avg(histories, 'val_acc')))
test_loss = []
test_accs = []

for i in range(0, iterations):
    cnn_ = create_model()
    cnn_.load_weights("C:/Users/korea/Desktop/fashion-mnist-master/log/fashion_mnist3-%i.hdf5" % i)

    score = cnn_.evaluate(test_data_shaped, test_target_shaped, verbose=0)
    test_loss.append(score[0])
    test_accs.append(score[1])

    print('Running final test with model %i: %0.4f loss / %0.4f acc' % (i, score[0], score[1]))

print('\nAverage loss / accuracy on testset: %0.4f loss / %0.5f acc' % (np.mean(test_loss),
                                                                        np.mean(test_accs)))
print('Standard deviation: (+-%0.4f) loss / (+-%0.4f) acc' % (np.std(test_loss), np.std(test_accs)))


def plot_acc_loss(title, histories, key_acc, key_loss):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    # Accuracy
    ax1.set_title('Model accuracy (%s)' % title)
    names = []
    for i, model in enumerate(histories):
        ax1.plot(model[key_acc])
        ax1.set_xlabel('epoch')
        names.append('Model %i' % i)
        ax1.set_ylabel('accuracy')
    ax1.legend(names, loc='upper left')
    # Loss
    ax2.set_title('Model loss (%s)' % title)
    for model in histories:
        ax2.plot(model[key_loss])
        ax2.set_xlabel('epoch')
        ax2.set_ylabel('loss')
    ax2.legend(names, loc='upper right')
    fig.set_size_inches(20, 5)
    save_fig(title)
    plt.show()


plot_acc_loss('training', histories, 'acc', 'loss')
plot_acc_loss('validation', histories, 'val_acc', 'val_loss')


# Evaluation for one model

RUN = 0 # you can choose one of the different models trained above
model = create_model()
model.load_weights("C:/Users/korea/Desktop/fashion-mnist-master/log/fashion_mnist3-%i.hdf5" % RUN)

predictions = model.predict_classes(test_data_shaped, verbose=0)

def plot_train_val(title, history):
    fig, (ax1, ax2) = plt.subplots(1, 2)

    # Accuracy
    ax1.set_title('Model accuracy - %s' % title)
    ax1.plot(history['acc'])
    ax1.plot(history['val_acc'])
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('accuracy')
    ax1.legend(['train', 'validation'], loc='upper left')

    # Loss
    ax2.set_title('Model loss - %s' % title)
    ax2.plot(history['loss'])
    ax2.plot(history['val_loss'])
    ax2.set_xlabel('epoch')
    ax2.set_ylabel('loss')
    ax2.legend(['train', 'validation'], loc='upper left')

    fig.set_size_inches(20, 5)
    save_fig(title)
    plt.show()

plot_train_val('Model %i' % RUN, histories[RUN])

def plot_confusion_matrix(cm,class_,title='Confusion matrix',cmap=plt.cm.Reds):
    """
    This function plots a confusion matrix
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    tick_marks = np.arange(len(class_))
    plt.xticks(tick_marks, class_, rotation=90)
    plt.yticks(tick_marks, class_)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True labels')
    plt.xlabel('Predicted labels')
    save_fig(title)
    plt.show()

plot_confusion_matrix(confusion_matrix(test_target, predictions), classes_fashion)
#
# cnn.add(Conv2D(32, (3, 3), padding='same'))
# cnn.add(BatchNormalization(axis=1))
# cnn.add(Activation('selu'))
#
# cnn.add(Conv2D(32, (3, 3), padding='same'))
# cnn.add(Activation('selu'))
#
# cnn.add(MaxPool2D(pool_size=(2, 2)))
# # 14
# cnn.add(Conv2D(64, (3, 3), padding='same'))
# cnn.add(BatchNormalization(axis=1))
# cnn.add(Activation('selu'))
#
# cnn.add(Conv2D(64, (3, 3), padding='same'))
# cnn.add(Activation('selu'))
#
# cnn.add(MaxPool2D(pool_size=(2, 2)))
#
# # 7
# cnn.add(Conv2D(128, (7, 7), padding='same'))
# cnn.add(Activation('selu'))
#
# cnn.add(Conv2D(256, (1, 1), padding='same'))
# cnn.add(Activation('selu'))
#
# # Converting 3D feature to 1D feature Vektor
# cnn.add(Flatten())
#
# # Fully Connected Layer
# cnn.add(Dense(256, activation='relu'))
# # Dropout
# cnn.add(Dropout(0.5))
# # Fully Connected Layer
# cnn.add(Dense(64, activation='relu'))
# # Normalization
# cnn.add(BatchNormalization())
#
# cnn.add(Dense(num_classes, activation='softmax'))
# cnn.compile(loss='categorical_crossentropy',
#             optimizer=optimizers.Adam(),
#             metrics=['accuracy'])
#
# return cnn
