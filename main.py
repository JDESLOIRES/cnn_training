# do python debgging => deployment => mapping /tmp/pycharm_project_921
import os
import numpy
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import psutil
# memory = psutil.virtual_memory().total / (1024.0 ** 3)

from sklearn.model_selection import train_test_split

import visualization
import load_data
import model
import training

# https://www.tensorflow.org/install/gpu#software_requirements

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.15
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

#################################################################################


class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

x_train, y_train = load_data.load_mnist('./data', kind='train')
x_test, y_test = load_data.load_mnist('./data', kind='t10k')

x_train, x_valid, y_train, y_valid = train_test_split(x_train,
                                                      y_train,
                                                      test_size=0.1)

y_train = tf.keras.utils.to_categorical(y_train, 10)
y_valid = tf.keras.utils.to_categorical(y_valid, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)



for i in range(9):
    print(i)
    # define subplot
    plt.subplot(330 + 1 + i)
    # plot raw pixel data
    plt.imshow(x_train[i].squeeze(), cmap=plt.get_cmap('gray'))
# show the figure
plt.show()

training = training.train_classifier(x_train, y_train,
                                     x_valid, y_valid,
                                     n_batch=64,
                                     val_size=0.1)

##################################################################

mlp = model.MyMLP()
training.fit_model(mlp,
                   optimizer=tf.keras.optimizers.Adam(learning_rate=10e-3),
                   loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                   nb_epoch=1,
                   filepath='/home/s999379/model_weights_MLP/',
                   filename='my_best_mlp_.h5'
                   )


##################################################################################

cnn = model.MyCNN()
filepath='/home/s999379/model_weights_CNN_3/'
filename='my_best_cnn_.h5'

training.fit_model(cnn,
                   optimizer=tf.keras.optimizers.Adam(learning_rate=10e-4),
                   loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                   nb_epoch=1,
                   filepath=filepath,
                   filename=filename
                   )

batch_x, batch_y = training.get_batch(1, x_train), training.get_batch(1, y_train)

#Take the best model
best_model = model.MyCNN()
input = tf.keras.Input(shape=(28,28,1))
best_model.call(input)
best_model.load_weights(filepath + filename)
_, flatten = best_model.call(batch_x,training = False)

#Plot embedding
visualization.embedding_tsne(flatten,batch_y)
#Visualize a layer
maping = visualization.map_layer(best_model,input_shape = (28,28,1))
conv_2 = maping.get_extractor('conv_2')
maping.plot_layer(conv_2,batch_x)





#################################################################################################
lenet = model.LeNet()

training.fit_model(lenet,
                   optimizer=tf.keras.optimizers.Adam(learning_rate=10e-4),
                   loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                   nb_epoch=1,
                   filepath='/home/s999379/model_weights_CNN/')
