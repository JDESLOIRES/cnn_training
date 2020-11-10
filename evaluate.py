import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import pandas as pd
import visualization
import load_data
import training
import model
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import random
import numpy as np

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 1
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

#################################################################################
#Path best model
filepath='/home/s999379/model_weights_CNN_3/'
filename='my_best_cnn_.h5'

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

x_test, y_test = load_data.load_mnist('./data', kind='t10k')
y_test = tf.keras.utils.to_categorical(y_test, 10)

##################################################################################


#Evaluate over 10 batch

def get_batch(i, array, nb_batch=64):
    start_id = i * nb_batch
    end_id = min((i + 1) * nb_batch, array.shape[0])
    batch_x = array[start_id:end_id]
    return batch_x

#Aggregate 10 random batch to get accuracy
batch_x_test = []
batch_y_test = []
nb_aggregated_random_batch = 2
my_randoms = random.sample(range(0,x_test.shape[0]//64), nb_aggregated_random_batch)
for i in my_randoms:
    print(i)
    batch_x_test.append(get_batch(i, x_test))
    batch_y_test.append(get_batch(i, y_test))

batch_x_test, batch_y_test = np.vstack(batch_x_test), np.vstack(batch_y_test)

#Evaluate
#Take the best model
best_model = model.MyCNN()
input = tf.keras.Input(shape=(28,28,1))
best_model.call(input)
best_model.load_weights(filepath + filename)
_, flatten = best_model.call(batch_x_test,training = False)


##Evaluate
metrics = training.evaluate_model(batch_x_test, batch_y_test)
y_pred, test_fm, *_ = metrics.apply_predictions(best_model)
print('Accuracy on those batch test :' + str(test_fm))

#TNSE
visualization.embedding_tsne(flatten,batch_y_test)



#Mapping predictions
maping = visualization.map_layer(best_model, input_shape=(28, 28, 1))
conv_2 = maping.get_extractor('conv_2')
maping.plot_layer(conv_2, batch_x_test)
