
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

import visualization
import load_data
import training

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.15
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

#################################################################################


class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


x_test, y_test = load_data.load_mnist('./data', kind='t10k')
y_test = tf.keras.utils.to_categorical(y_test, 10)


##################################################################################

cnn = tf.keras.models.load_model('/home/s999379/model_weights_CNN/my_best_cnn2',compile=True)

cnn.trainable_variables


cnn.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=10e-4),
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False))
#batch_test, batch_y_test = training.get_batch(1, x_test), training.get_batch(1, y_test)
tf.config.experimental_run_functions_eagerly(True)
input = tf.keras.Input(shape=(28,28,1))
cnn.call(input,training = False)

maping = visualization.map_layer(cnn,input_shape = (28,28,1))

conv_2 = maping.get_extractor('conv_2')
maping.plot_layer(conv_2,batch_x)

flatten = maping.get_extractor('flatten_8')
maping.embedding_tsne(flatten,batch_test,batch_y_test)


metrics = evaluate.evaluate_model(batch_test, batch_y_test, 64)
y_pred, test_fm, *_ = metrics.apply_predictions(cnn)


#autre fichier : main qui charge données test + modèle + calcul prédiction
#sauve weights + autre fichier : initialise un modèle avec les weights saved