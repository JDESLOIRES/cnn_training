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
import load_data
import model
import training
import numpy as np
#tf.test.is_gpu_available()
# https://www.tensorflow.org/install/gpu#software_requirements

#################################################################################


class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

x_train, y_train = load_data.load_mnist('./data', kind='train')

x_train, x_valid, y_train, y_valid = train_test_split(x_train,
                                                      y_train,
                                                      test_size=0.1)

y_train = tf.keras.utils.to_categorical(y_train, 10)
y_valid = tf.keras.utils.to_categorical(y_valid, 10)


for i in range(9):
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


##################################################################################

cnn = model.MyCNN()

#cnn.compile( optimizer=tf.keras.optimizers.Adam(learning_rate=10e-4), loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False))
#cnn.fit(x_train,y_train,epochs = 1)
#cnn.evaluate(x_train,y_train)
#cnn.summary()
#cnn.get_layer('conv_2').output


training.fit_model(cnn,
                   optimizer=tf.keras.optimizers.Adam(learning_rate=10e-4),
                   loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                   nb_epoch=1,
                   filepath='/home/s999379/model_weights_CNN/',
                   filename = 'my_best_cnn'
                   )

#cnn.get_layer('conv_2').output
#earlyPredictor = cnn(cnn.inputs,best_cnn_model.get_layer('conv_2').output)


best_cnn_model = tf.keras.models.load_model('/home/s999379/model_weights_CNN/my_best_cnn',compile = True)
best_cnn_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=10e-4),
                       loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False))

#np.testing.assert_allclose(best_cnn_model(input_arr), outputs)

best_cnn_model.fit(x_train, y_train)
best_cnn_model.evaluate(x_train,y_train)
best_cnn_model.get_layer(name='conv_2').output

outputs = [cnn.get_layer(name=output.name).get_output_at(0) for output in cnn.layers]
pred_model = tf.keras.models.Model(cnn.input,outputs)

###########################
#####Small example
###########################
del cnn


training_ex = x_train[:10]
training_ex = training_ex.reshape(10,28,28,1)

#model = tf.keras.Model(inputs = training_ex, outputs = )


for layer in cnn.layers:
    print(layer.name)


cnn.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=10e-4),
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False))
cnn.fit(training_ex,y_train[:10])
cnn.evaluate(x_train[-100:],y_train[-100:])
cnn.predict(x_train[-5:])
cnn.get_layer("conv_2").get_output_at(0)
cnn.get_layer("conv_2").output_shape()


y_pred = cnn(training_ex,training=False)

outputs = [cnn.get_layer(name=output.name).get_output_at(0) for output in cnn.layers]


original_model = tf.keras.Model(x_train, y_train)
activation_outputs = [layer.output for layer in original_model.layers]
activation_model = tf.keras.Model(inputs, activation_outputs)

activation_outputs = [layer.output for layer in original_model.layers]
activation_model = tf.keras.Model(inputs, activation_outputs)


successive_outputs = [layer for layer in best_cnn_model.layers[1:]]


#visualization_model = Model(img_input, successive_outputs)
visualization_model = tf.keras.models.Model(inputs = model.input, outputs = successive_outputs)

# Let's prepare a random input image of a cat or dog from the training set.
#cat_img_files = [os.path.join(train_cats_dir, f) for f in train_cat_fnames]
#dog_img_files = [os.path.join(train_dogs_dir, f) for f in train_dog_fnames]

#img_path = random.choice(cat_img_files + dog_img_files)

img = load_img(img_path, target_size=(150, 150))  # this is a PIL image

x   = img_to_array(img)                           # Numpy array with shape (150, 150, 3)
x   = x.reshape((1,) + x.shape)                   # Numpy array with shape (1, 150, 150, 3)

# Rescale by 1/255
x /= 255.0

# Let's run our image through our network, thus obtaining all
# intermediate representations for this image.
successive_feature_maps = visualization_model.predict(x)
###################

assert np.allclose(cnn.predict(x_train), best_cnn_model.predict(x_train))

functional_model = model.MyCNN(inputs=x_train, outputs=y_train, name="3_layer_mlp")

pretrained = cnn(
    cnn.inputs, cnn.layers[-1].input, name="pretrained_model"
)

intermediate_layer_model = cnn(inputs=cnn.inputs,
                                          outputs=cnn.get_layer('conv_2').output)


intermediate_layer_model.summary()


model_summary()


def get_layer_outputs(self):
    """
    :return: Names of all the layers kept by Keras
    """
    # outputs = [self.get_layer(name=output.name).get_output_at(0) for output in self.layers]
    outputs = [layer.output for layer in self.layers]
    return outputs


def get_layer_names(self):
    """
    :return: Names of all the layers kept by Keras
    """
    layer_names = [x.name for x in self.layers]
    return layer_names


def get_specific_layer(self, feature_name):
    features_layer = tf.keras.models.Model(
        inputs=self.inputs,
        outputs=self.get_layer(name=feature_name).output,
    )
    return features_layer


######################################################################################################
cnn, *_ = training.train_classifier(cnn)



x = cnn.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)

lay = cnn.get_layer_names()
cnn.get_layer_outputs()
lay = cnn.get_layer('conv_1')
cnn.output

for layer in cnn.layers[:6]:
    layer.trainable =  False


# Un-freeze the top layers of the model
cnn.trainable = True
# Let's take a look to see how many layers are in the base model
print("Number of layers in the base model: ", len(cnn.layers))

# Fine tune from this layer onwards
fine_tune_at = 5

# Freeze all the layers before the `fine_tune_at` layer
for layer in cnn.layers[:5]:
    print(layer)
    layer.trainable =  False

cnn.predict(x_train[:10])


cnn.compile(loss="categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate=10e-4),
            metrics=["accuracy"])

H = cnn.fit_generator(
	training_data = (x_train, y_train),
	validation_data=(x_valid, y_valid),
	steps_per_epoch=x_train.shape[0] // 64,
	epochs=1,
	verbose=1)









model = tf.keras.Sequential([
    cnn,
    tf.keras.layers.GlobalAveragePooling2D(),

])

# Compile the model to configure training parameters
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.

inception_model = build_model()


cnn.get_a
weights = [layer.get_weights() for layer in cnn.layers]
len(weights)
weights[0][0].shape
for w in weights:
    print(w.shape)

outputs = cnn.call(inputs=x_train[:10])
model = tf.keras.Model(inputs=(10,28,28,1), outputs = outputs)


best_cnn_model = tf.keras.models.load_model('/home/s999379/model_weights_CNN/my_best_cnn')
best_cnn_model.inputs

best_cnn_model.predict(x_valid)
best_cnn_model.get_layer('conv_1').output
cnn.get_layer('conv_2').output
outputs = [best_cnn_model.get_layer(name=output.name).get_output_at(0) for output in best_cnn_model.layers]

input = x_valid[:10]
final_output = best_cnn_model(input)

features_layer1 = tf.keras.models.Model(
    inputs=best_cnn_model.inputs,
    outputs=best_cnn_model.get_layer(name="conv_2").output,
)


extractor = tf.keras.Model(
    inputs=cnn.inputs,
    outputs=[layer.output for layer in cnn.layers],
)

print(features_layer1)
print(input)
print(final_output)