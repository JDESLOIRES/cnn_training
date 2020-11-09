from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

class map_layer:
    def __init__(self,model,input_shape = (28,28,1)):
        self.model = model
        self.input = tf.keras.Input(shape=input_shape)
        self.model.call(self.input)

    def get_extractor(self,layer_name):

        layers = np.array([layer.name for layer in self.model.layers])
        position = np.where(layers == layer_name)[0] +1

        extractor = tf.keras.Model(
            inputs=self.input,
            outputs=[layer.output for layer in self.model.layers[:position[0]]],
        )
        print(extractor.layers[-1].name)

        return extractor

    def plot_layer(self,
                   successive_feature_maps,
                   X,
                   height=8,width=8):

        successive_feature_maps = successive_feature_maps.predict(X)

        ix = 1
        for _ in range(height):
            for _ in range(width):
                # specify subplot and turn of axis
                ax = plt.subplot(height, width, ix)
                ax.set_xticks([])
                ax.set_yticks([])
                # plot filter channel in grayscale
                plt.imshow(successive_feature_maps[-1][0, :, :, ix - 1], cmap='gray')
                ix += 1

        # show the figure
        plt.show()

    def embedding_tsne(self,successive_feature_maps,X,y):
        sns.set(rc={'figure.figsize': (11.7, 8.27)})
        palette = sns.color_palette("bright", 10)

        tsne = TSNE(n_components=2)

        flatten_value = successive_feature_maps.predict(X)
        predictions = flatten_value[-1]
        x_std = StandardScaler().fit_transform(predictions)

        train_test_2D = tsne.fit_transform(x_std)
        y = np.argmax(y, axis=1).astype(float)

        sns.scatterplot(train_test_2D[:, 0], train_test_2D[:, 1],
                        hue=y,
                        legend='full', palette=palette)

        plt.show()
