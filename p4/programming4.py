from random import random
from keras.datasets import fashion_mnist
import keras
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.util import random_noise
from sklearn.decomposition import PCA

def return_autoencoder():
    '''Generates the autoencoder model'''

    # Encoder
    input_img = keras.Input(shape=(28, 28, 1))
    x = keras.layers.Conv2D(32, (3,3), activation='relu', padding='same')(input_img)
    x = keras.layers.MaxPooling2D((2,2), padding='same')(x)
    x = keras.layers.Conv2D(16, (3,3), activation='relu', padding='same')(x)
    x = keras.layers.MaxPooling2D((2,2), padding='same')(x)
    x = keras.layers.Flatten()(x)
    encoder = keras.layers.Dense(2, activation='relu')(x) 

    # Decoder
    x = keras.layers.Dense(784, activation='relu')(encoder)
    x = keras.layers.Reshape((7,7,16))(x)
    x = keras.layers.UpSampling2D((2,2))(x)
    x = keras.layers.Conv2D(16, (3,3), activation='relu', padding='same')(x)
    x = keras.layers.UpSampling2D((2,2))(x)
    x = keras.layers.Conv2D(32, (3,3), activation='relu', padding='same')(x)
    decoder = keras.layers.Conv2D(1, (3,3), activation='sigmoid',padding='same')(x)

    return keras.Model(input_img, decoder)

if __name__ == '__main__':

    '''Load and preprocess data'''
    train_data, test_data = fashion_mnist.load_data()
    train_images, train_labels = train_data
    test_images, test_labels = test_data

    '''Normalize intensity values'''
    train_images = np.array([np.reshape(image / 255, (28, 28, 1)) for image in train_images])
    test_images = np.array([np.reshape(image / 255, (28, 28, 1)) for image in test_images])
    
    # Match label indices to string values
    labeldict = {
        0: 'T-shirt',
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
    
    # Step 1: Design Autoencoder
    autoencoder = return_autoencoder()
    autoencoder.summary()

    # Step 2: Train and Predict with Autoencoder
    autoencoder.compile(
        optimizer='adam', loss='binary_crossentropy'
    )
    autoencoder.fit(
        train_images, train_images, 
        epochs=10, 
        validation_data=(test_images, test_images)
    )
    
    # Evaluate autoencoder predictions
    predictions = autoencoder.predict(test_images)
    for idx, test_val in enumerate(test_images):
        cv2.imwrite('sample{}.jpeg'.format(idx), test_val * 255)
        cv2.imwrite('sample{}pred.jpeg'.format(idx), predictions[idx] * 255)
        if idx > 5:
            break

    # Step 3: Add noise to datasets
    train_images_noisy = np.array([random_noise(image, mode='gaussian') for image in train_images])
    test_images_noisy = np.array([random_noise(image, mode='gaussian') for image in test_images])

    # Retrain nosiy autoencoder
    autoencoder_noisy = return_autoencoder()
    autoencoder_noisy.compile(
        optimizer='adam', loss='binary_crossentropy'
    )
    autoencoder_noisy.fit(
        train_images_noisy, train_images, 
        epochs=10, 
        validation_data=(test_images_noisy, test_images)
    )

    # Save test images with noisy autoencoder
    predictions = autoencoder_noisy.predict(test_images_noisy)
    for idx, test_val in enumerate(test_images_noisy):
        cv2.imwrite('sample{}noisy.jpeg'.format(idx), test_val * 255)
        cv2.imwrite('sample{}noisypred.jpeg'.format(idx), predictions[idx] * 255)
        if idx > 5:
            break
    
    # Step 4: Output 2D latent feature and plot results
    encoder = keras.Model(autoencoder.input, autoencoder.get_layer('dense').output)
    coordinates = encoder.predict(test_images)
    for label in set(test_labels):
        to_plot_x = []
        to_plot_y = []
        for coordinate, sample_label in zip(coordinates, test_labels):
            if sample_label == label:
                to_plot_x.append(coordinate[0])
                to_plot_y.append(coordinate[1])
        plt.scatter(to_plot_x, to_plot_y, label=labeldict[label])
    plt.legend()
    plt.title('CAE')
    plt.show()
    plt.clf()
    
    # Plot PCA results for comparison 
    pca = PCA(n_components=2)
    pca_components = []
    for image in test_images:
        pca_components.append(pca.fit_transform(np.reshape(image, (28,28))))
    
    for label in set(test_labels):
        to_plot_x = []
        to_plot_y = []
        for coordinate, sample_label in zip(pca_components, test_labels):
            if sample_label == label:
                to_plot_x.append(coordinate[0])
                to_plot_y.append(coordinate[1])
        plt.scatter(to_plot_x, to_plot_y, label=labeldict[label])
    plt.legend()
    plt.title('PCA')
    plt.show()