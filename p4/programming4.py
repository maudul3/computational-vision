from keras.datasets import fashion_mnist
import keras
import numpy as np
import cv2

def return_autoencoder():
    '''Generates the autoencoder model'''

    # Encoder
    input_img = keras.Input(shape=(28, 28, 1))
    x = keras.layers.Conv2D(32, (3,3), activation='relu', padding='same')(input_img)
    x = keras.layers.MaxPooling2D((2,2), padding='same')(x)
    x = keras.layers.Conv2D(16, (3,3), activation='relu', padding='same')(x)
    x = keras.layers.MaxPooling2D((2,2), padding='same')(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(2, activation='relu')(x) 

    # Decoder
    x = keras.layers.Dense(784, activation='relu')(x)
    x = keras.layers.Reshape((7,7,16))(x)
    x = keras.layers.UpSampling2D((2,2))(x)
    x = keras.layers.Conv2D(16, (3,3), activation='relu', padding='same')(x)
    x = keras.layers.UpSampling2D((2,2))(x)
    x = keras.layers.Conv2D(32, (3,3), activation='relu', padding='same')(x)
    x = keras.layers.Conv2D(1, (3,3), activation='sigmoid',padding='same')(x)

    return keras.Model(input_img, x)

if __name__ == '__main__':

    '''Load and preprocess data'''
    train_data, test_data = fashion_mnist.load_data()
    train_images, train_labels = train_data
    test_images, test_labels = test_data

    '''Normalize intensity values'''
    train_images = np.array([np.reshape(image / 255, (28, 28, 1)) for image in train_images])
    test_images = np.array([np.reshape(image / 255, (28, 28, 1)) for image in test_images])

    autoencoder = return_autoencoder()
    autoencoder.compile(
        optimizer='adam', loss='binary_crossentropy'
    )
    autoencoder.fit(
        train_images, train_images, 
        epochs=5, 
        validation_data=(test_images, test_images)
    )
    predictions = autoencoder.predict(test_images)
    for idx, test_val in enumerate(test_images):
        cv2.imwrite('sample{}.jpeg'.format(idx), test_val * 255)
        cv2.imwrite('sample{}pred.jpeg'.format(idx), predictions[idx] * 255)
        if idx > 5:
            break