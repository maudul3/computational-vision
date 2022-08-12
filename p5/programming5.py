from keras.datasets import fashion_mnist
import keras
import numpy as np
import cv2
import matplotlib.pyplot as plt

standard_covariance = [
   [ 1 if i == j else 0 for i in range(100) ] 
   for j in range(100)
]

BATCH_SIZE = 500

def return_generator():
    input = keras.Input(shape=(1, 100))
    x = keras.layers.Dense(1000)(input)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dense(1000)(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dense(1000)(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.BatchNormalization()(x) 
    x = keras.layers.Dense(784)(x) 
    generator = keras.layers.Reshape((28,28))(x)

    return keras.models.Model(input, generator)

def return_discriminator():
    input = keras.Input(shape=(28, 28))
    x = keras.layers.Flatten()(input)
    x = keras.layers.Dense(1000)(input)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dense(1000)(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dense(1000)(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.BatchNormalization()(x) 
    discriminator = keras.layers.Dense(1, activation='sigmoid')(x) 

    return keras.models.Model(input, discriminator) 

if __name__ == '__main__':
    # Step 1
    generator = return_generator()
    discriminator = return_discriminator()
    
    full_model = keras.models.Sequential()
    full_model.add(generator)
    full_model.add(discriminator)

    discriminator.trainable = False

    # Step 2
    '''Load and preprocess data'''
    train_data, test_data = fashion_mnist.load_data()
    train_images, train_labels = train_data
    test_images, test_labels = test_data

    '''Normalize intensity values'''
    train_images = np.array([np.reshape(image / 255, (28, 28)) for image in train_images])
    test_images = np.array([np.reshape(image / 255, (28, 28)) for image in test_images])

    print (1)
    training_noise = np.array(
        [np.random.multivariate_normal(
            mean=[ 0 for _ in range(100) ],
            cov=standard_covariance,
            size=None
            )
        for _ in range(len(train_images))]
    )

    for epoch in range(10):
        print (1)