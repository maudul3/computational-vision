from keras.datasets import fashion_mnist
import keras
import numpy as np
import cv2
import matplotlib.pyplot as plt

standard_covariance = [
   [ 1 if i == j else 0 for i in range(100) ] 
   for j in range(100)
]

BATCH_SIZE = 100

def return_generator():
    input = keras.Input(shape=(1, 100))
    x = keras.layers.Dense(600)(input)
    x = keras.layers.LeakyReLU(0.1)(x)
    x = keras.layers.BatchNormalization(momentum=0.8)(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(300)(x)
    x = keras.layers.LeakyReLU(0.1)(x)
    x = keras.layers.BatchNormalization(momentum=0.8)(x)
    x = keras.layers.Dense(150)(x)
    x = keras.layers.LeakyReLU(0.1)(x)
    x = keras.layers.BatchNormalization(momentum=0.8)(x)  
    x = keras.layers.Dense(784)(x) 
    generator = keras.layers.Reshape((28,28))(x)

    return keras.models.Model(input, generator)

def return_discriminator():
    input = keras.Input(shape=(28, 28))
    x = keras.layers.Flatten()(input)
    x = keras.layers.Dense(300)(x)
    x = keras.layers.LeakyReLU(0.1)(x)
    x = keras.layers.Dropout(0.8)(x)
    x = keras.layers.Dense(150)(x)
    x = keras.layers.LeakyReLU(0.1)(x)
    x = keras.layers.Dropout(0.8)(x)
    x = keras.layers.Dense(150)(x)
    x = keras.layers.LeakyReLU(0.1)(x)
    x = keras.layers.Dropout(0.8)(x)
    x = keras.layers.Dense(80)(x)
    x = keras.layers.LeakyReLU(0.1)(x)
    x = keras.layers.Dropout(0.8)(x)
    discriminator = keras.layers.Dense(1, activation='sigmoid')(x) 

    return keras.models.Model(input, discriminator) 

if __name__ == '__main__':
    # Step 1
    generator = return_generator()
    discriminator = return_discriminator()
    
    full_model = keras.models.Sequential()
    full_model.add(generator)
    full_model.add(discriminator)

    # Step 2
    '''Load and preprocess data'''
    train_data, test_data = fashion_mnist.load_data()
    #train_images, train_labels = train_data
    train_images, train_labels = test_data

    '''Normalize intensity values'''
    train_images = np.array([np.reshape(2*(image / 255) - 1, (28, 28)) for image in train_images])
    #test_images = np.array([np.reshape(image / 255, (28, 28))*2 - 1 for image in test_images])

    discriminator.compile(optimizer=keras.optimizers.adam_v2.Adam(learning_rate=1e-3), loss='binary_crossentropy')
    discriminator.trainable = False
    full_model.compile(optimizer=keras.optimizers.adam_v2.Adam(learning_rate=1e-3), loss='binary_crossentropy')

    for epoch in range(100):
        print ("EPOCH: ", epoch )
        for batch in range(int(len(train_images) / BATCH_SIZE)):
            print ("Batch {}: {} - {}".format(batch, batch*BATCH_SIZE, BATCH_SIZE*(batch+1)))
            noise_batch = np.random.normal(size=[BATCH_SIZE, 1, 100])
            noise_image_batch = generator.predict(noise_batch)
            train_batch = train_images[batch*BATCH_SIZE:(batch*BATCH_SIZE + BATCH_SIZE)]
            
            # Combine datasets
            total_batch = np.array([*noise_image_batch, *train_batch])
            # 0 = fake image, 1 = real image
            true_labels = np.array([*[0.1 for _ in range(BATCH_SIZE) ], *[0.9 for _ in range(BATCH_SIZE)]])
            temp = list(zip(total_batch, true_labels))
            np.random.shuffle(temp)
            total_batch = np.array([row[0] for row in temp])
            true_labels = np.array([row[1] for row in temp])
            
            discriminator.trainable = True

            disc_loss = discriminator.train_on_batch(total_batch, true_labels)

            discriminator.trainable = False

            noise_batch = np.random.normal(size=[BATCH_SIZE, 1, 100])

            '''np.array(
                [np.random.multivariate_normal(
                mean=[ 0 for _ in range(100) ],
                cov=standard_covariance,
                size=1
                )
            for _ in range(BATCH_SIZE)]
            )'''

            loss = full_model.train_on_batch(noise_batch, np.array([ 0.9 for _ in range(BATCH_SIZE)]))
            
            print ("Discriminator Loss: {}".format(disc_loss))
            print ("Full Loss: {}".format(loss))
        
        if epoch % 10 == 0:
            fake_images_sample = generator.predict(
                np.random.normal(size=[10, 1, 100])
            )
            
            for idx in range(len(fake_images_sample)):
                cv2.imwrite('epoch{}fakeimage{}.jpeg'.format(epoch, idx), (fake_images_sample[idx] + 1) * 255 / 2)