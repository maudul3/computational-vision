'''
CONVOLUTIONAL
'''
from keras.datasets import fashion_mnist
import keras
import numpy as np
import cv2
import matplotlib.pyplot as plt

BATCH_SIZE = 100

def return_generator():
    '''Function to return the generator model'''
    input = keras.Input(shape=(1, 100))
    x = keras.layers.Dense(7 * 7 * 128)(input)
    x = keras.layers.LeakyReLU(0.2)(x)
    x = keras.layers.Reshape((7, 7, 128))(x)
    x = keras.layers.BatchNormalization(momentum=0.8)(x)
    x = keras.layers.Conv2DTranspose(128, kernel_size =5, strides=2, padding="same")(x)
    x = keras.layers.LeakyReLU(0.1)(x)
    x = keras.layers.BatchNormalization(momentum=0.8)(x)
    x = keras.layers.Conv2DTranspose(1, kernel_size=5, strides=2, padding="same")(x)
    x = keras.layers.LeakyReLU(0.1)(x)
    generator = keras.layers.BatchNormalization(momentum=0.8)(x)  

    return keras.models.Model(input, generator)

def return_discriminator():
    '''Function tor return the discriminator model'''
    input = keras.Input(shape=(28, 28, 1))
    x = keras.layers.Conv2D(128, kernel_size=5, strides=2, padding="same")(input)
    x = keras.layers.LeakyReLU(0.1)(x)
    x = keras.layers.Dropout(0.8)(x)
    x = keras.layers.Conv2D(128, kernel_size=5, strides=2, padding="same")(x)
    x = keras.layers.LeakyReLU(0.1)(x)
    x = keras.layers.Dropout(0.8)(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(100)(x)
    x = keras.layers.LeakyReLU(0.1)(x)
    discriminator = keras.layers.Dense(1, activation='sigmoid')(x) 

    return keras.models.Model(input, discriminator) 

if __name__ == '__main__':
    # Step 1: Load generator and discriminator into full mode
    generator = return_generator()
    discriminator = return_discriminator()
    
    full_model = keras.models.Sequential()
    full_model.add(generator)
    full_model.add(discriminator)

    # Step 2
    '''Load and preprocess data'''
    _, train_data = fashion_mnist.load_data()
    train_images, train_labels = train_data

    '''Normalize intensity values'''
    train_images = np.array([np.reshape(2*(image / 255) - 1, (28, 28, 1)) for image in train_images])

    '''Compile the discriminator and full model'''
    discriminator.compile(optimizer=keras.optimizers.adam_v2.Adam(learning_rate=1e-3), loss='binary_crossentropy')
    discriminator.trainable = False
    full_model.compile(optimizer=keras.optimizers.adam_v2.Adam(learning_rate=1e-3), loss='binary_crossentropy')

    # Initial variables for loss arrays (for plotting) and loss variables for scalar loss 
    disc_total_loss = []
    gen_total_loss = []
    disc_loss = None
    gen_loss = None

    for epoch in range(100):
        print ("EPOCH: ", epoch )
        for batch in range(int(len(train_images) / BATCH_SIZE)):
            # Create batches
            print ("Batch {}: {} - {}".format(batch, batch*BATCH_SIZE, BATCH_SIZE*(batch+1)))
            noise_batch = np.random.normal(size=[BATCH_SIZE, 1, 100])
            noise_image_batch = generator.predict(noise_batch)
            train_batch = train_images[batch*BATCH_SIZE:(batch*BATCH_SIZE + BATCH_SIZE)]
            
            # Combine datasets and randomize order
            total_batch = np.array([*noise_image_batch, *train_batch])
            # 0 = fake image, 1 = real image
            true_labels = np.array([*[0.1 for _ in range(BATCH_SIZE) ], *[0.9 for _ in range(BATCH_SIZE)]])
            temp = list(zip(total_batch, true_labels))
            np.random.shuffle(temp)
            total_batch = np.array([row[0] for row in temp])
            true_labels = np.array([row[1] for row in temp])
            
            # Discriminator training
            discriminator.trainable = True

            disc_loss = discriminator.train_on_batch(total_batch, true_labels)

            discriminator.trainable = False

            noise_batch = np.random.normal(size=[BATCH_SIZE, 1, 100])

            # train generator
            gen_loss = full_model.train_on_batch(noise_batch, np.array([ 0.9 for _ in range(BATCH_SIZE)]))
            
            print ("Discriminator Loss: {}".format(disc_loss))
            print ("Full Loss: {}".format(gen_loss))

        # Add to loss lists for later plotting 
        disc_total_loss.append(disc_loss)
        gen_total_loss.append(gen_loss)
        
        # Save sample images 
        if epoch == 0 or (epoch + 1) % 10 == 0:
            fake_images_sample = generator.predict(
                np.random.normal(size=[10, 1, 100])
            )
            
            for idx in range(len(fake_images_sample)):
                cv2.imwrite('cganepoch{}fakeimage{}.jpeg'.format(epoch+1, idx), (fake_images_sample[idx] + 1) * 255 / 2)
    
    plt.plot([i for i in range(len(gen_total_loss))], gen_total_loss, label="Generator")
    plt.plot([i for i in range(len(disc_total_loss))], disc_total_loss, label="Discriminator")
    plt.xlabel("Epoch")
    plt.ylabel("Loss") 
    plt.title ("CNN-based GAN Loss")
    plt.legend()
    plt.savefig("cgan_loss.jpeg")