from __future__ import print_function
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras import layers, models, losses, optimizers
from matplotlib import pyplot
from pathlib import Path
from functools import partial
import cv2
import numpy as np

data_path = Path(__file__).parent.absolute() / Path('cats_dogs_dataset/dog vs cat/dataset')

def save_output(x, file_name='file'):
    with open( Path(__file__).parent.absolute() / Path(file_name), 'a') as f:
        f.write(x)

def load_images(paths):
    ''' Loads images, normalizes pixel values to 0 - 1,
    and resizes them to 150 x 150 x 3 '''
    images = []
    for path in paths:
        images.append(cv2.resize(cv2.imread(str(path)) / 255., (150,150)))
    return np.array(images)

def plot_filters(model):
    """ Plots the filters for a model"""
    filters = model.layers[1].get_weights()[0]

    # Normalize filter values to 0 and 1
    f_min, f_max = filters.min(), filters.max()
    filters = (filters - f_min) / (f_max - f_min)

    # plot the 32 filters
    n_filters, ix = 32, 1
    for i in range(n_filters):
        # get the filter
        f = filters[:, :, :, i]
        # plot each channel separately
        # specify subplot and turn of axis
        ax = pyplot.subplot(8, 4, ix)
        ax.set_xticks([])
        ax.set_yticks([])
        pyplot.imshow(f[:, :, :])
        ix += 1
    # show the figure
    pyplot.show()

def confusion_matrix(true_labels, predicted_labels):
    """ Prints each component of the confusion matrix
    Assumes dog == 0 and cat == 1
    """
    truecat = fakecat = truedog = fakedog = 0
    for tested, predicted in zip(true_labels, predicted_labels):
        if tested == predicted and tested == 0:
            truedog += 1
        elif tested == predicted and tested == 1: 
            truecat += 1
        elif predicted == 0:
            fakedog += 1
        else:
            fakecat += 1

    print ( "Accuracy: ",
     (truedog + truecat) / (truedog + truecat + fakedog + fakecat)
    )
    print ( "True dog: ", (truedog) / (truedog + truecat + fakedog + fakecat) )
    print ( "True cat: ", (truecat) / (truedog + truecat + fakedog + fakecat) )
    print ( "Fake dog: ", (fakedog) / (truedog + truecat + fakedog + fakecat) )
    print ( "Fake cat: ", (fakecat) / (truedog + truecat + fakedog + fakecat) )

if __name__ == "__main__":

    ''' part 1'''
    pre_model = InceptionResNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(150,150,3)
    )
    
    #plot_filters(pre_model)
    write_pre = partial(save_output, file_name='premodel')
    pre_model.summary(
        print_fn=write_pre
    )

    ''' Part 2 '''
    # Find relevant paths for all of the images
    cat_training_paths = list((data_path / Path('training_set/cats')).rglob('*.jpg'))
    dog_training_paths = list((data_path / Path('training_set/dogs')).rglob('*.jpg'))
    cat_test_paths = list((data_path / Path('test_set/cats')).rglob('*.jpg'))
    dog_test_paths = list((data_path / Path('test_set/dogs')).rglob('*.jpg'))

    # Load the training and test data for cats and dogs
    train_dogs = load_images(dog_training_paths)
    train_cats = load_images(cat_training_paths)
    test_dogs = load_images(dog_test_paths)
    test_cats = load_images(cat_test_paths)

    # Combine dog and cat data into train and test datasets
    test_data = np.array([*test_dogs, *test_cats])
    test_labels = np.array([0 for _ in range(len(test_dogs))] + [1 for _ in range(len(test_cats))])
    
    train_data = np.array([*train_dogs, *train_cats])
    train_labels = np.array([0 for _ in range(len(train_dogs))] + [1 for _ in range(len(train_cats))])
    
    ''' Part 3 '''
    # Set up transfer model
    model = models.Sequential()
    model.add(pre_model)
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    pre_model.trainable = False

    write_mod = partial(save_output, file_name='model_sum')
    model.summary(
        print_fn=write_mod
    )
    
    ''' Part 4 '''
    # i
    model.compile()
    predictions = [round(row[0]) for row in model.predict(test_data)]
    confusion_matrix(test_labels, predictions)
    
    # ii
    
    model.compile(
        optimizer=optimizers.adam_v2.Adam(learning_rate=1e-3),
        loss=losses.BinaryCrossentropy()
    )
    model.fit(train_data, train_labels, epochs=8, validation_data=(test_data, test_labels))
    predictions = [round(row[0]) for row in model.predict(test_data)]
    confusion_matrix(test_labels, predictions)

    print ("Part 4 iii (i)")
    pre_model_sub = models.Model(inputs=pre_model.input, outputs=pre_model.layers[99].output)
    write_pre = partial(save_output, file_name='premodel_sub')
    pre_model_sub.summary(
        print_fn=write_pre
    )

    model = models.Sequential()
    model.add(pre_model_sub)
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    pre_model_sub.trainable = False

    model.compile()
    truecat = fakecat = truedog = fakedog = 0
    predictions = [round(row[0]) for row in model.predict(test_data)]
    confusion_matrix(test_labels, predictions)

    print ("Part 4 iii (ii)")
    model.compile(
        optimizer=optimizers.adam_v2.Adam(learning_rate=1e-3),
        loss=losses.BinaryCrossentropy()
    )
    model.fit(train_data, train_labels, epochs=8, validation_data=(test_data, test_labels))
    predictions = [round(row[0]) for row in model.predict(test_data)]
    confusion_matrix(test_labels, predictions) 