from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras import layers, models
from matplotlib import pyplot
from pathlib import Path
import cv2
import numpy as np

data_path = Path(__file__).parent.absolute() / Path('cats_dogs_dataset/dog vs cat/dataset')

def load_images_and_resize(paths):
    '''Loads and resizes images to '''
    images = []
    for path in paths:
        images.append(cv2.resize(cv2.imread(str(path)), (150,150)))
    return np.array(images)

if __name__ == "__main__":

    ''' part 1'''
    pre_model = InceptionResNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(150,150,3)
    )

    filters = pre_model.layers[1].get_weights()[0]

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
    # pyplot.show()

    ''' Part 2 '''
    # Find relevant paths for all of the images
    cat_training_paths = list((data_path / Path('training_set/cats')).rglob('*.jpg'))
    dog_training_paths = list((data_path / Path('training_set/dogs')).rglob('*.jpg'))
    cat_test_paths = list((data_path / Path('test_set/cats')).rglob('*.jpg'))
    dog_test_paths = list((data_path / Path('test_set/dogs')).rglob('*.jpg'))

    # Load the training and test data for cats and dogs
    train_dogs = load_images_and_resize(dog_training_paths)
    train_cats = load_images_and_resize(cat_training_paths)
    test_dogs = load_images_and_resize(dog_test_paths)
    test_cats = load_images_and_resize(cat_test_paths)

    test_data = np.array([*test_dogs, *test_cats])
    test_labels = np.array([0 for _ in range(len(test_dogs))] + [1 for _ in range(len(test_cats))])
    ''' Part 3 '''
    # Set up transfer model
    model = models.Sequential()
    model.add(pre_model)
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    pre_model.trainable = False
    
    ''' Part 4 '''
    # i
    model.compile()
    results = model.evaluate(test_data, test_labels)
    print (1)
    # ii

    # iii