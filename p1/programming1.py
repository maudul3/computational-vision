import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

dir_path = Path(__file__).parent

def select_gaussian(padding):
    if padding == 1:
            filter_matrix = np.array([
                [1, 2, 1], 
                [2, 4, 2], 
                [1, 2, 1]
            ]) * (1/16)
    if padding == 2:
            filter_matrix = np.array([
                [1, 4, 7, 4, 1],
                [4, 16, 26, 16, 4],  
                [7, 26, 41, 26, 7],
                [4, 16, 26, 16, 4], 
                [1, 4, 7, 4, 1],
            ]) * (1/273)

    return filter_matrix
    

def filter(image, padding=1, filter_type='gaussian'):

    new_image = np.array(image)

    if filter_type == 'gaussian':
       filter_matrix = select_gaussian(padding)
    elif filter_type == 'dogx':
        filter_matrix = np.array([
                [1, 0, -1], 
                [2, 0, -2], 
                [1, 0, -1]
            ])
    elif filter_type == 'dogy':
        filter_matrix = np.array([
                [1, 2, 1], 
                [0, 0, 0], 
                [-1, -2, -1]
            ])


    for i in range(len(image)):
        for j in range(len(image[0])):
            if (i < padding 
               or j < padding
               or i > len(image) - (padding + 1)
               or j > len(image[0]) - (padding + 1) 
            ):
                new_image[i][j] = 0
            else:
                new_image[i][j] = sum(
                    sum(image[i-padding:i+(padding + 1), j-padding:j+(padding + 1)]*
                    filter_matrix)
                )
        
    return new_image

def sobel(image):
    dogx = filter(image, filter_type='dogx')
    dogy = filter(image, filter_type='dogy')

    return (
            np.sqrt(
                np.add(
                    np.square(dogx),
                    np.square(dogy)
                ),
            ).astype(dogx.dtype)
    )


if __name__ == '__main__':
    filterimg1 = plt.imread(dir_path / 'filter1_img.jpg')
    filterimg2 = plt.imread(dir_path / 'filter2_img.jpg')

    '''plt.imshow(filterimg1, cmap='gray')
    plt.show()
    plt.clf()
    
    post_filterimg1 = filter(filterimg1, padding=1)
    plt.imshow(post_filterimg1, cmap='gray')
    plt.show()
    plt.clf()

    post_filterimg1 = filter(filterimg1, padding=2)
    plt.imshow(post_filterimg1, cmap='gray')
    plt.show()

    dogx_filterimg1 = filter(filterimg1, filter_type='dogx')
    plt.imshow(dogx_filterimg1, cmap='gray')
    plt.show()

    dogy_filterimg1 = filter(filterimg1, filter_type='dogy')
    plt.imshow(dogy_filterimg1, cmap='gray')
    plt.show()'''

    sobel_filterimg1 = sobel(filterimg1)
    print (sobel_filterimg1)
    plt.imshow(sobel_filterimg1, cmap='gray')
    plt.show()

