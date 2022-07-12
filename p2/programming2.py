from cv2 import solve
from skimage import io
import numpy as np

def zero_pad(image):
    return np.array([
        [
            0 if i == 0 or j == 0 or i >= len(image) or j >= len(image[0]) 
            else image[i][j] for j in range(len(image[0])+1)
        ]
        for i in range (len(image)+1)
    ])

def filter(image, padding=1, filter_type='gaussian'):
    '''Filter function
    
    Inputs:
        image (np.array): 2D matrix
        padding (int): padding size, shorthand for filter size
        filter_type (str): gaussian, difference of gaussians in x and y

    Returns:
        np.array: filtered 2D image
    '''
    new_image = np.array(image)

    if filter_type == 'dogx':
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

def image_difference(frame_a, frame_b):
    """Image differences"""
    if (len(frame_a) != len(frame_b)) or (len(frame_a[0]) != len(frame_b[0]) ):
        raise Exception('Images not the same size')

    return np.array([
        [
            frame_b[i][j] - frame_a[i][j] 
            for j in range(len(frame_a[0]))
        ]
        for i in range (len(frame_a))
    ])

def solve_for_vectors(frame_x, frame_y, frame_t):
    """Solve for x and y vector components for each pixel"""
    vectors = np.array([
        [(0.0, 0.0) for _ in range(len(frame_t[0]))]
        for _ in range(len(frame_t))
    ])
    for i in range(1, len(frame_t)-1):
        for j in range(1, len(frame_t[0]) - 1):
            '''Setup matrices'''
            ix_vec = (frame_x[i-1:i+2, j-1:j+2]).flatten()
            iy_vec = (frame_y[i-1:i+2, j-1:j+2]).flatten()
            ixy_mat = np.stack((ix_vec, iy_vec)).transpose()
            t_vec = (frame_t[i-1:i+2, j-1:j+2]).flatten().reshape(-1,1)
            ixy_trans = ixy_mat.transpose()
            results = np.matmul(
                np.linalg.inv(
                    np.matmul(ixy_trans, ixy_mat)
                ),
                np.matmul(ixy_trans,t_vec)
            )
            vectors[i][j][0] = results[0][0]
            vectors[i][j][1] = results[1][0]
            print (i,j)
    return vectors




if __name__ == '__main__':
    """ Read in and apply zero padding to images """
    frame1_a = zero_pad(io.imread('frame1_a.png', as_gray=True))
    frame1_b = zero_pad(io.imread('frame1_b.png', as_gray=True)) 
    frame2_a = zero_pad(io.imread('frame2_a.png', as_gray=True))
    frame2_b = zero_pad(io.imread('frame2_b.png', as_gray=True))

    """Process frame 1"""
    frame1_ax = filter(frame1_a, filter_type='dogx')
    frame1_ay = filter(frame1_a, filter_type='dogy')
    frame1_t = image_difference(frame1_a, frame1_b)
    solve_for_vectors(frame1_ax, frame1_ay, frame1_t)