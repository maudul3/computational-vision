import cv2
import numpy as np

def zero_pad(image):
    """Returns a zero-padded image"""
    return np.array([
        [
            0 if i == 0 or j == 0 or i >= len(image) or j >= len(image[0]) 
            else image[i][j] for j in range(len(image[0])+1)
        ]
        for i in range (len(image)+1)
    ]).astype('uint8')

def image_difference(frame_a, frame_b):
    """Returns the difference of two images"""
    if (len(frame_a) != len(frame_b)) or (len(frame_a[0]) != len(frame_b[0]) ):
        raise Exception('Images not the same size')

    frame_aint = frame_a.astype('int16')
    frame_bint = frame_b.astype('int16')
    return np.array([
        [
            frame_bint[i][j] - frame_aint[i][j] 
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
            '''Setup vectors and use OLS-derived equation to solve'''
            ix_vec = (frame_x[i-1:i+2, j-1:j+2]).flatten()
            iy_vec = (frame_y[i-1:i+2, j-1:j+2]).flatten()
            ixy_mat = np.stack((ix_vec, iy_vec)).transpose()
            t_vec = (frame_t[i-1:i+2, j-1:j+2]).flatten().reshape(-1,1)
            ixy_trans = ixy_mat.transpose()
            results = np.matmul(
                np.linalg.pinv(
                    np.matmul(ixy_trans, ixy_mat)
                ),
                np.matmul(ixy_trans,-1*t_vec)
            )
            vectors[i][j][0] = results[0][0]
            vectors[i][j][1] = results[1][0]
    return vectors

def process(frame_a, frame_b, title):
    """Determine the vectors for the two frames and 
    plot the vectors as well as the magnitudes."""
    scale = 1
    delta = 0
    ddepth = cv2.CV_16S

    # Find Sobel X and Y frames
    frame_ax = cv2.Sobel(
        frame_a, 
        ddepth, 
        1,
        0,
        ksize=3, 
        scale=scale, 
        delta=delta,
        borderType=cv2.BORDER_DEFAULT
    )
    frame_ay = cv2.Sobel(
        frame_a, 
        ddepth,
        0,
        1,
        ksize=3, 
        scale=scale, 
        delta=delta,
        borderType=cv2.BORDER_DEFAULT
    )
    # Calculate temporal differences in frames
    frame_t = image_difference(frame_a, frame_b)
    # Find Vx and Vy for all pixels
    frame_vectors = solve_for_vectors(frame_ax, frame_ay, frame_t)
    # Determine the magnitude for all pixels vectors
    frame_magnitudes =  np.array([
        [
            np.sqrt(frame_vectors[i][j][0]**2 + frame_vectors[i][j][1]**2)
            for j in range(len(frame_vectors[0]))
        ]
        for i in range (len(frame_vectors))
    ])
    # Plot the vectors
    frame_rgb_img =np.array([
        [
            (frame_a[i][j], frame_a[i][j], frame_a[i][j])
            for j in range(len(frame_a[0]))
        ]
        for i in range (len(frame_a))
    ])
    for i in range(len(frame_a)):
        for j in range(len(frame_a[0])):
            if (i % 3 == 0 and j % 3 == 0) and frame_magnitudes[i][j] > 0.5:
                frame_rgb_img = cv2.arrowedLine(
                    frame_rgb_img,
                    (j, i),
                    (int(j + int(np.clip(frame_vectors[i][j][0], -20, 20))),
                    int(i + int(np.clip(frame_vectors[i][j][0], -20, 20)))),
                    (0,255,0),
                    1
                )
    # Save the vector and magnitude plots
    cv2.imwrite("./" + title + "vectors.png", frame_rgb_img)
    cv2.imwrite("./" + title + "magnitude.png", np.clip(frame_magnitudes*255, 0, 255))


if __name__ == '__main__':
    # Read in and apply zero padding to images 
    frame1_a = zero_pad(cv2.imread('frame1_a.png', 0))
    frame1_b = zero_pad(cv2.imread('frame1_b.png', 0)) 
    frame2_a = zero_pad(cv2.imread('frame2_a.png', 0))
    frame2_b = zero_pad(cv2.imread('frame2_b.png', 0))

    # Process frames 1 and 2 
    process(frame1_a, frame1_b, "frame1")
    process(frame2_a, frame2_b, "frame2")