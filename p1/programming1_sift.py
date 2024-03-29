import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def find_matches(des1, des2):
    """FINDS MATCHES
    
    Inputs:
        des1 (np.array): array of feature descriptors
        des2 (np.array): array of feature descriptors

    Returns:
        tuple(tuple(int, np.array), tuple(int, np.array)):
            tuple of tuples containing matching features
            and associated keypoint image indices
    """
    matches = []

    for i in range(len(des1)):
        e1 = des1[i]
        e2_idx = np.argmin(
            np.linalg.norm(e1 - des2, axis=1)
        )
        matches.append(((i, e1),(e2_idx, des2[e2_idx])))

    return matches
            

if __name__ == '__main__':
    img1 = cv.imread('SIFT1_img.jpg')
    img2 = cv.imread('SIFT2_img.jpg')

    sift = cv.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    matches = find_matches(des1, des2)

    dmatches = tuple(
            (cv.DMatch(
                m[0][0],
                m[1][0], 
                np.linalg.norm(m[0][1] - m[1][1])
            ),)
        for m in matches
    )
    #-- Draw matches
    img_matches = cv.drawMatchesKnn(
        img1, kp1, 
        img2, kp2, 
        dmatches, 
        None,
        flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    #-- Show detected matches
    plt.imshow(cv.cvtColor(img_matches, cv.COLOR_BGR2RGB))
    plt.show()

    top_10percent = int( 0.1 * len(dmatches) )
    dmatches_10perc = sorted(dmatches, key=lambda y: y[0].distance)[0:top_10percent]
    img_matches = cv.drawMatchesKnn(
        img1, kp1, 
        img2, kp2, 
        dmatches_10perc, 
        None,
        flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    #-- Show detected matches
    plt.imshow(cv.cvtColor(img_matches, cv.COLOR_BGR2RGB))
    plt.show()

