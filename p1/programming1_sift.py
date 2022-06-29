import cv2 as cv

if __name__ == '__main__':
    img1 = cv.imread('SIFT1_img.jpg')
    img2 = cv.imread('SIFT2_img.jpg')

    sift = cv.SIFT_create()
    gray= cv.cvtColor(img1,cv.COLOR_BGR2GRAY)
    kp = sift.detect(gray,None)
    img=cv.drawKeypoints(gray,kp,img1)
    cv.namedWindow("Display", flags=cv.WINDOW_AUTOSIZE)
    cv.imshow("Display", img)
    cv.waitKey(0)

    gray= cv.cvtColor(img2,cv.COLOR_BGR2GRAY)
    kp = sift.detect(gray,None)
    img=cv.drawKeypoints(gray,kp,img2)
    cv.namedWindow("Display", flags=cv.WINDOW_AUTOSIZE)
    cv.imshow("Display", img)
    cv.waitKey(0)
