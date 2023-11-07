import cv2 as cv
import matplotlib.pylab as plt 
import numpy as np 

img = cv.imread('Photos/cats.jpg')
cv.imshow('cats', img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Graysclae', gray)

# - grayscale histogram
gray_hist = cv.calcHist(images=[gray], channels=[0], mask=None, histSize=[256], ranges=[0,256])
# images=[gray] - list 
# channels=[0] - index of images, we are passing gray so 0
# histSize=[256] - num of bins 

# plt.figure()
# plt.title('Grayscale Histogram')
# plt.xlabel('Bins')
# plt.ylabel('# of pixels')
# plt.plot(gray_hist)
# plt.xlim([0,256]) # domain
# plt.show()
# --peak at 50-60 means there are almost 4000 pixels of the intensity of 50-60 - cats.jpg
# --peak at 220 -- almost 7000 pixels of intensity 220 (cause white)

# --we can create a mask and then create a hiostogram of masked area
blank = np.zeros(img.shape[:2], 'uint8')
masking = cv.circle(blank, (img.shape[1]//2, img.shape[0]//2), 100, 255, -1)
mask = cv.bitwise_and(img, img, mask=masking)
cv.imshow('Mask', mask)

# - hist for masked gray image
# gray_hist = cv.calcHist(images=[gray], channels=[0], mask=masking, histSize=[256], ranges=[0,256])


# plt.figure()
# plt.title('Grayscale Histogram with Mask')
# plt.xlabel('Bins')
# plt.ylabel('# of pixels')
# plt.plot(gray_hist)
# plt.xlim([0,256]) # domain
# plt.show()


plt.figure()
plt.title('Colored Histogram with Mask ')
plt.xlabel('Bins')
plt.ylabel('# of pixels')
# - coloring histogram
colors = ['b','g','r']
for i, col in enumerate(colors):
    hist = cv.calcHist(images=[img], channels=[i], mask=None,histSize=[256], ranges=[0,256])
    plt.plot(hist, color=col)
    plt.xlim([0,256]) # domain
plt.show()

cv.waitKey(0)