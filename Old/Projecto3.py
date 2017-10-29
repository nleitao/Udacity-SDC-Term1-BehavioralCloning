import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import pickle

# with open('test.p', 'rb') as f:
#     train_data = pickle.load(f)
# X_train = train_data['features']
# y_train = train_data['labels']
#
# print(X_train.shape)
# print(y_train.shape)
#
# print(y_train)
#
# (12630, 32, 32, 3)
# (12630,)
# [16  1 38 ...,  6  7 10]


# reading in an image
# printing out some stats and plotting

#image = mpimg.imread('data/data/IMG/center_2016_12_01_13_30_48_287.jpg')
#image = mpimg.imread('data/data/IMG/center_2016_12_01_13_34_06_757.jpg')
#image = mpimg.imread('data/data/IMG/center_2016_12_01_13_35_35_878.jpg')
#image = mpimg.imread('data/data/IMG/center_2016_12_01_13_37_29_447.jpg')
image = mpimg.imread('data/data/IMG/center_2016_12_01_13_39_12_047.jpg')


print('This image is:', type(image), 'with dimesions:', image.shape)
#plt.imshow(image)
#plt.show()
# -> size of image: (160,320,3) , is already a numpy.ndarray

#preprocess image (code copied from project 1)
#gray scale xxx
#gaussian xxx
#normalize xx
#laplacian?
#Clahe?
#cannyxxx
#mask image? xxx

def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def normalize(img):
    a=0
    b=1
    minClr=0
    maxClr=255
    return a+(((img-minClr)*(b-a))/(maxClr-minClr))

def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def canny(img, low_threshold, high_threshold):
    return cv2.Canny(img, low_threshold, high_threshold)


#parameters for image preprocessing
midpoint=65
vertices = np.array([[(0,image.shape[0]),(image.shape[1], image.shape[0]), (image.shape[1], midpoint), (0,midpoint)]], dtype=np.int32)
vertices2 = np.array([[(0,0),(0,image.shape[0]),(120, 100),(200,100), (image.shape[1], image.shape[0]), (image.shape[1],0),(0,0)]], dtype=np.int32)
kernel_size_gaussian=9
low_threshold_canny=10
high_threshold_canny=170

image=grayscale(image)

plt.imshow(image,cmap='gray')
plt.show()

image=gaussian_blur(image,kernel_size_gaussian)
image=canny(image,low_threshold_canny,high_threshold_canny)
image=normalize(image)
image=region_of_interest(image,vertices)
image=region_of_interest(image,vertices2)

plt.imshow(image,cmap='gray')
plt.show()

# print(image)
# print(image.shape)
#image size is now (160,320)


image = cv2.resize(image, (160, 80),interpolation = cv2.INTER_CUBIC) #resize to 32x32xY\n",

#image size is now (80,160)

#image.reshape((-1,80,160,1)).astype(np.float32)

#print(image.shape)
#plt.imshow(image,cmap='gray')
#plt.show()

