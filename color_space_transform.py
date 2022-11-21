import matplotlib.pyplot as plt
import cv2
import os


img_path = 'original_image'
im_list = ["1.bmp",
           "2.bmp",
           "3.bmp",
           "4.bmp"]


def color_space_transform(i):
    folder = 'tif/image' + str(i)
    # Original image1: BGR
    img = cv2.imread(os.path.join(img_path, im_list[i]))
    cv2.imwrite(os.path.join(folder, 'BGR.tif'), img)

    # HSV
    img_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    plt.imshow(img_HSV)
    cv2.imwrite(os.path.join(folder, 'HSV.tif'), img_HSV)

    # YCrCb
    img_YcrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    cv2.imwrite(os.path.join(folder, 'YCrCb.tif'), img_YcrCb)

    # LAB
    img_LAB = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    cv2.imwrite(os.path.join(folder, 'LAB.tif'), img_LAB)


for i in range(4):
    color_space_transform(i)
