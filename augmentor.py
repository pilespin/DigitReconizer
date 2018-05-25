
import os


def rotate(img, angle):
	num_rows, num_cols = img.shape
	rotation_matrix = cv2.getRotationMatrix2D((num_cols/2, num_rows/2), angle, 1)
	img_rotation = cv2.warpAffine(img, rotation_matrix, (num_cols, num_rows))
	return img_rotation

def show(img):
	cv2.imshow('image',img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

import numpy as np
import cv2

# Load an color image in grayscale
img = cv2.imread("smallmnist/0/10.png", 0)
if (img is None):
	print("Image not read")

folder = 'new/'
if not os.path.exists(folder):
	os.mkdir(folder)
# show(img)
for angle in range(360):
	cv2.imwrite(folder + 'rotated_' + str(angle) + '.png', rotate(img, angle))
