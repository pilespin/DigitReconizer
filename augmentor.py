
import os
import numpy as np
import cv2


def rotate(img, angle):
	num_rows, num_cols = img.shape[:2]
	rotation_matrix = cv2.getRotationMatrix2D((num_cols/2, num_rows/2), angle, 1)
	img_rotation = cv2.warpAffine(img, rotation_matrix, (num_cols, num_rows))
	return img_rotation

def show(img):
	cv2.imshow('image',img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def scale(img, scale):
	new = cv2.resize(img, None, fx=scale, fy=scale)
	return new

def erosion(img, kernel_size=2, iterations = 1):
	kernel = np.ones((kernel_size, kernel_size), np.uint8)
	new = cv2.erode(img, kernel, iterations = iterations)
	return new

def dilatation(img, kernel_size=2, iterations = 1):
	kernel = np.ones((kernel_size, kernel_size), np.uint8)
	new = cv2.dilate(img, kernel, iterations = iterations)
	return new

def translation(img, x=0, y=0):
	num_rows, num_cols = img.shape[:2]
	M = np.float32([[1,0,x],[0,1,y]])
	new = cv2.warpAffine(img,M,(num_cols,num_rows))
	return new

def flip(img):
	new = cv2.flip(img, 1)
	return new

def binarize(img, threshold=0, max=255):
	_, new = cv2.threshold(img, threshold, max, cv2.THRESH_BINARY);
	return new

def blur(img, kernel_size=2):
	# kernel = np.ones((kernel_size, kernel_size), np.uint8)
	new = cv2.blur(img, (kernel_size,kernel_size))
	return new



img = cv2.imread("smallmnist/0/10.png")
if (img is None):
	print("Image not read")

folder = 'new/'
if not os.path.exists(folder):
	os.mkdir(folder)

for angle in range(360):
	cv2.imwrite(folder + 'rotated_' + str(angle) + '.png', rotate(img, angle))

for foldername in os.listdir(folder):
	if foldername[0] != '.':
		print("Load img: " + foldername)
		for x in range(-10, 10):
			for y in range(-10, 10):
				cv2.imwrite(folder + foldername + '_translated_' + str(x) + '_' + str(y) + '.png', translation(img, x, y))



# for x in range(-10, 10):
# 	for y in range(-10, 10):
# 		cv2.imwrite(folder + 'translated_' + str(x) + '_' + str(y) + '.png', translation(img, x, y))



# cv2.imshow("Original image", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()