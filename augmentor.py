
import os
import numpy as np
import cv2

from skimage.transform import swirl
from skimage.io import imsave
import sys	

def rotate(img, angle):
	# tmp = swirl(img, rotation=90)

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



# img = cv2.imread("smallmnist/3/30.png")
# if (img is None):
# 	print("Image not read")

path_new_dataset = 'new/'
# path_new_dataset = 'new2/'
path_dataset = 'smallmnist/'
# path_dataset = 'mnist_png/training/'
if not os.path.exists(path_new_dataset):
	os.mkdir(path_new_dataset)

output = []
for foldername in os.listdir(path_dataset):
	if foldername[0] != '.':
		print("Load folder: " + foldername)
		# current_path = path_new_dataset + foldername + '/'
		output.append(foldername)
print(output)

for current in output:
	print(current)

	current_path = path_dataset + current + '/'
	current_path_new = path_new_dataset + current + '/'
	if not os.path.exists(current_path_new):
		os.mkdir(current_path_new)
	print("path: " + current_path)

	for file in os.listdir(current_path):
		sys.stdout.write(".")
		sys.stdout.flush()
		img = cv2.imread(current_path + file)
		num_rows, num_cols = img.shape[:2]

		if (img is None):
			print("Image not read: " + file)

		for x in range(-3, 3):
			y = num_rows*2
			imsave(current_path_new + file + '_swirl_' + str(x) + '_' + str(y) + '.png', swirl(img, rotation=0, strength=x, radius=y))


		for x in range(-10, 10,3):
			for y in range(-10, 10,3):
				cv2.imwrite(current_path_new + file + '_translated_' + str(x) + '_' + str(y) + '.png', translation(img, x, y))


		for angle in range(-40, 40, 20):
			cv2.imwrite(current_path_new + file +'_rotated_' + str(angle) + '.png', rotate(img, angle))



# cv2.imshow("Original image", tmp)
# cv2.imshow("new image", tmp)
# cv2.waitKey(0)
# cv2.destroyAllWindows()