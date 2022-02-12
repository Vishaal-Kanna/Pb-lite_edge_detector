#!/usr/bin/env python

"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Homework 0: Alohomora: Phase 1 Starter Code


Author(s): 
Nitin J. Sanket (nitin@terpmail.umd.edu)
PhD Candidate in Computer Science,
University of Maryland, College Park

Chahat Deep Singh (chahat@terpmail.umd.edu) 
PhD Student in Computer Science,
University of Maryland, College Park

Vishaal Kanna Sivakumar (vishaal@terpmail.umd.edu)
M.Eng. Student, Robotics
University of Maryland, College Park

"""

# Code starts here:

import glob
import numpy as np
import math
import cv2
import matplotlib.pyplot as plt
from scipy import signal
from skimage.transform import rotate
from sklearn.cluster import KMeans

def Gaussian1D(sigma,size):
	e = 2.718
	gauss1D = np.zeros(size)
	s = int(size/2)
	for i in range(-s,s):
		gauss1D[i+s+1] = (e**(-((i+1)*(i+1))/(2*(sigma*sigma))))/(2*3.14*(sigma*sigma))
	return gauss1D

def Gaussian2D(sigma1,sigma2,size):
	e = 2.718
	if size%2==0:
		add=0
	else:
		add=1
	gauss2D = np.zeros((size,size))
	s = int(size/2)
	for i in range(-s, s):
		for j in range(-s, s):
			gauss2D[i+s+add, j+s+add] = (e**(-((i+1)*(i+1)/(2*(sigma1*sigma1))+(j+1)*(j+1)/(2*(sigma2*sigma2)))))/(2*3.14*(sigma1*sigma1/2+sigma2*sigma2/2))
	return gauss2D

def Gabor(sigma1,sigma2,size,theta,Lamda,psi):
	e = 2.718
	if size%2==0:
		add=0
	else:
		add=1
	Gabor_filter = np.zeros((size,size))
	s = int(size/2)
	for i in range(-s, s):
		for j in range(-s, s):
			i_theta = i*math.cos(theta) + j*math.sin(theta)
			j_theta = -i*math.sin(theta) + j*math.cos(theta)
			Gabor_filter[i+s+add, j+s+add] = (e**(-((i_theta+1)*(i_theta+1)/(2*(sigma1*sigma1))+(j_theta+1)*(j_theta+1)/(2*(sigma2*sigma2)))))*(math.cos(2*3.14*i_theta/Lamda+psi))
	return Gabor_filter

def Gabor_bank(size,Lamda,psi):
	filter_bank=[]
	sigma = np.linspace(7,15,num=5)
	theta = np.linspace(0, 3.14, num=8)
	k=1
	for i in range(0,5):
		for j in range(0,8):
			Gabor_filter = Gabor(sigma[i],sigma[i],size,theta[j],(i+1)*Lamda,psi)
			filter_bank.append(Gabor_filter)
			plt.subplot(5,8,k)
			plt.axis('off')
			plt.imshow(filter_bank[k-1],cmap='gray')
			k = k+1
	plt.subplots_adjust(wspace=0.15, hspace=0.5)
	plt.show()
	return filter_bank

def DoG_bank(scales,orientations,size):
	filter_bank = []
	sobel = np.matrix([[1,0,-1],[2,0,-2],[1,0,-1]])
	k=1
	for s in scales:
		DoG_base = signal.convolve2d(Gaussian2D(s,s,size), sobel, boundary='symm', mode='same')
		for o in orientations:
			DoG_rot = rotate(DoG_base,o)
			filter_bank.append(DoG_rot)
			plt.subplot(scales.shape[0],orientations.shape[0],k)
			plt.axis('off')
			#plt.imshow(filter_bank[k-1],cmap='gray')
			k = k+1
	plt.subplots_adjust(wspace=0.15, hspace=-0.8)
	#plt.show()
	return filter_bank

def LM_bank(size,base_scale):
	filter_bank = []
	first_der = 0.5*np.matrix([[-1], [0], [1]])
	sec_der = np.matrix([[1], [-2], [1]])
	lapl = np.matrix([[0,1,0],[1,-4,1],[0,1,0]])
	k=1
	s = base_scale
	for i in range(1,4):
		for j in range(1,13):
			if j <= 6:
				LM_base = signal.convolve2d(Gaussian2D(s, 3 * s, size), first_der, boundary='symm', mode='same')
			else:
				LM_base = signal.convolve2d(Gaussian2D(s, 3 * s, size), sec_der, boundary='symm', mode='same')
			LM_rot = rotate(LM_base, (j-1)*360/12)
			filter_bank.append(LM_rot)
			plt.subplot(4,12,k)
			plt.axis('off')
			plt.imshow(filter_bank[k-1],cmap='gray')
			k = k+1
		s = s * (2 ** 0.5)
	s = base_scale
	for j in range(1,13):
		if j <= 4:
			LM_base = signal.convolve2d(Gaussian2D(s, s, size), lapl, boundary='symm', mode='same')
		elif j <= 8:
			LM_base = signal.convolve2d(Gaussian2D(3 * s, 3 * s, size), lapl, boundary='symm', mode='same')
		else:
			LM_base = Gaussian2D(s, s, size)
		filter_bank.append(LM_base)
		plt.subplot(4,12,k)
		plt.axis('off')
		#plt.imshow(filter_bank[k-1],cmap='gray')
		k = k+1
		s = s * (2 ** 0.5)
		if j%4 == 0:
			s = base_scale
	plt.subplots_adjust(wspace=0.15, hspace=-0.8)
	#plt.show()
	return filter_bank

def Texton(img, DoG_filter_bank, LM_filter_bank, Gabor_filter_bank):
	Texton_filtered = np.zeros((img.size,len(DoG_filter_bank)+len(LM_filter_bank)+len(Gabor_filter_bank)))
	for i in range(0,len(DoG_filter_bank)):
		Texton_filtered_temp = cv2.filter2D(img, -1, DoG_filter_bank[i]) #(img, DoG_filter_bank[i], boundary='symm', mode='same')
		Texton_filtered_temp = Texton_filtered_temp.reshape((1, img.size))
		Texton_filtered[:, i] = Texton_filtered_temp
	for i in range(len(DoG_filter_bank),len(DoG_filter_bank)+len(LM_filter_bank)):
		Texton_filtered_temp = cv2.filter2D(img, -1, LM_filter_bank[i-len(DoG_filter_bank)]) # boundary='symm', mode='same')
		Texton_filtered_temp = Texton_filtered_temp.reshape((1, img.size))
		Texton_filtered[:, i] = Texton_filtered_temp
	for i in range(len(DoG_filter_bank)+len(LM_filter_bank),len(DoG_filter_bank)+len(LM_filter_bank)+len(Gabor_filter_bank)):
		Texton_filtered_temp = cv2.filter2D(img, -1, Gabor_filter_bank[i-len(DoG_filter_bank)-len(LM_filter_bank)]) #, boundary='symm', mode='same')
		Texton_filtered_temp = Texton_filtered_temp.reshape((1, img.size))
		Texton_filtered[:, i] = Texton_filtered_temp

	return Texton_filtered

def half_disk(r, o):
	half_disk_l = np.zeros((2*r,2*r))
	half_disk_r = np.zeros((2*r,2*r))
	for i in range(0,2*r):
		for j in range(0,r):
			if (i-r+1)**2+(j-r+1)**2 <= r**2:
				half_disk_l[i,j] = 1
				half_disk_r[i, 2*r-j-1] = 1
	half_disk_l = rotate(half_disk_l, o)
	half_disk_r = rotate(half_disk_r, o)
	return half_disk_l, half_disk_r

def half_disks_gen(rad,orientations):
	half_disk_masks = []
	for r in rad:
		for o in orientations:
			half_disk_mask_l, half_disk_mask_r = half_disk(int(r),o)
			half_disk_masks.append(half_disk_mask_l)
			half_disk_masks.append(half_disk_mask_r)
	for k in range(1,len(half_disk_masks)+1):
		plt.subplot(2*rad.shape[0],orientations.shape[0],k)
		plt.axis('off')
		#plt.imshow(half_disk_masks[k-1])

	plt.subplots_adjust(wspace=0.15, hspace=0.5)
	#plt.show()
	return half_disk_masks

def bin_map(map,bin_value):
	bin = map*0
	for i in range(map.shape[0]):
		for j in range(map.shape[1]):
			if map[i,j] == bin_value:
				bin[i,j] = 1
	return bin

def Gradient(map, num_bins, masks):
	map_gradient = np.zeros((map.shape[0],map.shape[1],int(len(masks)/2)))
	for m in range(0,int(len(masks)/2)):
		chi_sq_dist = np.zeros((map.shape))
		for i in range(1,num_bins):
			tmp = map*0
			tmp[map == i] = 1
			g_i = cv2.filter2D(tmp, -1, masks[2*m]) #, boundary='symm', mode='same')
			h_i = cv2.filter2D(tmp, -1, masks[2*m+1]) #, boundary='symm', mode='same')
			chi_sq_dist += ((g_i-h_i)**2)/(g_i+h_i+0.00001)
		map_gradient[:,:,m] = chi_sq_dist
	map_gradient_mean = np.mean(map_gradient, axis=2)
	return map_gradient_mean

def main():
	img_list = []
	img_list_canny = []
	img_list_sobel = []
	img_list_gt = []

	for filename in glob.glob('BSDS500/Images/*.jpg'):
		#img = cv2.imread(filename)
		img_list.append(filename)
	img_list.sort()

	for filename in glob.glob('BSDS500/SobelBaseline/*.png'):
		img_list_sobel.append(filename)
	img_list_sobel.sort()

	for filename in glob.glob('BSDS500/CannyBaseline/*.png'):
		img_list_canny.append(filename)
	img_list_canny.sort()

	fs = 25

	"""
	Generate Difference of Gaussian Filter Bank: (DoG)
	Display all the filters in this filter bank and save image as DoG.png,
	use command "cv2.imwrite(...)"
	"""
	scales = np.linspace(1, 7, num=4)
	orientations = np.linspace(360/16, 360, num=16)
	DoG_filter_bank = DoG_bank(scales,orientations,fs)

	"""
	Generate Leung-Malik Filter Bank: (LM)
	Display all the filters in this filter bank and save image as LM.png,
	use command "cv2.imwrite(...)"
	"""
	LM_filter_bank = LM_bank(fs,1)

	"""
	Generate Gabor Filter Bank: (Gabor)
	Display all the filters in this filter bank and save image as Gabor.png,
	use command "cv2.imwrite(...)"
	"""
	Gabor_filter_bank = Gabor_bank(fs,2*3.14,0)

	"""
	Generate Half-disk masks
	Display all the Half-disk masks and save image as HDMasks.png,
	use command "cv2.imwrite(...)"
	"""
	rad = np.linspace(5,15,num=3)
	orientations = np.linspace(0,180-180/8,num=8)
	half_disk_masks = half_disks_gen(rad,orientations)

	for index in range(0,len(img_list)):
		img_color = cv2.imread(img_list[index])
		img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
		#plt.imshow(img_gray)
		#plt.show()

		"""
		Generate Texton Map
		Filter image using oriented gaussian filter bank
		"""
		Texton_filtered = Texton(img_gray, DoG_filter_bank, LM_filter_bank, Gabor_filter_bank)

		"""
		Generate texture ID's using K-means clustering
		Display texton map and save image as TextonMap_ImageName.png,
		use command "cv2.imwrite('...)"
		"""
		Texton_map_cluster = KMeans(n_clusters=16, random_state=0).fit(Texton_filtered)
		labels = Texton_map_cluster.labels_
		Texton_map = np.reshape(labels, (img_gray.shape))
		plt.imshow(Texton_map, 'hsv')
		plt.show()
		cv2.imwrite('BSDS500/Texton_Maps/TextonMap_{}.png'.format(index), Texton_map)

		"""
		Generate Texton Gradient (Tg)
		Perform Chi-square calculation on Texton Map
		Display Tg and save image as Tg_ImageName.png,
		use command "cv2.imwrite(...)"
		"""
		Tg = Gradient(Texton_map, 16, half_disk_masks)
		img_f = np.array(Tg * (1 / Tg.max()) * 255, dtype=np.uint8)
		Tg_thin = cv2.ximgproc.thinning(img_f)
		plt.imshow(Tg_thin, cmap='gray')
		plt.show()
		cv2.imwrite('BSDS500/Texton_Gradients/Tg_{}.png'.format(index), Tg)

		"""
		Generate Brightness Map
		Perform brightness binning 
		"""
		Brightness = img_gray.reshape((img_gray.shape[0]*img_gray.shape[1]),1)
		Brightness_map_cluster = KMeans(n_clusters=16, random_state=0).fit(Brightness)
		labels = Brightness_map_cluster.labels_
		Brightness_map = np.reshape(labels, (img_gray.shape))
		#plt.imshow(Brightness_map, 'hsv')
		#plt.show()

		"""
		Generate Brightness Gradient (Bg)
		Perform Chi-square calculation on Brightness Map
		Display Bg and save image as Bg_ImageName.png,
		use command "cv2.imwrite(...)"
		"""
		Bg = Gradient(Brightness_map, 16, half_disk_masks)
		img_f = np.array(Bg * (1 / Bg.max()) * 255, dtype=np.uint8)
		Bg_thin = cv2.ximgproc.thinning(img_f)
		plt.imshow(Bg_thin, 'gray')
		plt.show()
		cv2.imwrite('BSDS500/Brightness_Gradients/Bg_{}.png'.format(index), Bg)

		"""
		Generate Color Map
		Perform color binning or clustering
		"""
		Color = img_color.reshape((img_gray.shape[0]*img_gray.shape[1]),3)
		Color_map_cluster = KMeans(n_clusters=16, random_state=0).fit(Color)
		labels = Color_map_cluster.labels_
		Color_map = np.reshape(labels, (img_gray.shape))
		#plt.imshow(Color_map, 'hsv')
		#plt.show()

		"""
		Generate Color Gradient (Cg)
		Perform Chi-square calculation on Color Map
		Display Cg and save image as Cg_ImageName.png,
		use command "cv2.imwrite(...)"
		"""
		Cg = Gradient(Color_map, 16, half_disk_masks)
		img_f = np.array(Cg * (1 / Cg.max()) * 255, dtype=np.uint8)
		Cg_thin = cv2.ximgproc.thinning(img_f)
		plt.imshow(Cg_thin, 'gray')
		plt.show()
		cv2.imwrite('BSDS500/Color_Gradients/Cg_{}.png'.format(index), Cg)

		"""
		Read Sobel Baseline
		use command "cv2.imread(...)"
		"""
		Sobel_img = cv2.imread(img_list_sobel[0])

		"""
		Read Canny Baseline
		use command "cv2.imread(...)"
		"""
		Canny_img = cv2.imread(img_list_canny[0])

		"""
		Combine responses to get pb-lite output
		Display PbLite and save image as PbLite_ImageName.png
		use command "cv2.imwrite(...)"
		"""
		w1 = 0.5
		Pb_canny = cv2.cvtColor(Canny_img,cv2.COLOR_BGR2GRAY)
		Pb_sobel = cv2.cvtColor(Sobel_img, cv2.COLOR_BGR2GRAY)
		Pb_edges = 0.2*(Tg_thin+Bg_thin+Cg_thin+Pb_canny+Pb_sobel)
		print(Pb_edges.max)
		plt.imshow(Pb_edges, 'gray')
		plt.show()
		cv2.imwrite('BSDS500/Pb_edges/PbLite_{}.png'.format(index), Pb_edges)

if __name__ == '__main__':
    main()
 


