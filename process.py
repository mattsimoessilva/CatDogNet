from os import makedirs
from os import listdir
from shutil import copyfile
from random import seed
from random import random
import os

# create directories
subdirs = ['train/train']
for subdir in subdirs:
	# create label subdirectories
	labeldirs = ['dogs/', 'cats/']
	for labldir in labeldirs:
		newdir = subdir + labldir
		makedirs(newdir, exist_ok=True)

# seed random number generator
seed(1)
# define ratio of pictures to use for validation
val_ratio = 0.25
# move training dataset images into subdirectories
src_directory = 'train/train'
for file in listdir(src_directory):
	src = src_directory + '/' + file
	dst_dir = 'train/train'
	if file.startswith('cat'):
		dst = dst_dir + 'cats/'  + file
		copyfile(src, dst)
		os.remove(src)  # remove the original file
	elif file.startswith('dog'):
		dst = dst_dir + 'dogs/'  + file
		copyfile(src, dst)
		os.remove(src)  # remove the original file