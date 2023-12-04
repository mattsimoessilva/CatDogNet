from os import makedirs
from os import listdir
from shutil import copyfile
from random import seed
from random import random

# create directories
subdirs = ['train/train/', 'test/test/']
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
# copy training dataset images into subdirectories
src_directory = 'train/train/'
for file in listdir(src_directory):
	src = src_directory + '/' + file
	dst_dir = 'train/train/'
	if random() < val_ratio:
		dst_dir = 'test/test/'
	if file.startswith('cat'):
		dst = dst_dir + 'cats/'  + file
		copyfile(src, dst)
	elif file.startswith('dog'):
		dst = dst_dir + 'dogs/'  + file
		copyfile(src, dst)

