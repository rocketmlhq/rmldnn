import os
import shutil
import random

src='./data/test/'
dest='./test_samples/'

os.mkdir(dest)

for directory in os.listdir(src):
    random_file=random.choice(os.listdir(src + directory))
    shutil.copy(src + directory + '/' + random_file, dest)
    os.rename(dest + random_file, dest + directory + random_file)

