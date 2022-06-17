import os
import shutil
import random
os.mkdir('test_sample')
src='./data/test/'
dest='./test_sample/'
for directory in os.listdir(src):
    random_file=random.choice(os.listdir(src+directory))
    shutil.copy(src+directory+'/'+random_file,dest)