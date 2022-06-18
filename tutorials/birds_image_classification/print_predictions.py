import numpy as np
import os

right = 0
size  = 400

for i in range(size):
    x = np.argmax(np.load('./predictions/output_1_' + str(i) +'.npy'))
    print(x, end=' ')
    if (x == i):
        right += 1

print("\n\nAccuracy is " + str(100 * right / size) +'%')
