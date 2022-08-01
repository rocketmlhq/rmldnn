import numpy as np
import h5py as h5

right = 0
size  = 400
pred = h5.File('/predictions/output_1.h5', 'r')
i=0

for dataset in pred:
    print(np.argmax(pred[dataset][()]), end=' ')
    x=np.argmax(pred[dataset][()])
    if (x == i):
        right += 1
    i+=1
    
print("\n\nAccuracy is " + str(100 * right / size) +'%')
