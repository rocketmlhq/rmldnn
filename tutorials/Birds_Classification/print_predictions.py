import numpy as np
import os
for file in sorted(os.listdir('./predictions/')):
    print(np.argmax(np.load('./predictions/' + file)), end=' ')