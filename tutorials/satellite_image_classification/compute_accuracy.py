import numpy as np
import h5py as h5

right = 0
total = 0
class_labels = { "AnnualCrop"           :0,
                 "Forest"               :1,
                 "HerbaceousVegetation" :2,
                 "Highway"              :3,
                 "Industrial"           :4,
                 "Pasture"              :5,
                 "PermanentCrop"        :6,
                 "Residential"          :7,
                 "River"                :8,
                 "SeaLake"              :9
               }

h5file = h5.File('./predictions/output_1.h5', 'r')

for group in h5file:
    for dset in h5file[group]:
        pred_label = np.argmax(h5file[group][dset][()])
        if pred_label == class_labels[group]:
            right += 1
        total += 1

print(f"Accuracy: {100 * right / total:.1f}%")
