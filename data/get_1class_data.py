import numpy as np
import pickle
import cv2
import ipdb

select_label = 0

with open('/mnt/data1/yjzhu/ntu_preproc_keypoint/raw_data', 'rb') as f:
    raw = pickle.load(f)

data = raw['data']
class1_data = []
for idx, label in enumerate(raw['labels']):
    # only select class label = 0
    if label == select_label:
        select_data = data[idx]
        # resize is similar to uniform selection
        im = cv2.resize(select_data, (16, 24))
        # mins is (1 x 3) vector: [x, y, z]
        # mins = np.min(im, axis=(0, 1))
        # maxs = np.max(im, axis=(0, 1))
        # im = (im-mins)/(maxs-mins)*255
        class1_data.append(im)
    else:
        pass
class1_data = np.array(class1_data)
print(len(class1_data), 'samples found! label =', select_label )

# save the selected data
with open('class1_data.pkl', 'wb') as f:
    pickle.dump(class1_data, f)


