import os
import pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt
from glob import glob
import ipdb

# the i_th sequence to be shown.
gt_flag = False
if gt_flag:
    out_dir = 'gt'
    pred_path = 'class1_data.pkl'
else:
    out_dir = 'pred'
    pred_path = 'pred_data.pkl'

img_res = (1080,1920)
index = 1

with open(pred_path, 'rb') as fin:
    data = pickle.load(fin)

# ipdb.set_trace()
# data: (931, 24, 16, 3) numpy.ndarray
# data = data[900:]
pairs=[[0,1], [1,2], [2,3], [3,4], [1,5], [5,6], [6,7], [7,8], [8,9], [6,10], [10,11], [11,12], [1,13], [13,14], [14,15]]
points=np.zeros((16,2)).astype(np.int32)
# all the frames
for ind in range(data[index].shape[0]):
    visdata = data[index][:,:,:2][ind].reshape(-1,1)[:,0]
    #ipdb.set_trace()
    im=np.zeros(img_res).astype(np.uint8)
    for i in range(16):
        j=i
        #x=int(visdata[2*j]/256*1920)
        #y=int(visdata[2*j+1]/256*1080)
        x=int(visdata[2*j])
        y=int(visdata[2*j+1])
        points[i]=x,y
        cv2.circle(im, (x,y), 1, 255, 2)
    for pair in pairs:
        cv2.line(im, tuple(points[pair[0]]), tuple(points[pair[1]]), 255, 1)
    cv2.imwrite(out_dir + '/%02d.png' % ind, im)
print('Plot %d frames: Done!' % ind)

# images to video
img_path = sorted(glob(out_dir + '/*.png'))
video_path = out_dir + '/exp.avi'
# ipdb.set_trace()
# fourcc = cv2.VideoWriter_fourcc(*'H264')
fourcc = cv2.VideoWriter_fourcc(*'XVID')
fps = 10
video_res = (1920, 1080)
video_writer = cv2.VideoWriter(video_path, fourcc, fps, video_res)

for path in img_path:
    img = cv2.imread(path)
    img = cv2.resize(img, video_res)
    video_writer.write(img)
video_writer.release()
print('Image to video: Done!')




