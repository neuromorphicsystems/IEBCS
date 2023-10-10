import cv2
from cv2 import data
import numpy as np
import os
import sys
import h5py
cur_dir = os.getcwd()
sys.path.append(cur_dir + "/../../src")
dir_res = "track_dataset_v3"
nb_display = 1
tw = 4000
res = [260, 346]
img = np.zeros((res[0], res[1], 3), dtype=np.uint8)
fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
out = cv2.VideoWriter('trackDataset.avi', fourcc, 100.0, (res[1], res[0]))
for id in range(1, nb_display + 1, 1):
    datafile = h5py.File('{}/{}_event.h5'.format(dir_res, id), 'r+')
    events = np.array(datafile["events"]["recording"])
    datafile = h5py.File('{}/{}_gt.h5'.format(dir_res, id), 'r+')
    print(datafile)
    gt = np.array(datafile["events_GT"]["recording"])
    for t in range(int(events[3, 1]) + tw, int(events[3, -1]), tw):
        img[:, :, :] = 125
        inds_events = np.where((t < events[3, :]) & (events[3, :] < t + tw))
        inds_gt = np.where((t < gt['ts']) & (gt['ts'] < t + tw))
        img[events[1, inds_events], events[0, inds_events], 0] = 125 + 125 * (events[2, inds_events] * 2 - 1)
        img[events[1, inds_events], events[0, inds_events], 1] = 125 + 125 * (events[2, inds_events]  * 2 - 1)
        img[events[1, inds_events], events[0, inds_events], 2] = 125 + 125 * (events[2, inds_events]  * 2 - 1)
        img = cv2.applyColorMap(img, cv2.COLORMAP_VIRIDIS)
        if len(inds_events[0]) > 0 and len(inds_gt[0]) > 0:
            img = cv2.circle(img, (int(gt['x'][inds_gt[0][-1]]), int(gt['y'][inds_gt[0][-1]])), 5, (255, 0, 255))
            img = cv2.putText(img, '{}'.format(id),
                              (int(gt['x'][inds_gt[0][-1]]), int(gt['y'][inds_gt[0][-1]]) + 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255))
            img = cv2.putText(img, '{} s'.format(gt['ts'][inds_gt[0][-1]] / 1e6), (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                              (255, 0, 255))
        cv2.imshow("View", img)
        cv2.waitKey(1)
        out.write(img)
out.release()








