"""
    Make video of the events
"""
import sys
sys.path.append("../../src")
import cv2
import numpy as np
from dat_files import load_dat_event
fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
filename = '../../data/tmp/events/MNIST_ev_0_1_td.dat'
ts, x, y, p = load_dat_event(filename)
res = [28, 28]
out = cv2.VideoWriter('nmnist_events.avi', fourcc, 20.0, (res[1], res[0]))
tw = 1000
img = np.zeros((res[0], res[1]), dtype=np.uint8)
tsurface = np.zeros((res[0], res[1]), dtype=np.uint64)
indsurface = np.zeros((res[0], res[1]), dtype=np.uint64)
img = np.zeros((res[0], res[1]), dtype=np.uint8)
tsurface = np.zeros((res[0], res[1]), dtype=np.uint64)
indsurface = np.zeros((res[0], res[1]), dtype=np.uint64)
for t in range(ts[0], ts[-1], tw):
    ind = np.where((ts > t)&(ts < t + tw))
    tsurface[:, :] = 0
    tsurface[y[ind], x[ind]] = t + tw
    indsurface[y[ind], x[ind]] = p[ind]
    ind = np.where(tsurface > 0)
    img[:, :] = 125
    img[ind] = 125 + (2 * indsurface[ind] - 1) * np.exp(-(t + tw - tsurface[ind].astype(np.float32))/ (tw/30)) * 125
    img_c = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img_c = cv2.applyColorMap(img_c, cv2.COLORMAP_VIRIDIS)
    cv2.imshow("debug", img_c)
    cv2.waitKey(1)
    out.write(img_c)
out.release()