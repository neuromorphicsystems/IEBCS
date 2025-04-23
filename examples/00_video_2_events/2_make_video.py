"""
    Make video of the events
"""
import cv2
import numpy as np
import sys
sys.path.append("../../src")
from dat_files import load_dat_event

fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
filename = './outputs/ev_100_10_100_40_0.4_0.01.dat'
ts, x, y, p = load_dat_event(filename)
res = [1920, 1080]
out = cv2.VideoWriter('{}.avi'.format(filename[:-4]), fourcc, 20.0, (res[0], res[1]))
tw = 1000
img         = np.zeros((res[1], res[0]), dtype=float)
tsurface    = np.zeros((res[1], res[0]), dtype=np.int64)
indsurface  = np.zeros((res[1], res[0]), dtype=np.int8)

for t in range(ts[0], ts[-1], tw):
    # Get events in the current time window
    ind = np.where((ts > t) & (ts < t + tw))

    # Create a matrix holding the time stamps of the events
    tsurface[:, :] = 0
    tsurface[y[ind], x[ind]] = t + tw

    # And another holding their polarity (use -1 for OFF events)
    indsurface[y[ind], x[ind]] = 2.0 * p[ind] - 1

    # Find which pixels to process
    ind = np.where(tsurface > 0)

    # And update the image
    img[:, :] = 125
    img[ind] = 125 + indsurface[ind] * np.exp(-(t + tw - tsurface[ind].astype(np.float32))/ (tw/30)) * 125

    # Convert to color and display
    img_c = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    img_c = cv2.putText(img_c, '{} us'.format(t + tw), (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                          (255, 255, 255))
    img_c = cv2.applyColorMap(img_c, cv2.COLORMAP_VIRIDIS)
    cv2.imshow("debug", img_c)
    cv2.waitKey(1)
    
    # Write video to file
    out.write(img_c)
out.release()