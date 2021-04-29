# Joubert Damien, 03-02-2020
"""
    Script converting a video into events. The framerate of the video might not be the real framerate of the original video
    . The user specifies this parameter at the beginning.
    Please run get_video_youtube.py before
"""
import cv2
import sys
sys.path.append("../../src")
from event_buffer import EventBuffer
from dvs_sensor import init_bgn_hist_cpp, DvsSensor
from event_display import EventDisplay
import dsi
import numpy as np
filename = "../../data/video/See Hummingbirds Fly Shake Drink in Amazing Slow Motion  National Geographic.mp4"
cap = cv2.VideoCapture(filename)
dsi.initSimu([cap.get(cv2.CAP_PROP_FRAME_HEIGHT)], [cap.get(cv2.CAP_PROP_FRAME_WIDTH)])
dsi.initLatency([200], [50], [50], [300])
dsi.initContrast([0.3], [0.3], [0.05])
init_bgn_hist_cpp("../../data/noise_pos_161lux.npy", "../../data/noise_pos_161lux.npy")
isInit = False
dt = int(1e6 / cap.get(cv2.CAP_PROP_FPS))  # dt between two fames of the video
dt = 1000  # FPS must be 1 kHz
ev_full = EventBuffer(1)
ed = EventDisplay("Events", cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT), dt*2)
time = 0
while cap.isOpened():
    ret, im = cap.read()
    im = cv2.cvtColor(im, cv2.COLOR_RGB2LUV)[:, :, 0]
    cv2.imshow("t", im)
    cv2.waitKey(1)
    if not isInit:
        dsi.initImg(im)
        isInit = True
    else:
        buf = dsi.updateImg(im, [dt])
        ev = EventBuffer(1)
        ev.add_array(np.array(buf["ts"], dtype=np.uint64),
                         np.array(buf["x"], dtype=np.uint16),
                         np.array(buf["y"], dtype=np.uint16),
                         np.array(buf["p"], dtype=np.uint64),
                         100000000)
        ed.update(ev, dt)
        ev_full.increase_ev(ev)
        time += dt
        if time > 1e6:
            break

cap.release()
ev_full.write("ev_0.3_td.dat")