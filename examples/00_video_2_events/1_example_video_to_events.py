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
lat = 100
jit = 10
ref = 100
tau = 300
th = 0.3
th_noise = 0.01
cap = cv2.VideoCapture(filename)
dsi.initSimu(int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
dsi.initLatency(lat, jit, ref, tau)
dsi.initContrast(th, th, th_noise)
init_bgn_hist_cpp("../../data/noise_pos_161lux.npy", "../../data/noise_pos_161lux.npy")
isInit = False
dt = 1000  # FPS must be 1 kHz
ev_full = EventBuffer(1)
ed = EventDisplay("Events", cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT), dt*2)
time = 0
out = cv2.VideoWriter('video_{}_{}_{}_{}_{}_{}_nonoise.avi'.format(lat, jit, ref, tau, th, th_noise),
    cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 20.0, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
while cap.isOpened():
    ret, im = cap.read()
    out.write(im)
    if im is None:
        break
    im = cv2.cvtColor(im, cv2.COLOR_RGB2LUV)[:, :, 0]
    cv2.imshow("t", im)
    cv2.waitKey(1)
    if not isInit:
        dsi.initImg(im)
        isInit = True
    else:
        buf = dsi.updateImg(im, dt)
        ev = EventBuffer(1)
        ev.add_array(np.array(buf["ts"], dtype=np.uint64),
                         np.array(buf["x"], dtype=np.uint16),
                         np.array(buf["y"], dtype=np.uint16),
                         np.array(buf["p"], dtype=np.uint64),
                         100000000)
        ed.update(ev, dt)
        ev_full.increase_ev(ev)
        time += dt
        if time > 0.1e6:
            break
out.release()
cap.release()
ev_full.write('ev_{}_{}_{}_{}_{}_{}.dat'.format(lat, jit, ref, tau, th, th_noise))