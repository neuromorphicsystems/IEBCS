import numpy as np
import os
import sys
sys.path.append("./src")
from dvs_sensor import DvsSensor, EventBuffer
from event_display import EventDisplay
from tqdm import tqdm
import h5py

dir_res = "track_dataset_v3"
if not os.path.exists(dir_res):
    os.mkdir(dir_res)

# Parameters
res = [346, 260]
th = 0.15
th_noise = 0.03
lat = 300
jit = 100
tau = 1000
bgn = 0.01
ref = 100
radius = np.linalg.norm(res, 2)
background = 25                   # Smallest brightness seen by the event sensor
speeds = [10000, 1000, 100]       # px / s
sizes = [1, 2, 3]                 # the size of the PSF is 3 x sizes
sat_mag = [12, 9, 6]              # Exponential brightness of the object
nb_runs = 1                       # Number of repeats for each option
noise_distribs = [["data/noise_pos_0.1lux.npy", "data/noise_neg_0.1lux.npy"],]
noise_names = ["3klux", "161lux", "0.1lux"]
x = np.tile(np.arange(0, res[0], 1), (res[1], 1))
y = np.tile(np.arange(0, res[1], 1), (res[0], 1)).transpose()
ed = EventDisplay("Events", res[1], res[0], 20000)

# Save params
f = open('{}/params_v2.csv'.format(dir_res), "w")
id = 0
for noise in enumerate(noise_distribs):
    for speed in speeds:
        for size in sizes:
            for m in sat_mag:
                for r in range(0, nb_runs, 1):
                    id += 1
                    f.write('{};{};{};{};{};{}\n'.format(id, speed, size, m, r, noise))
f.close()

# Generate sequences
for i, noise in enumerate(noise_distribs):
    filename = 'Noise_{}'.format(noise_names[i])
    # Initialise the DVS sensor
    dvs = DvsSensor("MySensor")
    dvs.set_shape(res[0], res[1])
    dvs.set_dvs_sensor(th_pos=th, th_neg=th, th_n=th_noise, lat=lat, tau=tau, jit=jit, bgnp=bgn, bgnn=bgn, ref=ref)
    # dvs.init_bgn_hist("./data/noise_pos_161lux.npy", "./data/noise_neg_161lux.npy")
    dvs.init_image(np.zeros((res[1], res[0])))
    time_simu = 0
    id = 0
    for speed in speeds:
        dur_simu = radius / speed * 1e6
        dt = int(1e6 / (2 * speed))
        for size in sizes:
            for m in sat_mag:
                for r in range(0, nb_runs, 1):
                    ev = EventBuffer(1)
                    # Initial position/speeds of the object
                    angle = np.random.uniform(0, 2 * np.pi)
                    x0 = radius // 2 * np.cos(angle)
                    y0 = radius // 2 * np.sin(angle)
                    dx = - speed * np.cos(angle) / 1e6 * dt
                    dy = - speed * np.sin(angle) / 1e6 * dt
                    x0 += res[0] // 2
                    y0 += res[1] // 2
                    id += 1
                    t0 = time_simu
                    for t in tqdm(range(0, int(dur_simu)+1, dt)):
                        time_simu += dt
                        x0 += dx
                        y0 += dy
                        if 0 <= x0 < res[0] and 0 <= y0 < res[1]:
                            ct = np.exp(-np.sqrt(np.power(x0 - x, 2) + np.power(y0 - y, 2)) / (size))
                        else:
                            ct = np.zeros((res[1], res[0]))
                        img = 1 + ct * np.exp((background-m)/2.5)
                        buf = dvs.update(img, dt)
                        ev.increase_ev(buf)
                    ev.sort()
                    events = np.zeros((4, ev.i), dtype=np.uint32)
                    events[0, :] = ev.get_x()
                    events[1, :] = ev.get_y()
                    events[2, :] = ev.get_p()
                    events[3, :] = ev.get_ts()
                    with h5py.File('{}/{}_event.h5'.format(dir_res, id),
                                   'w') as h5f:
                        g = h5f.create_group("events")
                        g.create_dataset("recording", compression="gzip",
                                         data=events)

                    # Compute and save GT
                    x0 = radius // 2 * np.cos(angle) + res[0] // 2
                    y0 = radius // 2 * np.sin(angle) + res[1] // 2
                    x_gt = x0 + dx / dt * (ev.get_ts() - t0)
                    y_gt = y0 + dy / dt * (ev.get_ts() - t0)
                    events_gt = np.empty(ev.i, dtype=[('x', np.float32), ('y', np.float32), ('id', np.uint16), ('ts', np.uint32)])
                    events_gt['x'] = x_gt
                    events_gt['y'] = y_gt
                    events_gt['id'][:] = id
                    events_gt['ts'] = ev.get_ts()
                    with h5py.File('{}/{}_gt.h5'.format(dir_res, id),
                                   'w') as h5f:
                        g = h5f.create_group("events_GT")
                        g.create_dataset("recording", compression="gzip",
                                         dtype=[('x', np.float32), ('y', np.float32), ('id', np.uint16), ('ts', np.uint32)],
                                         data=events_gt)




