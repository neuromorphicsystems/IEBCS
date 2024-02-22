import numpy as np
import pandas as pd
import struct
import matplotlib.pyplot as plt
import cv2
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.font_manager import FontProperties
from datetime import datetime
import loris

fontP = FontProperties()
fontP.set_size('small')


def load_dat_event(filename, start=0, stop=-1, display=False):
    """ Load .dat event.
        Warning: only tested on the VGA sensor on V2 Prophesee event
        Args:
            filename: Path of the .dat file
            start: starting timestamp (us)
            stop: if different than -1, last timestamp
            display: display file infos
        Returns:
             ts, x, y, pol numpy arrays of timestamps, position and polarities
     """
    f = open(filename, 'rb')
    if f == -1:
        print("The file does not exist")
        return
    else:
        if display: print("Load DAT Events: " + filename)
    l = f.readline()
    all_lines = l
    while l[0] == 37:
        p = f.tell()
        if display: print(l)
        l = f.readline()
        all_lines = all_lines + l
    f.close()
    all_lines = str(all_lines)
    f = open(filename, 'rb')
    f.seek(p, 0)
    evType = np.uint8(f.read(1)[0])
    evSize = np.uint8(f.read(1)[0])
    p = f.tell()
    l_last = f.tell()
    if start > 0:
        t = np.uint32(struct.unpack("<I", bytearray(f.read(4)))[0])
        dat = np.uint32(struct.unpack("<I", bytearray(f.read(4)))[0])
        while t < start:
            p = f.tell()
            t = np.uint32(struct.unpack("<I", bytearray(f.read(4)))[0])
            dat = np.uint32(struct.unpack("<I", bytearray(f.read(4)))[0])

    if stop > 0:
        t = np.uint32(struct.unpack("<I", bytearray(f.read(4)))[0])
        dat = np.uint32(struct.unpack("<I", bytearray(f.read(4)))[0])
        while t < stop:
            l_last = f.tell()
            t = np.uint32(struct.unpack("<I", bytearray(f.read(4)))[0])
            dat = np.uint32(struct.unpack("<I", bytearray(f.read(4)))[0])
    else:
        l_last = f.seek(0, 2)

    num_b = (l_last - p) // evSize * 2
    f.close()
    data = np.fromfile(filename, dtype=np.uint32, count=num_b, offset=p)
    ts = data[::2]
    v = 0
    ind = all_lines.find("Version")
    if ind > 0:
        v = int(all_lines[ind+8])
    if v >= 2:
        x_mask = np.uint32(0x00007FF)
        y_mask = np.uint32(0x0FFFC000)
        pol_mask = np.uint32(0x10000000)
        x_shift = 0
        y_shift = 14
        pol_shift = 28
    else:
        x_mask = np.uint32(0x00001FF)
        y_mask = np.uint32(0x0001FE00)
        pol_mask = np.uint32(0x00020000)
        x_shift = 0
        y_shift = 9
        pol_shift = 17
    x = data[1::2] & x_mask
    x = x >> x_shift
    y = data[1::2] & y_mask
    y = y >> y_shift
    pol = data[1::2] & pol_mask
    pol = pol >> pol_shift
    if len(ts) > 0:
        if display: 
            print("First Event: ", ts[0], " us")
            print("Last Event: ", ts[-1], " us")
            print("Number of Events: ", ts.shape[0])
    return ts, x, y, pol


def write_event_dat(filename, ts, x, y, pol,
                    event_type='dvs', width=None, height=None):
    """ Write the events in a .DAT file
        The file header begins with %, then event type (one byte uint8) and event lenght
        (one byte uint8), then the data are stores ts (4 bytes uint32) and x-y-pol (4 bytes uint32)
        Args:
            filename: path of the file to create
            ts: stimestamp
            x, y: positions of the pixels
            p: polarities (0 or 1)
    """
    f = open(filename, 'wb')
    if f == -1:
        print("Impossible to open the file")
        return
    if event_type in ['dvs', 'cd', 'td']:
        f.write(bytes("% Data file containing CD events.\n", encoding='utf8'))
    elif event_type in ['aps', 'em']:
        f.write(bytes("% Data file containing EM events.\n", encoding='utf8'))
    else:
        raise Exception("Specify a valid event type: 'dvs', 'cd', 'td', 'aps' or 'em'")

    f.write(bytes("% Version 2\n", encoding='utf8'))
    f.write(bytes("% Date " + str(datetime.now().replace(microsecond=0)) + '\n', encoding='utf8'))

    if width is None:
        width = x.max() + 1
    if height is None:
        height = y.max() + 1
    f.write(bytes("% Height " + str(height) + '\n', encoding='utf8'))
    f.write(bytes("% Width " + str(width) + '\n', encoding='utf8'))

    f.write(bytes(np.uint8([0])))  # Event Type
    f.write(bytes(np.uint8([8])))  # Event length
    arr = np.zeros(2 * ts.shape[0], dtype=np.uint32)
    arr[::2] = ts
    x_mask = np.uint32(0x00007FF)
    y_mask = np.uint32(0x0FFFC000)
    pol_mask = np.uint32(0x10000000)
    x_shift = 0
    y_shift = 14
    pol_shift = 28
    buf = np.array(x, dtype=np.uint32) << x_shift
    arr[1::2] += x_mask & buf
    buf = np.array(y, dtype=np.uint32) << y_shift
    arr[1::2] += y_mask & buf
    buf = np.array(pol, dtype=np.uint32) << pol_shift
    arr[1::2] += pol_mask & buf
    arr.tofile(f)
    f.close()


def write_event_es(filename, ts, x, y, pol,
                   event_type='dvs', width=None, height=None):
    """ Write the events in a .es file
        Args:
            filename: path of the file to create
            ts: stimestamp
            x, y: positions of the pixels
            p: polarities (0 or 1)
    """
    events = np.recarray(shape=ts.shape,
                         dtype=[('t', '<u8'), ('x', '<u2'),
                                ('y', '<u2'), ('is_increase', '?')])
    events['t'] = ts
    events['x'] = x
    events['y'] = y
    events['is_increase'] = pol
    if width is None:
        width = x.max() + 1
    if height is None:
        height = y.max() + 1
    file_dict = {'type': event_type, 'width': int(width), 'height': int(height), 'events': events}
    loris.write_events_to_file(file_dict, filename)


def write_event_csv(filename, ts, x, y, pol):
    """ Write the events in a .csv file"""
    df = pd.DataFrame({'ts': ts,
                       'x': x,
                       'y': y,
                       'p': pol})

    df = df.astype({'ts': '<u8', 'x': '<u2', 'y': '<u2', 'p': '?'})
    df.to_csv(filename, header=True, index=False)


def view_event(ts, x, y, pol, min_x=0, max_x=640, min_y=0, max_y=480, t_min=0, t_max=-1):
    """ Visualise events in a 3d space
        Time scale is in ms
        Args:
            ts, x, y, pol: events
            min_y, max_y: Keeps events in [min_y max_y]
            min_x, max_x: Keeps events in [min_x max_x]
            t_min, t_max: Keeps events in [t_min t_max]
    """
    ind_pos = np.where(
        (pol == 1) & (y > min_y) & (y < max_y) & (x > min_x) & (x < max_x) & (ts > t_min) & (ts < t_max))
    ind_neg = np.where(
        (pol == 0) & (y > min_y) & (y < max_y) & (x > min_x) & (x < max_x) & (ts > t_min) & (ts < t_max))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x[ind_pos[0][:]], y[ind_pos[0][:]], ts[ind_pos[0][:]] / 1000, c='r', alpha=0.5, s=1)
    ax.scatter(x[ind_neg[0][:]], y[ind_neg[0][:]], ts[ind_neg[0][:]] / 1000, c='g', alpha=0.5, s=1)
    ax.set_xlabel('x (px)')
    ax.set_ylabel('y (px) ')
    #ax.set_zlabel(r' time ($\mu$s)')
    ax.set_zlabel(r' time (ms)')
    ax.set_xlim3d(min_x, max_x)
    ax.set_ylim3d(min_y, max_y)
    ax.set_zlim3d(t_min / 1000, t_max / 1000)
    # ax.set_xticks(np.arange(min_y, max_y, 1))
    # ax.set_yticks(np.arange(min_x, max_x, 1))
    ax.set_proj_type('ortho')
    plt.show()

def make_video_event(ts, x, y, pol, min_x=0, max_x=640, min_y=0, max_y=480, t_min=0, t_max=-1, dt=100, filename="text.avi"):
    """ Create a video of the events in a 3d space
        Time scale is in ms
        Args:
            ts, x, y, pol: events
            min_y, max_y: Keeps events in [min_y max_y]
            min_x, max_x: Keeps events in [min_x max_x]
            t_min, t_max: Keeps events in [t_min t_max]
    """
    fig = plt.figure()

    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    out = cv2.VideoWriter(filename, fourcc, 10.0, (640, 480))
    for t in range(int(t_min), int(t_max), int(dt)):
        plt.clf()
        ax = fig.add_subplot(111, projection='3d')
        ind_pos = np.where(
            (pol == 1) & (y > min_y) & (y < max_y) & (x > min_x) & (x < max_x) & (ts > t_min) & (ts < t))
        ind_neg = np.where(
            (pol == 0) & (y > min_y) & (y < max_y) & (x > min_x) & (x < max_x) & (ts > t_min) & (ts < t))

        ax.scatter(x[ind_pos[0][:]], y[ind_pos[0][:]], ts[ind_pos[0][:]] / 1000, c='r', alpha=0.5, s=1)
        ax.scatter(x[ind_neg[0][:]], y[ind_neg[0][:]], ts[ind_neg[0][:]] / 1000, c='g', alpha=0.5, s=1)
        ax.set_xlabel('x (px)')
        ax.set_ylabel('y (px) ')
        #ax.set_zlabel(r' time ($\mu$s)')
        ax.set_zlabel(r' time (ms)')
        ax.set_xlim3d(min_x, max_x)
        ax.set_ylim3d(min_y, max_y)
        ax.set_zlim3d(t_min / 1000, t_max / 1000)
        # ax.set_xticks(np.arange(min_y, max_y, 1))
        # ax.set_yticks(np.arange(min_x, max_x, 1))
        ax.set_proj_type('ortho')
        plt.draw()
        filename = './buf.png'
        plt.savefig(filename)
        im1 = cv2.imread(filename)
        out.write(im1)
    out.release()

