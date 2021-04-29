# Damien JOUBERT 17-01-2020
import numpy as np
import os.path
from dat_files import write_event_dat, write_event_es, write_event_csv, load_dat_event


class EventBuffer():
    """ Structure to handle a buffer of dvs events """
    x = 0  # Array of x values
    y = 0  # Array of y values
    ts = 0  # Array of timestamps values (us)
    p = 0  # Array of polarity values (0 negative, 1 positive)
    i = 0  # Position of the next event

    def __init__(self, size):
        """ Resize the buffers
            Args:
                size: size of the new buffer, Minimum: 1
        """
        if isinstance(size, str):
            ts, x, y, pol = load_dat_event(size)
            self.ts = np.array(ts, dtype=np.uint64)
            self.x = np.array(x, dtype=np.uint16)
            self.y = np.array(y, dtype=np.uint16)
            self.p = np.array(pol, dtype=np.uint8)
            self.i = ts.shape[0]
        else:
            if size == 0:
                size = 1
            self.x = np.zeros(size, dtype=np.uint16)
            self.y = np.zeros(size, dtype=np.uint16)
            self.p = np.zeros(size, dtype=np.uint8)
            self.ts = np.zeros(size, dtype=np.uint64)

    def get_x(self):
        return self.x[:self.i]

    def get_y(self):
        return self.y[:self.i]

    def get_p(self):
        return self.p[:self.i]

    def get_ts(self):
        return self.ts[:self.i]

    def increase(self, nsize):
        """ Increase the size of a buffer to self.shape[0] size + nsize
            Args:
                nsize: number of free space added
        """
        prev_shape = self.x.shape[0]
        x = np.zeros(prev_shape + nsize, dtype=np.uint16)
        y = np.zeros(prev_shape + nsize, dtype=np.uint16)
        p = np.zeros(prev_shape + nsize, dtype=np.uint8)
        ts = np.zeros(prev_shape + nsize, dtype=np.uint64)
        x[:prev_shape] = self.x
        y[:prev_shape] = self.y
        p[:prev_shape] = self.p
        ts[:prev_shape] = self.ts
        self.x = x
        self.y = y
        self.p = p
        self.ts = ts

    def remove_time(self, t_min, t_max):
        ind = np.where((self.ts < t_min)|(self.ts > t_max))
        self.x = np.delete(self.x, ind)
        self.y = np.delete(self.y, ind)
        self.ts = np.delete(self.ts, ind)
        self.p = np.delete(self.p, ind)
        self.i = self.ts.shape[0]

    def remove_elt(self, nsize):
        """
            Remove the nsize first elements
        """
        if self.i - nsize < 0:
            nsize = self.i
        ind = np.arange(0, nsize, 1)
        self.i = self.i-nsize
        self.x = np.delete(self.x, ind)
        self.y = np.delete(self.y, ind)
        self.ts = np.delete(self.ts, ind)
        self.p = np.delete(self.p, ind)

    def remove_ev(self, p):
        """
            Remove the event at the position p
        """
        if self.i <= p:
            return
        self.x = np.delete(self.x, p)
        self.y = np.delete(self.y, p)
        self.ts = np.delete(self.ts, p)
        self.p = np.delete(self.p, p)

    def remove_row(self, r, t):
        """
            Remove the event in the row r at the time t
        """
        if t == -1:
            ind = np.where((self.y == r) & (self.ts > 0))
        else:
            ind = np.where((self.y == r) & (self.ts < t) & (self.ts > 0))
        self.i -= ind[0].shape[0]
        self.x = np.delete(self.x, ind)
        self.y = np.delete(self.y, ind)
        self.ts = np.delete(self.ts, ind)
        self.p = np.delete(self.p, ind)

    def increase_ev(self, ev):
        """ Increase the event buffer with anotehr event buffer
            If ev can be inserted into self, ev inserted, if not, increase the size of a buffer to original
            self.shape[0] + ev.shape[0]
            Args:
                ev: the EventBuffer added
            """
        if len(self.x) > 0 and not ev is None:
            if self.i + ev.x.shape[0] > self.x.shape[0] - 1:
                prev_shape = self.x.shape[0]
                x = np.zeros(prev_shape + ev.ts.shape[0], dtype=np.uint16)
                y = np.zeros(prev_shape + ev.ts.shape[0], dtype=np.uint16)
                p = np.zeros(prev_shape + ev.ts.shape[0], dtype=np.uint8)
                ts = np.zeros(prev_shape + ev.ts.shape[0], dtype=np.uint64)
                x[:self.i] = self.x[:self.i]
                y[:self.i] = self.y[:self.i]
                p[:self.i] = self.p[:self.i]
                ts[:self.i] = self.ts[:self.i]
                x[self.i:self.i + ev.x.shape[0]] = ev.x
                y[self.i:self.i + ev.x.shape[0]] = ev.y
                p[self.i:self.i + ev.x.shape[0]] = ev.p
                ts[self.i:self.i + ev.x.shape[0]] = ev.ts
                self.x = x
                self.y = y
                self.p = p
                self.ts = ts
            else:
                self.x[self.i:self.i + ev.i] = ev.x[:ev.i]
                self.y[self.i:self.i + ev.i] = ev.y[:ev.i]
                self.p[self.i:self.i + ev.i] = ev.p[:ev.i]
                self.ts[self.i:self.i + ev.i] = ev.ts[:ev.i]
            self.i += ev.i

    def copy(self, i1, ep, i2):
        """ Copy the i2 th event of the EventBuffer ep in to the i1 th position
            Args:
                i1: self will have a new event in i1
                ep: EventBuffer where the event comes from
                i2: i2th event from ep is takem
         """
        if i1 < len(self.x):
            self.x[i1] = ep.x[i2]
            self.y[i1] = ep.y[i2]
            self.ts[i1] = ep.ts[i2]
            self.p[i1] = ep.p[i2]
            self.i = i1 + 1

    def merge(self, ep1, ep2):
        """ Resize tje EventBuffer and merge into the two EventBuffer ep1 nd ep2, sorted them with their timestamps
            Args:
                ep1, ep2: eventBuffer
        """
        self.__init__(len(ep1.x) + len(ep2.x))
        i1 = 0
        i2 = 0
        for j in range(0, ep1.i + ep2.i, 1):
            if i1 == ep1.i:
                self.copy(j, ep2, i2)
                i2 += 1
            elif i2 == ep2.i:
                self.copy(j, ep1, i1)
                i1 += 1
            else:
                if ep1.ts[i1] < ep2.ts[i2]:
                    self.copy(j, ep1, i1)
                    i1 += 1
                else:
                    self.copy(j, ep2, i2)
                    i2 += 1
        self.i = ep1.i + ep2.i

    def sort(self):
        """ Sort the EventBuffer according to its timestamp """
        ind = np.argsort(self.ts[:self.i])
        self.ts[:self.i] = self.ts[:self.i][ind]
        self.x[:self.i] = self.x[:self.i][ind]
        self.y[:self.i] = self.y[:self.i][ind]
        self.p[:self.i] = self.p[:self.i][ind]

    def add(self, ts, y, x, p):
        """
            Add an event (ts, x, y, p) the the EventBuffer (push strategy)
            If y == -1, if means that x[0} contains the x position and x[1] the y's one
            Args:
                ts, y, x, p: new event array
        """
        if self.x.shape[0] == self.i:
            self.increase(1000)
            self.add(ts, y, x, p)
        else:
            self.ts[self.i] = ts
            self.x[self.i] = x
            self.y[self.i] = y
            self.p[self.i] = p
            self.i += 1

    def add_array(self, ts, y, x, p, inc=1000):
        """
            Add n event (ts, x, y, p) the the EventBuffer (push strategy)
            Args:
                ts, y, x, p: new event array
                inc: increment size
        """
        s = len(ts)
        if s > len(self.ts) - self.i:
            self.increase(inc)
            self.add_array(ts, y, x, p)
        else:
            self.ts[self.i:self.i+s] = ts
            self.x[self.i:self.i+s] = x
            self.y[self.i:self.i+s] = y
            self.p[self.i:self.i+s] = p
            self.i += s

    def write(self, filename, event_type='dvs', width=None, height=None):
        """ Write the events into a .dat or .es file
            Args:
                filename: path of the file
        """
        # sort events to have a monotonically timestamps
        self.sort()

        ext = os.path.splitext(filename)[1]
        if ext == '.dat':
            write_event_dat(filename, self.ts[:self.i], self.x[:self.i],
                            self.y[:self.i], self.p[:self.i],
                            event_type=event_type, width=width, height=height)
        elif ext == '.es':
            write_event_es(filename, self.ts[:self.i], self.x[:self.i],
                           self.y[:self.i], self.p[:self.i],
                           event_type=event_type, width=width, height=height)
        elif ext == '.csv':
            write_event_csv(filename, self.ts[:self.i], self.x[:self.i],
                            self.y[:self.i], self.p[:self.i])
