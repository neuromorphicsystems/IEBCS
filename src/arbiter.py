import numpy as np
import cv2
from event_buffer import EventBuffer


class BottleNeckArbiter():
    t_per_event = 0.1        # Time spent to process one event (us)
    ev_acc = EventBuffer(0)  # Events accumulated
    time = 0                 # Current time (us)

    def __init__(self, t_per_event, time):
        """ Init the sensor by creating the Blender Camera
        Args:
           t_per_event: # Time spent to process one event (us)
           time: starting time (us)
        """
        self.t_per_event = t_per_event
        self.time = time

    def process(self, new_ev, dt):
        """
        Args:
            new_event: incomming events as EventBuffer
            dt: time since the last update (us)
        """
        tps_process = float(self.t_per_event) * (self.ev_acc.i + new_ev.i)
        self.time = self.time + dt
        release_ev = EventBuffer(0)
        if tps_process == 0:
            return release_ev
        self.ev_acc.increase_ev(new_ev)
        nb_event_pross = min(self.ev_acc.i, int(dt / tps_process))
        delta = 0
        if self.ev_acc.ts[0] < self.time - dt:
            delta = self.time - dt - self.ev_acc.ts[0]
        release_ev.add_array(self.ev_acc.ts[:nb_event_pross] + delta + tps_process * np.arange(0, nb_event_pross, 1),
                             self.ev_acc.y[:nb_event_pross],
                             self.ev_acc.x[:nb_event_pross],
                             self.ev_acc.p[:nb_event_pross])
        self.ev_acc.remove_elt(nb_event_pross)
        return release_ev


class RowArbiter():
    t_per_event = 0.1        # Time spent to process one event (us)
    ev_acc = EventBuffer(0)  # Events accumulated
    time = 0                 # Current time (us)
    def __init__(self, t_per_event, time):
        """ Init the sensor by creating the Blender Camera
        Args:
           t_per_event: # Time spent to process one event (us)
           time: starting time (us)
        """
        self.t_per_event = t_per_event
        self.time = time

    def process(self, new_ev, dt):
        """
        Args:
            new_event: incomming events as EventBuffer
            dt: time since the last update (us)
        """
        tps_process = float(self.t_per_event) * (self.ev_acc.i + new_ev.i)
        self.time = self.time + dt
        release_ev = EventBuffer(0)
        if tps_process == 0:
            return release_ev
        self.ev_acc.increase_ev(new_ev)
        nb_event_pross = int(dt / tps_process)
        i = 0
        delta = 0
        if self.ev_acc.ts[0] < self.time - dt:
            delta = self.time - dt - self.ev_acc.ts[0]
        while self.ev_acc.i > 0 and self.ev_acc.ts[0] <= self.time and nb_event_pross > i:
            i += 1
            ind = np.where((self.ev_acc.y == self.ev_acc.y[0])&(self.ev_acc.ts <= self.time))
            ts_inter = np.full(ind[0].shape, self.ev_acc.ts[0] + delta + tps_process * i)
            release_ev.add_array(ts_inter, self.ev_acc.y[ind], self.ev_acc.x[ind], self.ev_acc.p[ind])
            self.ev_acc.remove_row(self.ev_acc.y[0], -1)
        return release_ev


class SynchonousArbiter():
    clock_period = 0.001     # Clock's period (us)
    ev_acc = EventBuffer(0)  # Events accumulated
    time = 0                 # Current time (us)
    cur_row = 0              # current row processed
    max_row= 200             # Number of rows

    def __init__(self, max_row, clock_period, time):
        """ Init the sensor by creating the Blender Camera
        Args:
           t_per_event: # Time spent to process one event (us)
           time: starting time (us)
        """
        self.clock_period = clock_period
        self.time = time
        self.cur_row = 0
        self.max_row = max_row

    def process(self, new_ev, dt):
        """
        Args:
            new_event: incomming events as EventBuffer
            dt: time since the last update (us)
        """
        nb_row_processed = int(dt // self.clock_period)
        t_max = self.time + dt
        release_ev = EventBuffer(0)
        self.ev_acc.increase_ev(new_ev)
        for i in range(0, nb_row_processed, 1):
            self.time = self.time + self.clock_period
            self.cur_row = (self.cur_row + 1) % self.max_row
            ind = np.where((self.ev_acc.y[:self.ev_acc.i] == self.cur_row)&(self.ev_acc.ts[:self.ev_acc.i] < self.time))
            if len(ind[0]) > 0:
                ts_inter = np.full(ind[0].shape, self.time)
                release_ev.add_array(ts_inter, self.ev_acc.y[:self.ev_acc.i][ind], self.ev_acc.x[:self.ev_acc.i][ind],
                                   self.ev_acc.p[:self.ev_acc.i][ind])
                self.ev_acc.remove_row(self.cur_row, self.time)
            if self.ev_acc.i == 0:
                break
        self.time = t_max
        return release_ev

