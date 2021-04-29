"""
    APIs to vizualize the NMNIST dataset
"""
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import cv2
import numpy as np
import os
import sys
cur_dir = os.getcwd()
sys.path.append(cur_dir + "/../../src")
from dat_files import load_dat_event
from mnist import MNIST

SIZE_DIGIT = 28


def display_img_test():
    """
        Display the test images one by one, if one day you don't know what to do
    """
    path_blender = "/home/joubertd/Documents/Soft/Blender/Saccades/Imager/blender_to_event/"
    mndata = MNIST(path_blender + "mnist/")
    images, labels = mndata.load_testing()
    images = np.array(images)
    labels = np.array(labels)
    print(labels)
    ind_4 = np.where(labels == 4)
    for i in range(0, ind_4[0].shape[0], 1):
        img = images[ind_4[0][i]].reshape((28, 28))
        print(i)
        cv2.imshow("test", img.astype(np.uint8))
        cv2.waitKey()

def getTextImg(id, path, nb_img_text=10):
    """
        Return the image presented to the sensor
        id: position in the dataset
        path: of the dataset
        nb_img_text: number of images per texture
    """
    nb_file = id // nb_img_text
    nb_sub_file = id - nb_file * nb_img_text
    sub_file_text = os.path.join(path, "text", 'img_temp_{}.png'.format(nb_file))
    sub_file_lab = os.path.join(path, "text", "NMNIST_lab.npy")
    img = cv2.imread(sub_file_text)
    img = img[::-1, :]
    labels = np.load(sub_file_lab)
    size_img = nb_img_text * SIZE_DIGIT
    plt.figure()
    plt.imshow(img[size_img//2-14:size_img//2+14, nb_sub_file*28:(nb_sub_file+1)*28])
    plt.colorbar()
    plt.title(str(labels[id]))
    plt.show()

def getSpikes(id,  path, nb_img_text=10, dataset_split=10, tw=1000, display=False):
    """
        Return the image presented to the sensor
        id: position in the dataset
        path: of the dataset
        dataset_split: number of textures per event files
        nb_img_text: number of images per texture
        display: Display video or display if True, otherwise display the integration of the events
    """
    nb_file = id // (dataset_split*nb_img_text)
    nb_sub_file = id % (dataset_split*nb_img_text)
    sub_file = os.path.join(path, "events", 'MNIST_ev_{}_{}_td.dat'.format(nb_file*10, (nb_file + 1)*10))
    sub_file_lab = os.path.join(path, "labels", 'MNIST_lab_events_{}_{}.npy'.format(nb_file*10, (nb_file + 1)*10))
    ts, x, y, p = load_dat_event(sub_file)
    labels = np.load(sub_file_lab)
    ts_start = int(labels[0, id]) # (id) * 34 * 100 #int(labels[0, id])
    ts_stop = ts_start + 29 * 1000
    img = np.zeros((28, 28), dtype=np.uint8)
    for t in range(ts_start, ts_stop, tw):
        ind = np.where((ts > t) & (ts < t + tw))
        img[:, :] = 125
        img[y[ind], x[ind]] = 125 * (1 + (p[ind]*2-1))
        if display:
            cv2.imshow("spikes", img)
            cv2.waitKey(-1)
    if not display:
        ind = np.where((ts > ts_start) & (ts < ts_stop))
        plt.figure()
        plt.plot(ts[ind], p[ind])
        plt.show()

        img[:, :] = 125
        img[y[ind], x[ind]] = 125 * (1 + (p[ind]*2-1))
        plt.figure()
        plt.imshow(img)
        plt.title(labels[1, id])
        plt.draw()

if __name__ == '__main__':
    display_img_test()