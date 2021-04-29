"""
    Generate the textures for the MNIST dataset
"""
import cv2
from mnist import MNIST
import numpy as np
from tqdm import tqdm
import os
import sys

cur_dir = os.getcwd()
sys.path.append(cur_dir + "/../../src")

type = "NMNIST_test"

path_text = "../../data/text/"
if not os.path.exists(path_text):
    os.mkdir(path_text)

if os.path.exists(path_text + "NMNIST_lab.npy"):
    print("Textures already generated, please make sure everything is archived")
else:
    mndata = MNIST("../../data/mnist/")
    images, labels = mndata.load_training() if type == "NMNIST_train" else mndata.load_testing() 
    images = np.array(images)
    labels = np.array(labels)
    p = np.random.permutation(images.shape[0])
    images = images[p, :]
    labels = labels[p]
    # Create background
    offset = 100  # Background light
    nb_images_per_texture = 100
    nb_img = images.shape[0]   
    for t in tqdm(range(0, 1, 1), desc="Texture Generation"):
        img_texture = np.full([2800, 2800], offset)
        for i in range(0, 100, 1):
            img_mnist = np.reshape(images[i + t * nb_images_per_texture, :], (28, 28))
            img_mnist = img_mnist + 100
            img_mnist[img_mnist > 255] = 255
            img_texture[img_texture.shape[0]//2-14:img_texture.shape[0]//2+14, i*28:(i+1)*28] = img_mnist[::-1, :]
        cv2.imwrite(path_text + 'img_temp_{}.png'.format(t), img_texture)

    np.save(path_text + "NMNIST_lab.npy", labels)
