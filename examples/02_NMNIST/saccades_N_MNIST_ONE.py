"""
    Simulate saccadic movement along several axis directions in front of One NMNIST digit
"""
import bpy
import mathutils
import cv2
import math
import numpy as np
import os
import sys
cur_dir = os.getcwd()
sys.path.append(cur_dir + "/../../src")
from dvs_sensor_blender import Blender_DvsSensor
from event_display import EventDisplay
from event_buffer import EventBuffer
from tqdm import tqdm

path_NMNIST = os.path.abspath(os.getcwd()) + "/../../data/"

if not os.path.exists(path_NMNIST + "text"):
    os.mkdir(path_NMNIST + "text")
if not os.path.exists(path_NMNIST + "labels"):
    os.mkdir(path_NMNIST + "labels")
if not os.path.exists(path_NMNIST + "tmp"):
    os.mkdir(path_NMNIST + "tmp")
if not os.path.exists(path_NMNIST + "events"):
    os.mkdir(path_NMNIST + "events")

text_start = 0
text_end = 1
print(path_NMNIST + 'tmp/tmp_{}_{}'.format(text_start, text_end))
if not os.path.exists(path_NMNIST + 'tmp/tmp_{}_{}'.format(text_start, text_end)):
    os.mkdir(path_NMNIST + 'tmp/tmp_{}_{}'.format(text_start, text_end))

scene = bpy.context.scene
bpy.ops.object.select_all(action='DESELECT')
bpy.data.objects['Camera'].select = True
bpy.ops.object.delete()
bpy.data.objects['Cube'].select = True
bpy.ops.object.delete()
bpy.data.objects['Lamp'].location = mathutils.Vector((5, 5, 5))
bpy.data.objects['Lamp'].data.energy = 10
bpy.data.objects['Lamp'].data.type = 'HEMI'

# Create background
offset = 100  # Background light
bpy.ops.mesh.primitive_plane_add(radius=10.0, calc_uvs=True, view_align=False, enter_editmode=False,
                                 location=(-10.0 + text_start * 20, 0.0, 5.0), rotation=(0, 0, 0))
pl = bpy.context.active_object
pl.name = 'PlaneSky'
mat = bpy.data.materials.new('Sky')
tx = bpy.data.textures.new('SkyMap', 'IMAGE')
tx.extension = 'REPEAT'
img_texture = np.full([2800, 2800], 100)
cv2.imwrite(path_NMNIST + "img_text.png", img_texture)
im = bpy.data.images.load(path_NMNIST + "img_text.png")
bpy.data.textures['SkyMap'].image = im
bpy.data.materials['Sky'].texture_slots.add()
bpy.data.materials['Sky'].texture_slots[0].texture = tx
pl.data.materials.append(bpy.data.materials['Sky'])

# Generate the annimation for a given sequence
t = 0
bpy.ops.mesh.primitive_plane_add(radius=10.0, calc_uvs=True, view_align=False, enter_editmode=False,
                                 location=(10 + 20.0 * t, 0.0, 5.0), rotation=(0, 0, 0))
pl = bpy.context.active_object
pl.name = 'PlaneSky_{}'.format(t)
mat = bpy.data.materials.new('Sky_{}'.format(t))
tx = bpy.data.textures.new('SkyMap_{}'.format(t), 'IMAGE')
tx.extension = 'REPEAT'
im = bpy.data.images.load(path_NMNIST + 'text/img_temp_{}.png'.format(t))
bpy.data.textures['SkyMap_{}'.format(t)].image = im
bpy.data.materials['Sky_{}'.format(t)].texture_slots.add()
bpy.data.materials['Sky_{}'.format(t)].texture_slots[0].texture = tx
pl.data.materials.append(bpy.data.materials['Sky_{}'.format(t)])

# Create the camera
ppsee = Blender_DvsSensor("PropheseeSensor")
ppsee.set_shape(28, 28)
ppsee.set_sensor(28, 28, 0.015)
ppsee.set_dvs_sensor(th_pos=0.15, th_neg=0.15, th_n=0.05, lat=100, tau=300, jit=30, bgn=0.1)
ppsee.init_tension()
ppsee.init_bgn_hist("../../data/noise_pos_0.1lux.npy", "../../data/noise_neg_0.1lux.npy")
ppsee.set_sensor_optics(20)
scene.objects.link(ppsee.cam)
scene.camera = ppsee.cam
ppsee.set_angle([math.pi, 0, 0.0])
ppsee.set_position([0.0, 0.0, 0.0])
ppsee.init_tension()

scene.render.image_settings.file_format = 'PNG'
scene.render.filepath = path_NMNIST + 'tmp/tmp_{}_{}/image_annim'.format(text_start, text_end)
scene.render.resolution_x = 2 * ppsee.def_x
scene.render.resolution_y = 2 * ppsee.def_y
#scene.render.engine = 'CYCLES'

ed = EventDisplay("Events", 28, 28, 2000)
ed.display_time = False
ev = EventBuffer(ppsee.def_x * ppsee.def_y)

# Saccade viwed as a state machine
phase_saccade = 0
size_digit = 0.1
off_y = 0.02
pos = np.array([size_digit*0.5 + 2 * size_digit * text_start, off_y, 0.0])  # x, y
speed = 0.01
t = 0
pos_img = 0
dt = 1000  # us
labels = np.load(path_NMNIST + "text/NMNIST_lab.npy")
lab = np.zeros((2, len(labels)))  # [ ts, lab]
bpy.context.scene.frame_set(pos_img)
ppsee.set_position([pos[0], pos[1], pos[2]])
# Constants
nb_img_hory = int(size_digit / (speed))
nb_img_diag = int(size_digit / (speed * np.sqrt(2)))
off = np.zeros((4, 3))
off[0, :] = np.array([size_digit, 0.0, 0.0])
off[1, :] = np.array([-size_digit / 2, -size_digit / 2, 0.0])
off[2, :] = np.array([-size_digit / 2, size_digit / 2, 0.0])
off[3, :] = np.array([size_digit * 2, 0.0, 0.0])
for nb_digit in tqdm(range(0, 10, 1)):
    lab[0, nb_digit] = t
    lab[1, nb_digit] = labels[nb_digit]
    # Saccade state machine
    # 0
    pos = pos + off[0, :]
    pos_img = pos_img + nb_img_hory
    bpy.context.scene.frame_set(pos_img)
    ppsee.set_position([pos[0], pos[1], pos[2]])
    ppsee.cam.keyframe_insert(data_path="location", index=-1)
    # 1
    pos = pos + off[1, :]
    pos_img = pos_img + nb_img_diag
    bpy.context.scene.frame_set(pos_img)
    ppsee.set_position([pos[0], pos[1], pos[2]])
    ppsee.cam.keyframe_insert(data_path="location", index=-1)
    # 2
    pos = pos + off[2, :]
    pos_img = pos_img + nb_img_diag
    bpy.context.scene.frame_set(pos_img)
    ppsee.set_position([pos[0], pos[1], pos[2]])
    ppsee.cam.keyframe_insert(data_path="location", index=-1)
    # 3
    pos = pos + off[3, :]
    pos_img = pos_img + nb_img_hory
    bpy.context.scene.frame_set(pos_img)
    ppsee.set_position([pos[0], pos[1], pos[2]])
    ppsee.cam.keyframe_insert(data_path="location", index=-1)
    t = t + dt * 2 * (nb_img_diag + nb_img_hory)

bpy.context.scene.frame_end = pos_img
bpy.context.scene.frame_current = bpy.context.scene.frame_start
bpy.ops.render.render(animation=True)

# Events
t = 0
fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
out = cv2.VideoWriter('nmnist.avi', fourcc, 20.0, (ppsee.def_x, ppsee.def_y))
for i in range(1, pos_img, 1):
    t = t + dt
    filename = path_NMNIST + 'tmp/tmp_{}_{}/'.format(text_start, text_end) + 'image_annim{0:04}.png'.format(i)
    im = cv2.imread(filename)
    out.write(im)
    if t == dt:
        ppsee.init_image(im)
    else:
        ev_buf = ppsee.update(im, dt)
        ev.increase_ev(ev_buf)
        ed.update(ev_buf, dt)
    cv2.imshow("Blender", im)
    cv2.waitKey(10)
out.release()
ev.write(path_NMNIST + 'events/MNIST_ev_{}_{}_td.dat'.format(text_start, text_end))
np.save(path_NMNIST + 'labels/MNIST_lab_events_{}_{}.npy'.format(text_start, text_end), lab)
