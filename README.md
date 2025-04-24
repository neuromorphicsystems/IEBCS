# IEBCS
ICNS Event Based Camera Simulator 

Available in Matlab and Python.

This repository presents the simulator used at ICNS to assess EBC. It contains:
* /cpp/: C++ implementation of the pixel simulation model and the interface Python/C++. 
* /matlab/: interface Matlab/C++
* /data/: Stores distributions used to sample the noise of the sensor and other resources. 
* /examples/: Several different ways of using the simulator (see below)'*

This Figure summarizes the differences with others tools, such as 
[ESIM](https://github.com/uzh-rpg/rpg_esim) and 
 [V2E](https://github.com/SensorsINI/v2e):

![alt text](data/img/schema_framework.png)

## -- Requirements -- 

Tested on MacOS and Windows with Python 3.10, but there is no reason it should not work on Linux.


Create a virtual environment using conda in MacOS:
```
conda env create -f simuDVSICNS_MacOS.yml 
conda activate IEBCS
```


Create a virtual environment using conda in Linux:
```
conda env create -f simuDVSICNS_Linux.yml 
conda activate IEBCS
```

Create a virtual environment using conda in Windows:
```
conda env create -f simuDVSICNS_Windows.yml 
conda activate IEBCS
```

To use the C++ interface, run compile_test.sh to install and test the Python/C++ interface.
```
cd cpp
./compile_test.sh
```

This uses the python build module, which can also be installed with pip:
```
pip install build
```

Additional requirements: 
* pytube: example 00 downloads a video from youtube.
* Blender: examples 01 and 02 are using [Blender 3.6](https://pypi.org/project/bpy/), which can be installed with pip.
* MNIST: example 02 uses the MNIST dataset, the [specific package](https://pypi.org/project/python-mnist/) that is used in this case can be installed using pip.
```
conda install -c conda-forge pytube
pip install bpy
pip install python-mnist
```

## -- Examples --

### 00: Video -> events

Simulate events from a video. To initialize the C++ sensor:
```
dsi.initSimu(cap.get(cv2.CAP_PROP_FRAME_HEIGHT), cap.get(cv2.CAP_PROP_FRAME_WIDTH))
dsi.initLatency(200, 50, 50, 300)
dsi.initContrast(0.3, 0.3, 0.05)
init_bgn_hist_cpp("../../data/noise_pos_161lux.npy", "../../data/noise_pos_161lux.npy")
```
The first line initialize the definition of the sensor, then:  
* latency = 200 μs   
* jitter = 50 μs  
* refractory period = 50 μs  
* time constant log front-end = 300 μs
* positive/negative log threshold = 0.3  
* threshold mismatch = 0.05  
* The noise is sampled from 2 distributions acquired with a real sensor under 161lux.
<img src="data/img/aps_00.gif" alt="drawing" width="300"/>
<img src="data/img/ev_00.gif" alt="drawing" width="300"/>

The artifacts are due to the low framerate compared to the speed of the wings. This can be compensated using slow motion estimations as done in V2E, provided that the estimation is not adding noise. 

### 01: Blender spinning cube

This example shows how to render a camera rotating in front of a textured cube. In this case, the object is rendered every ms. In this case, the cheese textured is a Fourme d'Ambert (blue cheese).
0_example_spinning_cube.py runs the experiment, generates event based data, and saves the result to disk. 1_make_video.py generates a visualisation of the event based data.

<img src="data/img/aps_01.gif" alt="drawing" width="300"/>
<img src="data/img/ev_01.gif" alt="drawing" width="300"/>


### 02: Blender custom NMNIST

The NMNIST dataset is simulated (for 10 digits here). 
The script "0_generate_textures_MNIST.py" create the texture used in blender to generate the saccadic movement: 
And the script "1_saccades_NMNIST.py" generates the images from Blender and runs the simulator. 2_make_video.py generates a visualisation of the event based data. Note that compared to the previous example, all the positions of the camera are computed first and then Blender is called to render this sequence faster. 
Finally, the script "nmnist_util.py" provides some API to read the spikes and the labels if you want to use your favorite ML framework.

<img src="data/img/aps_02.gif" alt="drawing" width="50"/>
<img src="data/img/ev_02.gif" alt="drawing" width="50"/>

### 03: Tracking dataset

This script generates a tracking dataset with object of various size and contrasts.  
<img src="data/img/ev_03.gif" alt="drawing" width="200"/> 

Every object has its own spike and ground truth file.
