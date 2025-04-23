# IEBCS
ICNS Event Based Camera Simulator 

The code base has been recently cut down to a maintainable reference python implementation. For now, if you want the blender interoperability or the C++ implementation, download a previous version of this repository at your own risk.

This repository contains:
* /data/: Stores distributions used to sample the noise of the sensor and other resources. 
* /examples/: An example of loading frames, and passing them as input to the simulator.

This Figure summarizes the differences with others tools, such as 
[ESIM](https://github.com/uzh-rpg/rpg_esim) and 
 [V2E](https://github.com/SensorsINI/v2e):

![alt text](data/img/schema_framework.png)

## -- Requirements -- 

Tested with Python 3.12.

Create a virtual environment using conda:
```
pip install opencv-python
pip install tqdm
```

Additional requirements: 
* yt-dlp: example 00 downloads a video from youtube.
```
pip install yt-dlp
```

## -- Examples --

### 00: Video -> events

Simulate events from a video.

The example sets up a sensor with the following parameters:  
* latency = 100 μs   
* jitter = 10 μs  
* refractory period = 100 μs  
* time constant log front-end = 40 μs
* positive/negative log threshold = 0.4  
* threshold noise = 0.01  
* The noise is sampled from 2 distributions acquired with a real sensor under 161lux.
<img src="data/img/aps_00.gif" alt="drawing" width="300"/>
<img src="data/img/ev_00.gif" alt="drawing" width="300"/>

The artifacts are due to the low framerate compared to the speed of the wings. This can be improved by using high frame rate video, or by generating intermediate frames.

To use, first make sure your working directory is:
```
examples/00_video_to_events
```

Then run the python files contained within this directory in order.
A video showing events, as well as the event file in binary format, should be generated within the `./output` directory.