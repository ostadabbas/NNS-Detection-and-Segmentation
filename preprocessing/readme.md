# Preprocessing Section

## 1. General description
The preprocessing section aims for three fucntions:
    
1. Trimming the long recordings (10hr+) into short video clips (e.g. 2.5 sec) for the classifier input.
2. Cropping the full scale frames in the short video clips to generate interested-area-only videos.
3. Converting all cropped videos into optical flow videos.

## 2. Prerequisites
Packages including:

* python 3.6
* opencv (`pip3 install opencv-python==4.5.5.62`; and `pip3 install opencv-contrib-python==4.5.5.62` for MOSSE tracker)  
* pandas  
* numpy  
* tqdm  
* loguru
* retinaface (`pip3 install retina-face`)
* VisualStudio 2017 (for installing python c++ component)
* Coarse2Fine Optical Flow  

```
cd preprocessing
git clone https://github.com/pathak22/pyflow.git
(p.s. comment line 9: //#define _LINUX_MAC in pyflow\src\project.h)
cd pyflow/
pip3 install cython
python3 setup.py build_ext -i
```


Folder set up for the data and the code:`   
```
+ RawData 
    + annotations    
        + R7 
            - R7_Emma.csv
    + long videos    
         - Copy of R7_sleep.mp4
+ preprocessing
    - No.1 Trimming.py
    - No.2 Cropping.py
    - util.py
```

## 3. Trimming
No.1 Trimming.py file aims to prepare 2-sec pure NNS and non-NNS video clips for the video classifiers, via the 
following steps:

1. extract starting frame index and ending frame index for all NNS and pacifier events.
2. generate a binary vector in the length of the input long video as the label template of NNS and pacifier actions.
3. calculate the intersection of the binary label vectors of NNS and pacifier actions to get the label vector of the
   NNS events within pacifier events.
4. set up short clip window size, and then calculate the starting frame index and ending frame index of the short
   clips by segmenting continuous windows on the pacifier-NNS event label vector.
5. save the video clips basing on the starting frame index and ending frame index of the short clips.


To run the trimming process, directly using the following code in the terminal:
```
python3 No.1 Trimming.py --subj R7 --window 26 --dataUsage classification
```
where `subj` indicates the subject name in the `RawData` folder, `window` represents the length of the output short 
video clips. `dataUsage` indicates the purpose of the current data preparation, either for ~ 2 sec video clip prepared 
for action recognition classifier training and evaluation, or for ~ 1 min video clips for further action segmentation 
task.

The trimmed short video clips will be saved in the `trimming results` folder under `RawData` folder:
```
+ RawData 
    - annotations    
    - long videos
    + data for classification    
        + trimming results 
            + R7
                - NNS
                - Non-NNS
                - Transition
```

## 4. Cropping
No.2 Cropping.py file aims to crop the area of interest out of the full scale short video clips generate by 
No.1 Trimming.py in step 3. The general idea is as following:

1. read the frames in the short clip, then use RetinaFace facial detector to calculate the primary facial bounding box 
for each frame.

2. use MOSSE tracker to propagate the first detected facial bounding box forward (and backward too if the first 
detected bbox is not in the first frame.) through all frames, so that the bounding box are temporally related.

3. perform data augmentation including inflation, rotation, and flipping on all bounding boxes.

4. calculate the trajectory of all bounding boxes within the video. Then apply an average kernel on the trajectory and
 apply it back to the bounding boxes to achieve video stabilization.
 
To run the cropping process, directly using the following code in the terminal:
```
python3 No.2 Cropping.py --subj R7 --action NNS --augment True --rotate False --bboxSize 550 --blackedgeCutoff 1.2
--inflation_thresh 0.3 --dataUsage classification
```
`subj` indicates the subject name in the `RawData` folder,  
`action` represents the type of action (`NNS`, `Non-NNS`, or `Transition`),  
`augment` is the data augmentation indicator (`1` for applying data augmentation and `0` for no augmentation),  
`rotate` indicates the degree to rotate the input video for the retina face detector if necessary, since retina face 
works best when the face is perfectly vertical in the frame. Possible choices including `cv2.ROTATE_90_CLOCKWISE`, 
`cv2.ROTATE_90_COUNTERCLOCKWISE`, and `cv2.ROTATE_180`.'  
`bboxSize` represents the output video size (output is a square bounding box).   
`blackedgeCutoff` is used to inflate the output during video stabilization (`1.2` represents the output is inflated by 
120%). Sometimes the stabilized video has black edge. This inflation can reduce it.'   
`inflation_thresh` increase the corner padding and output bbox size (`0.3` represents the padding dimention is `30%` of 
the output size). The direct output from retina face is too tight.
`dataUsage` is used to find the data for corresponding task.


The cropped short video clips will be saved in the `cropping results` folder under `RawData` folder:
```
+ RawData 
    - annotations    
    - long videos 
    + data for classification  
        - trimming results  
        + cropping results 
            +aug_results
                + R7
                    - NNS
                    - Non-NNS
                    - Transition
```
 ## 5. Optical Flow 
We used **Python** wrapper for Ce Liu's [C++ implementation](https://people.csail.mit.edu/celiu/OpticalFlow/) of 
Coarse2Fine Optical Flow. This is **super fast and accurate** optical flow method based on Coarse2Fine warping method 
from Thomas Brox. This python wrapper has minimal dependencies, and it also eliminates the need for C++ OpenCV library. 
For real time performance, one can additionally resize the images to a smaller size.


After install the Coarse2Fine Optical Flow following the instruction in section 2, a optical flow demo 
(`OpticalFlowDemoForVideo.py`) that takes video clips as input and output the visualization of the optical flow videos.
```
python3 OpticalFlowDemoForVideo.py --path ./RawData/cropping results/aug_reults/R7/NNS/0.mp4 --saveDir ./pyflow/examples/demoVideoResult
```
where `--path` is the directory of the input video demo, `--saveDir` is the saving path.

To process the cropped videos in batches, use the `No.3 OpticalFlow.py`:
```
python3 No. OpticalFlow.py --subj R7 --action NNS --aug aug_results --dataUsage classification
```
 where `subj`, `action` (`NNS`, `Non-NNS`, `Transition`), `aug` (`aug_results`, `noAug_results`) are the directory info 
 in the RawData folder. `dataUsage` is used to find the data for corresponding task.
The optical flow of the cropped short video clips will be saved in the `optical flow results` folder under `RawData` 
folder:
```
+ RawData 
    - annotations    
    - long videos
    + data for classification   
        - trimming results  
        - cropping results 
        + optical flow results
            +aug_results
                + R7
                    - NNS
                    - Non-NNS
                    - Transition
    + data for segmentation   
        - trimming results  
        - cropping results 
        + optical flow results
            + R7
                - 0.mp4

```