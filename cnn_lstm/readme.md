# Action Recognition Model
## 1. General description
The goal of the pipeline is to train a cnn-lstm based model for classifying the NNS vs Non-NNS actions. The training 
data are the optical flow of the area-of-interest-only short videos generated from the preprocessing section.

## 2. Prerequisites
Packages including:

* python 3.6
* PyTorch (ver. 0.4+ required) 
* FFmpeg
* FFprobe  

## 3. Set up dataset folder
(Manually checking on the data generated from cropping process is recommended.)
Ues the `copyData.py` file to build up the data folder by transferring the optical flow clips from RawData folder to the 
cnn_lstm subfolder:
```
python3 copyData.py --augment Ture --testSubj R7 --Transition False
```
Where `augment` represents the type of the input data (either augmented with inflation, rotation, and flipping; or no 
data augmentation.). `test` indicates the subject that never involved while training and evaluation. `Transition` 
indicate what kinds of data are involved in the following training process (binary classes NNS vs Non-NNS, or three 
classes NNS vs Non-NNS vs Transition).

After data transferring, the data should follow the structure below:
```
+ data 
    + testFold
        + video_data
            - NNS
            - Non-NNS    
                - 0001.mp4
    + trainFold 
        + video_data
            - NNS
            - Non-NNS    
                - R1_0001.mp4
```
## 4. Data preparation for training
directly click and run the `generate_data.sh` executable file via git bash. This file will segment the videos into 
frames and generate the index file for the later training process. The data folder should generate separated frame and 
index json file as following structure:
```
+ data 
    + testFold
        - video_data
        - annotation
        - image_data
        - trainval.txt
    - trainFold 
```
## 5. Train the model
For NNS vs Non-NNS data, directly run the file `main_binaryClass_pipeline.py`:
```
python3  main_binaryClass_pipeline.py
```
parameters are within the file, the parameters can be set in the augment class within the file, the trained model weight
will be storaged in the snapshots folder naming after `spatial model _ temporal model` (e.g. `./snapshots/Res18_lstm/Res18_lstm-Epoch-18.pth`)