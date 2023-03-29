# 2022/08/03 Editted by Shaotong Zhu
# prepare 2-sec pure NNS and non-NNS video clips for the video classifiers

# 1. extract starting frame index and ending frame index for all NNS and pacifier events.
# 2. generate a binary vector in the length of the input long video as the label template of NNS and pacifier actions.
# 3. calculate the intersection of the binary label vectors of NNS and pacifier actions to get the label vector of the
#    NNS events within pacifier events.
# 4. set up short clip window size, and then calculate the starting frame index and ending frame index of the short
#    clips by segmenting continuous windows on the pacifier-NNS event label vector.
# 5. save the video clips basing on the starting frame index and ending frame index of the short clips.


import pandas as pd
import os
import cv2
import numpy as np
import glob
from tqdm import tqdm
from loguru import logger
import argparse

from UtilTrim import extract_timestamps, intervalFramewise, outputInterval, transitionCounting, cutOutTransition, \
    extract_raw_frames_from_video, videoGen


parser = argparse.ArgumentParser(description='Demo for trimming long videos into short video clips.')
parser.add_argument('--subj', type=str, default='R7', help='Input subject name in the RawData folder.')
parser.add_argument('--window', type=int, default=26, help='Output short video clip length.')
parser.add_argument('--dataUsage', type=str, default='classification', help='the usage of the generated data (for '
                                                                            'training the classifier or segmentation '
                                                                            'evaluation).')

args = parser.parse_args()

# load long videos
subj = args.subj
window = args.window
dataUsage = args.dataUsage

# subj = 'R7'
# window = 26

clipPath = '../RawData/long videos/Copy of ' + subj + '_sleep.mp4'
video = cv2.VideoCapture(clipPath)
fps = video.get(cv2.CAP_PROP_FPS)
length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
# rawFrame = extract_raw_frames_from_video(video)

# load labels.
csvDir = glob.glob(os.path.join('../RawData/annotations/', subj, '*.csv'))[0]
saveDir = os.path.join('../RawData', 'data for ' + dataUsage, '/trimming results/', subj)
if not os.path.exists(saveDir):
    os.makedirs(saveDir)

# find the starting indexes and ending indexes of all NNS and pacifier events.
nns_intervals, pacifier_intervals = extract_timestamps(csvDir)

# converting into the binary label template vector.
nnsLabelTemplate = intervalFramewise(nns_intervals, fps, length)
pacLabelTemplate = intervalFramewise(pacifier_intervals, fps, length)

# get intersections for pacifier-NNS action.
pacNNSLabelTemplate = np.zeros(length)
pacNNSLabelTemplate[
    list(set(list(map(list, np.where(pacLabelTemplate == 1)))[0]) & set(
        list(map(list, np.where(nnsLabelTemplate == 1)))[0]))] = 1
pacNNSIndex = list(np.where(pacNNSLabelTemplate == 1))[0]
# segment intersections into 2-sec intervals, without sampling.
pacNNSstartList, pacNNSendList = outputInterval(pacNNSIndex, window=window)

# get intersections.
pacNonNNSLabelTemplate = np.zeros(length)
pacNonNNSLabelTemplate[
    list(set(list(map(list, np.where(pacLabelTemplate == 1)))[0]) & set(
        list(map(list, np.where(nnsLabelTemplate == 0)))[0]))] = 1
pacNonNNSIndex = list(np.where(pacNonNNSLabelTemplate == 1))[0]

# segment intersections into 2-sec intervals, without sampling.
pacNonNNSstartList, pacNonNNSendList = outputInterval(pacNonNNSIndex, window=window)

# segment video clips using intervals.
saveDir_NNS = os.path.join(saveDir, 'NNS')
if not os.path.exists(saveDir_NNS):
    os.makedirs(saveDir_NNS)
print('start processing NNS clips:\n')
videoGen(video, pacNNSstartList, pacNNSendList, saveDir_NNS, fps)

# segment video clips using intervals.
saveDir_NonNNS = os.path.join(saveDir, 'Non-NNS')
if not os.path.exists(saveDir_NonNNS):
    os.makedirs(saveDir_NonNNS)
print('start processing Non-NNS clips:\n')
videoGen(video, pacNonNNSstartList, pacNonNNSendList, saveDir_NonNNS, fps)

# statistic of the used frames and total frames fro NNS and non-NNS case.

# add random frames ahead to the starting frame of each pacifier-NNS event, in each clip, only 1 transition happens.
# returning the new starting frame list and ending frame list for the transition clips. For each clip, corresponding
# NNS vs. Non-NNS frame-wise labels are saved.
newStart, newEnd, proportion, newLabel = cutOutTransition(pacNNSstartList, window, pacNNSLabelTemplate, transCounting=1)

# segment video clips using starting frame and ending frame above.
saveDir_Trans = os.path.join(saveDir, 'Transition')
if not os.path.exists(saveDir_Trans):
    os.makedirs(saveDir_Trans)
print('start processing Transition clips:\n')
videoGen(video, newStart, newEnd, saveDir_Trans, fps, proportion, newLabel)
