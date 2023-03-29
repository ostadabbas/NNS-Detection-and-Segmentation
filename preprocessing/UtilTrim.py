import pandas as pd
import os
import cv2
import numpy as np
from tqdm import tqdm
from loguru import logger


# all the event names in the annotation csv files.
NNS_ANNOTATIONS = ['{"TEMPORAL-SEGMENTS":"NNS"}',
                   '{"TEMPORAL-SEGMENTS":"NNS Events"}',
                   '{"TEMPORAL-SEGMENTS":"NNS Event"}',
                   '{"TEMPORAL-SEGMENTS":"NNS Burst"}',
                   '{"Activity":"NNS"}'
                   ]

PACIFIER_ANNOTATIONS = ['{"TEMPORAL-SEGMENTS":"Pacifier"}',
                        '{"TEMPORAL-SEGMENTS":"Pacifier Events"}',
                        '{"TEMPORAL-SEGMENTS":"Pacifier Event"}',
                        '{"TEMPORAL-SEGMENTS":"Pacificer"}',
                        '{"TEMPORAL-SEGMENTS":"Pacificer Events"}',
                        '{"TEMPORAL-SEGMENTS":"Pacificer Event"}',
                        '{"Activity":"Pacifier"}'
                        ]


def extract_timestamps(csvDir):
    """
    Read the annotation table from the csv files. Extract the frame indexes for all NNS and pacifier events.
    :param
        csvDir: Directory of the annotation csv file.
    :return:
        nns_intervals: array of [starting frame index, ending frame index] for all NNS events.
        pacifier_intervals: array of [starting frame index, ending frame index] for all pacifier events.
    """
    annotation_table = pd.read_csv(csvDir)
    nns_table = annotation_table[annotation_table.metadata.isin(NNS_ANNOTATIONS)]
    assert len(nns_table) > 0, 'No NNS events found'
    nns_intervals = np.array(list(map(list, zip(nns_table.temporal_segment_start, nns_table.temporal_segment_end))))
    nns_lengths = nns_intervals[:, 1] - nns_intervals[:, 0]
    mean_length = np.mean(nns_lengths).round(3)
    median_length = np.median(nns_lengths).round(3)
    logger.info(
        f'Average NNS event length {mean_length} and median event length is {median_length}'
        f' with a total of {len(nns_intervals)} NNS events.')
    pacifier_table = annotation_table[annotation_table.metadata.isin(PACIFIER_ANNOTATIONS)]
    pacifier_intervals = []
    if len(pacifier_table) > 0:
        pacifier_intervals = np.array(
            list(map(list, zip(pacifier_table.temporal_segment_start, pacifier_table.temporal_segment_end))))
        pacifier_lengths = pacifier_intervals[:, 1] - pacifier_intervals[:, 0]
        mean_length = np.mean(pacifier_lengths).round(3)
        median_length = np.median(pacifier_lengths).round(3)
        logger.info(
            f'Average Pacifier event length {mean_length} and median event length is {median_length}'
            f' with a total of {len(nns_intervals)} Pacifier events.')

    return nns_intervals, pacifier_intervals


def intervalFramewise(intervals, fps, length):
    """
    Generate a binary vector in the length of the input long video as the label template of corresponding actions
    (0: no action; 1: action happening), helping later video clip localization.
    :param
        intervals: array of [starting frame index, ending frame index] for all events.
        fps: frame rate of the input long video.
        length: total frame number of the input long video.
    :return:
        a binary vector in the length of the input long video as the label template.
    """
    template = np.zeros(length)
    for intervalIndex in range(len(intervals)):
        start = round(intervals[intervalIndex][0] * fps)
        end = round(intervals[intervalIndex][1] * fps) + 1
        template[start:end] += 1
    return template


def outputInterval(indexList, window):
    """
    calculate the starting frame index and ending frame index of the short clips by segmenting continuous windows on the
    pacifier-NNS event label vector.
    :param indexList: a binary vector in the length of the input long video as the pacifier-NNS label template.
    :param window: frame number within the output short video clips.
    :return: startList, endList of the sampled short video clips.
    """
    startList, endList = [], []

    startIndex = 0
    while startIndex + window <= len(indexList) - 1:
        if indexList[startIndex + window] - indexList[startIndex] == window:
            startList.append(indexList[startIndex])
            endList.append(indexList[startIndex + window])
            startIndex += window
        else:
            startIndex += 1
    return startList, endList


def transitionCounting(sequence):
    counting = 0
    for i in range(1, len(sequence), 1):
        if sequence[i] != sequence[i - 1]:
            counting += 1
    return counting


def cutOutTransition(startIndex, window, pacNNSLabel, transCounting=1):
    """
    return the start and end index for the transition clips, only 1 action transition in each clip.
    :param startIndex:
    :param window:
    :param pacNNSLabel:
    :param transCounting:
    :return:
    """

    # NNS proportion fit Gaussian distribution, return the proportion for the naming of clip.
    np.random.seed(0)
    proportion = np.random.normal(0, 1, [len(startIndex), ])
    # normalize proportion
    _range = np.max(proportion) - np.min(proportion)
    proportion = (proportion - np.min(proportion)) / _range
    pushForward = list(map(int, window * proportion))

    newStart = []
    newEnd = []
    newProportion = []
    newLabel = []
    for i in range(len(pushForward)):

        if (startIndex[i] - pushForward[i]) < 0:
            continue
        if len(newEnd) > 0 and (startIndex[i] - pushForward[i]) <= newEnd[-1]:
            continue
        sequence = pacNNSLabel[(startIndex[i] - pushForward[i]):(startIndex[i] - pushForward[i]) + window]
        if transitionCounting(sequence) != transCounting:
            continue
        newStart.append(startIndex[i] - pushForward[i])
        newEnd.append(startIndex[i] - pushForward[i] + window)
        newProportion.append(proportion[i])
        newLabel.append(sequence)

    return newStart, newEnd, newProportion, newLabel


# 2. prepare into frames
def extract_raw_frames_from_video(video, start, end):
    """
    extract a list of frames according the starting index and ending index.
    :param video:
    :param start:
    :param end:
    :return:
    """
    rawFrames = {}
    video.set(cv2.CAP_PROP_POS_FRAMES, start)
    for frame_ndx in range(end - start):
        ret, frame = video.read()
        assert ret
        rawFrames[frame_ndx] = frame
    return rawFrames


def videoGen(video, startList, endList, saveDir, fps, proportion=None, newLabel=None):
    """
        Save the short video clips.
        :param video:
        :param startList:
        :param endList:
        :param saveDir:
        :return:
        """
    fw = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    for i in tqdm(range(len(startList))):
        if proportion is not None:
            clip_path = os.path.join(saveDir, str(i) + '_' + str(format(proportion[i], '.3f')) + '.mp4')
        else:
            clip_path = os.path.join(saveDir, f'{i}.mp4')
        out = cv2.VideoWriter(clip_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (fw, fh))
        start = startList[i]
        end = endList[i]
        rawFrame = extract_raw_frames_from_video(video, start, end)
        if newLabel is not None:
            actionLabelList = newLabel[i]
        for f in range(end - start):
            frame = rawFrame[f]
            if newLabel is not None:
                if actionLabelList[f] == 0:
                    action = 'Non-NNS'
                    color = (0, 255, 0)
                else:
                    action = 'NNS'
                    color = (0, 255, 255)
                word_x = 60
                word_y = 60
                cv2.putText(frame, action, (word_x, word_y), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            out.write(frame)
        out.release()