# for the first round pure action data tuning

import cv2
import numpy as np
import os
import argparse

from UtilCrop import \
    extract_raw_frames_from_video, \
    track_face_raw, \
    track_mouth_raw, \
    horizontalFlip, \
    rotation, \
    cropFace, \
    trajectoryCal, \
    trajectorySmooth, \
    chop2VideoVer2, \
    chop2Video, \
    adjustBboxDim, \
    process_bbox_raw


parser = argparse.ArgumentParser(description='Demo for cropping short video clips in to area of interest.')
parser.add_argument('--subj', type=str, default='R7', help='Input subject name in the RawData folder.')
parser.add_argument('--action', type=str, default='NNS', help='Type of action.')
parser.add_argument('--augment', default=True, help='Data augmentation indicator.')
parser.add_argument('--rotate', default=False, help='Rotate the input video for the retina face detector if necessary, '
                                                    'since retina face works best when the face is perfectly vertical '
                                                    'in the frame. Possible choices including cv2.ROTATE_90_CLOCKWISE, '
                                                    'cv2.ROTATE_90_COUNTERCLOCKWISE, and cv2.ROTATE_180.')
parser.add_argument('--bboxSize', type=int, default=550, help='Output video size (in square bounding box).')
parser.add_argument('--blackedgeCutoff', type=float, default=1.2, help='For video stabilization output cut off. '
                                                                       'Sometimes the stabilized video has black edge. '
                                                                       'This inflation can reduce it.')
parser.add_argument('--inflation_thresh', type=float, default=0.3, help='For corner padding and output bbox size. The '
                                                                        'direct output from retina face is too tight.')
parser.add_argument('--dataUsage', type=str, default='classification', help='the usage of the generated data (for '
                                                                            'training the classifier or segmentation '
                                                                            'evaluation).')


args = parser.parse_args()

# load long videos
subj = args.subj
action = args.action
augment = args.augment
rotate = args.rotate
dataUsage = args.dataUsage
if dataUsage == 'segmentation':
    augment = False

# for fixing the output video dimension
bboxSize = args.bboxSize

# for video stabilization output cut off. Sometimes the stabilized video has black edge. This inflation can reduce it.
blackedgeCutoff = args.blackedgeCutoff

# for corner padding and output bbox size. The direct output from retina face is too tight
# new_x = max(0, x - inflation_thresh * w)
# new_h = h + inflation_thresh * h * 2
inflation_thresh = args.inflation_thresh

# 1. read all file's name in the directory
videoDir = os.path.join('../RawData', 'data for ' + dataUsage, 'trimming results/', subj, action)

if augment:
    saveDir = os.path.join('../RawData', 'data for ' + dataUsage, 'cropping results/aug_reults/', subj, action)
else:
    saveDir = os.path.join('../RawData', 'data for ' + dataUsage, 'cropping results/noAug_reults/', subj, action)

if dataUsage == 'segmentation':
    augment = False
    saveDir = os.path.join('../RawData', 'data for ' + dataUsage, 'cropping results/', subj, action)

if not os.path.exists(saveDir):
    os.makedirs(saveDir)

fileList = os.listdir(videoDir)
fileList.sort()
assert len(fileList) > 0, "No file in the " + videoDir

# 4. random number setup
np.random.seed(10)

flipIndicator = [round(i) for i in np.random.uniform(0, 1, [len(fileList), ])]

rotationIndicator = [i for i in np.random.normal(0, 30, [len(fileList), ])]

# for index, filename in enumerate(tqdm(fileList)):
for index, filename in enumerate(fileList):

    # filename = '0003.mp4'
    # index = 3
    print(filename)

    # 2. read a given video into frame list.
    # change to For loop in the future
    clipPath = os.path.join(videoDir, filename).replace('\\', '/')
    video = cv2.VideoCapture(clipPath)
    fps = video.get(cv2.CAP_PROP_FPS)
    raw_frames = extract_raw_frames_from_video(video, rotate)

    # 3. detect and track the facial bounding box
    bounding_boxes_dict = track_face_raw(raw_frames, inflation_thresh)
    # bounding_boxes_dict = track_mouth_raw(raw_frames, inflation_thresh)

    if bounding_boxes_dict is None:
        continue

    # save_dir = 'F:/shawn/NEU_project/mini_project/VideoClassificationDataPrep/2_Face_cropping/' \
    #            'Unaugmented_videoclips'
    # chop2Video(raw_frames, bounding_boxes_dict, save_dir, fps, filename+'_No3')


    if augment:
        if flipIndicator[index]:
            for i in range(len(raw_frames)):
                raw_frames[i], bounding_boxes_dict[i] = horizontalFlip(raw_frames[i], bounding_boxes_dict[i])
                raw_frames[i], bounding_boxes_dict[i] = rotation(img=raw_frames[i],
                                                                 bboxes=np.array(bounding_boxes_dict[i]),
                                                                 angle=rotationIndicator[index], inflation_thresh=inflation_thresh)
        else:
            for i in range(len(raw_frames)):
                raw_frames[i], bounding_boxes_dict[i] = rotation(img=raw_frames[i],
                                                                 bboxes=np.array(bounding_boxes_dict[i]),
                                                                 angle=rotationIndicator[index], inflation_thresh=inflation_thresh)
    else:
        for i in range(len(bounding_boxes_dict)):
            bounding_boxes_dict[i] = process_bbox_raw(bounding_boxes_dict[i], inflation_thresh)


    # print(f'tracked bounding box shape: {bounding_boxes_dict[0]}')
    bounding_boxes_dict = adjustBboxDim(bounding_boxes_dict, raw_frames)
    # 6. cropped frame list
    croppedFrame = cropFace(raw_frames, bounding_boxes_dict)
    # for img in croppedFrame:
    #     plt.figure("Image")  # 图像窗口名称
    #     plt.imshow(img)
    #     plt.show()

    # print(f"croppedFrame shape: {croppedFrame[0].shape}")

    # 7. generate the trajectory
    transforms = trajectoryCal(croppedFrame)
    # transforms[:, 2] = 0

    # 8. smooth the trajectory
    transforms_smooth = trajectorySmooth(transforms, SMOOTHING_RADIUS=50)
    # transforms_smooth[:, 2] = 0

    # 9. save video
    chop2VideoVer2(croppedFrame, bounding_boxes_dict, transforms_smooth, saveDir, fps, filename, bboxSize,
                   blackedgeCutoff)

    #   break
    # break