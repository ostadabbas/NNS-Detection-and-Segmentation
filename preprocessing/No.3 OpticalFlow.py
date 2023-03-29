# optical flow video generation demo
# 09/22/2022 edited by Shaotong Zhu


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# from __future__ import unicode_literals
import numpy as np
from PIL import Image
import time
import argparse
from pyflow import pyflow
import glob
import os
import cv2
import shutil


# frame segmentation:
def frameSeg(videoDir, saveDir):
    """
    :param videoDir: e.g. './demo-video/0084.mp4'
    :param saveDir: e.g. 'NNS-Frame'
    :return:
    """

    vidcap = cv2.VideoCapture(videoDir)
    success, image = vidcap.read()
    count = 0
    # success = True
    while success:
        cv2.imwrite(saveDir + "/frame%03d.jpg" % count, image)  # save frame as JPEG file
        success, image = vidcap.read()
        if cv2.waitKey(10) == 27:
            break
        count += 1


parser = argparse.ArgumentParser(
    description='Video optical flow calculation using Coarse2Fine')
parser.add_argument('--viz', default=True)
parser.add_argument('--subj', type=str, default='R7')
parser.add_argument('--action', type=str, default='NNS')
parser.add_argument('--aug', type=str, default='aug_results')
parser.add_argument('--dataUsage', type=str, default='classification', help='the usage of the generated data (for '
                                                                            'training the classifier or segmentation '
                                                                            'evaluation).')
args = parser.parse_args()

# 1. read all file's name in the directory
# rootDir = 'F:/shawn/NEU_project/mini_project/cnn-lstm-master/data/video_data'
# rootDir = 'F:/shawn/NEU_project/mini_project/Augmented_videoclips/raw/R3/Emma/NNS/normal'
# action = 'Non-NNS'  # 'Non-NNS'

subj = args.subj
action = args.action
aug = args.aug
dataUsage = args.dataUsage
videoDir = os.path.join('../RawData', 'data for ' + dataUsage, 'cropping results', aug, subj, action)
saveDir = os.path.join('../RawData', 'data for ' + dataUsage, 'optical flow results', aug, subj, action)

fileList = os.listdir(videoDir)
method = 'Pyflow-result'

if not os.path.exists(saveDir):
    os.mkdir(saveDir)
saveDir = os.path.join(saveDir, action)
if not os.path.exists(saveDir):
    os.mkdir(saveDir)

for index, fileName in enumerate(fileList):
    print(f'{index}/{len(fileList)}')

    fileDir = os.path.join(videoDir, fileName).replace('\\', '/')
    tempDir = os.path.join(saveDir, 'temp').replace('\\', '/')
    if not os.path.exists(tempDir):
        os.mkdir(tempDir)
    frameSeg(fileDir, tempDir)

    images = glob.glob(os.path.join(tempDir, '*.png')) + \
             glob.glob(os.path.join(tempDir, '*.jpg'))

    images = sorted(images)

    if len(images) == 0: break

    im1 = np.array(Image.open(images[0]))
    h, w, _ = im1.shape
    # video generate

    fileIndex = fileName.replace('.mp4', '')
    out = cv2.VideoWriter(saveDir + f'/{fileIndex}.mp4',
                          cv2.VideoWriter_fourcc(*'mp4v'), 10, (w, h))

    i = 0
    for imfile1, imfile2 in zip(images[:-1], images[1:]):
        # im1 = np.array(Image.open(imfile1))
        # im2 = np.array(Image.open(imfile2))
        im1 = np.array(Image.open(imfile1).resize((100, 100)))
        im2 = np.array(Image.open(imfile2).resize((100, 100)))
        im1 = im1.astype(float) / 255.
        im2 = im2.astype(float) / 255.

        # Flow Options:
        alpha = 0.012
        ratio = 0.75
        minWidth = 20
        nOuterFPIterations = 7
        nInnerFPIterations = 1
        nSORIterations = 30
        colType = 0  # 0 or default:RGB, 1:GRAY (but pass gray image with shape (h,w,1))

        s = time.time()
        u, v, im2W = pyflow.coarse2fine_flow(
            im1, im2, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations,
            nSORIterations, colType)
        e = time.time()
        # print('Time Taken: %.2f seconds for image of size (%d, %d, %d)' % (
        #     e - s, im1.shape[0], im1.shape[1], im1.shape[2]))
        flow = np.concatenate((u[..., None], v[..., None]), axis=2)
        # np.save('examples/outFlow.npy', flow)

        if args.viz:
            import cv2

            hsv = np.zeros(im1.shape, dtype=np.uint8)
            hsv[:, :, 0] = 255
            hsv[:, :, 1] = 255
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            hsv[..., 0] = ang * 180 / np.pi / 2
            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR) / 255.
            rgb = cv2.resize(rgb, (w, h))
            # blend = cv2.addWeighted(rgb, 0.7, im2, 0.3, 0)

            # output = np.concatenate((im2, rgb, blend), axis=1)

            # cv2.imwrite(f'{clip_path}/frame{i + 1}-{i + 2}.png', output * 255)
            out.write((rgb * 255).astype(np.uint8))

            i += 1

    shutil.rmtree(tempDir)