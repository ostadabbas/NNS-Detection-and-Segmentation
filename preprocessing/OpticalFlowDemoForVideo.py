# Author: Deepak Pathak (c) 2016

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

parser = argparse.ArgumentParser(
    description='Demo for video optical flow calculation using Coarse2Fine')
parser.add_argument('--viz', default=True)
parser.add_argument('--path', type=str, default='./pyflow/examples/demoVideo.mp4')
parser.add_argument('--saveDir', type=str, default='./pyflow/examples/demoVideoResult')
args = parser.parse_args()


def frameDevide(videoDir, frameSaveDir):

    if not os.path.exists(frameSaveDir):
        os.makedirs(frameSaveDir)

    # 读取视频文件
    video = cv2.VideoCapture(videoDir)

    # 检查视频是否成功打开
    if not video.isOpened():
        print("无法打开视频文件")
        print(videoDir)

    # 获取视频帧率
    fps = int(video.get(cv2.CAP_PROP_FPS))

    # # 创建用于保存帧的文件夹
    # if not os.path.exists('frames'):
    #     os.makedirs('frames')

    # 循环遍历视频帧
    frame_count = 0
    while True:
        # 读取视频的单个帧
        ret, frame = video.read()

        # 如果无法读取到帧，则退出循环
        if not ret:
            break

        # 保存当前帧到文件夹中
        filename = f'frame_{frame_count:04d}.jpg'
        cv2.imwrite(os.path.join(frameSaveDir, filename), frame)

        # 增加帧数计数器
        frame_count += 1

    # 关闭视频文件
    video.release()


method = 'Pyflow-result'
if not os.path.exists(args.saveDir):
    os.mkdir(args.saveDir)

frameSaveDir = os.path.join(args.saveDir, 'temp')
if not os.path.exists(frameSaveDir):
    os.mkdir(frameSaveDir)

frameDevide(args.path, frameSaveDir)

images = glob.glob(os.path.join(frameSaveDir, '*.png')) + \
         glob.glob(os.path.join(frameSaveDir, '*.jpg'))

images = sorted(images)

im1 = np.array(Image.open(images[0]))
h, w, _ = im1.shape
# video generate
clip_path = os.path.join(args.saveDir, f'{method}.mp4')
out = cv2.VideoWriter(clip_path, cv2.VideoWriter_fourcc(*'mp4v'), 10, (w * 3, h))

i = 0
for imfile1, imfile2 in zip(images[:-1], images[1:]):
    im1 = np.array(Image.open(imfile1))
    im2 = np.array(Image.open(imfile2))
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
    print('Time Taken: %.2f seconds for image of size (%d, %d, %d)' % (
        e - s, im1.shape[0], im1.shape[1], im1.shape[2]))
    flow = np.concatenate((u[..., None], v[..., None]), axis=2)
    np.save('pyflow/examples/outFlow.npy', flow)

    if args.viz:
        import cv2

        hsv = np.zeros(im1.shape, dtype=np.uint8)
        hsv[:, :, 0] = 255
        hsv[:, :, 1] = 255
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR) / 255.
        blend = cv2.addWeighted(rgb, 0.7, im2, 0.3, 0)

        # plt.imshow(im1)
        # plt.show()
        # plt.imshow(rgb)
        # plt.show()
        # plt.imshow(blend)
        # plt.show()

        output = np.concatenate((im2, rgb, blend), axis=1)

        # plt.imshow(output)
        # plt.show()
        # cv2.imwrite('examples/outFlow_new.png', rgb)
        # cv2.imwrite(f'NNS-optical/rgb{i}.png', rgb)
        # cv2.imwrite(f'./Pyflow-result/frame{i + 1}-{i + 2}.png', rgb * 255)
        # cv2.imwrite(f'./concatenate_results/Pyflow-result/frame{i + 1}-{i + 2}.png', output * 255)

        # image concatenate:
        test = (output * 255).astype(np.uint8)
        out.write(test)

        i += 1
        # cv2.imwrite('examples/car2Warped_new.jpg', im2W[:, :, ::-1] * 255)
