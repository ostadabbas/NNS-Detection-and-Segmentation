import cv2
import numpy as np
import os
import argparse
import time
import torch

from preprocessing.UtilCrop import \
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

from preprocessing.pyflow import pyflow
from cnn_lstm.mean import get_mean, get_std
from segmentation import generate_model, resume_model, process_video, idx_to_class
from cnn_lstm.opts import parse_opts
from PIL import Image
from evaluation import windows_to_frames

def main(opts):

    start = time.time()
    print(opts)
    device = torch.device("cpu")
    model = generate_model(opts, device)

    model_path = opts.model_weights
    resume_model(opts, model, model_path)
    model.eval()

    clipPath = opts.input_clip_path

    video = cv2.VideoCapture(clipPath)
    fps = video.get(cv2.CAP_PROP_FPS)

    # extract raw frames from video
    raw_frames = extract_raw_frames_from_video(video, rotate=False)
    og_h, og_w, _ = raw_frames[0].shape 

    # detect and track the facial bounding box
    bounding_boxes_dict = track_face_raw(raw_frames, inflation_thresh = opts.inflation_thresh)

    if bounding_boxes_dict is None:
        print('No faces found')
        exit(0)

    for i in range(len(bounding_boxes_dict)):
        bounding_boxes_dict[i] = process_bbox_raw(bounding_boxes_dict[i], inflation_thresh = opts.inflation_thresh)

    bounding_boxes_dict = adjustBboxDim(bounding_boxes_dict, raw_frames)
    
    # cropped frame list
    croppedFrame = cropFace(raw_frames, bounding_boxes_dict)

    # generate the trajectory
    transforms = trajectoryCal(croppedFrame)

    # smooth the trajectory
    transforms_smooth = trajectorySmooth(transforms, SMOOTHING_RADIUS=50)

    # stabilized frames
    chopped_frames = chop2VideoVer2(croppedFrame, bounding_boxes_dict, transforms_smooth, None, fps, clipPath, bboxSize = opts.bboxSize,
                    inflationFactor = opts.blackedgeCutoff)

    h, w, _ = chopped_frames[0].shape

    # Performing optical flow on the chopped frames
    flow_frames = []
    i = 0
    for imfile1, imfile2 in zip(chopped_frames[:-1], chopped_frames[1:]):
        im1 = cv2.resize(imfile1,(100,100))
        im2 = cv2.resize(imfile2,(100,100))
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
        # print('Time Taken: %.2f seconds for frame %d of size (%d, %d, %d)' % (
        #     e - s, i + 1, im1.shape[0], im1.shape[1], im1.shape[2]))
        flow = np.concatenate((u[..., None], v[..., None]), axis=2)

        hsv = np.zeros(im1.shape, dtype=np.uint8)
        hsv[:, :, 0] = 255
        hsv[:, :, 1] = 255
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR) / 255.
        rgb = cv2.resize(rgb, (w, h))

        flow_img = (rgb * 255).astype(np.uint8)
        flow_PIL = Image.fromarray(flow_img.astype('uint8'), 'RGB')
        flow_frames.append(flow_PIL)
        i+=1

    
    # Performing segmentation on the video with the given window size and stride
    save_indices, overlap_conf, overlap_pred = process_video(flow_frames, model, opts)

    # Using the smoothed approach to convert window results to frame-wise results
    smoothed_conf = windows_to_frames(overlap_conf, opts.window_stride, 'average')

    # Setup output video file
    out_vid_name = opts.output_video_path
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    out = cv2.VideoWriter(out_vid_name, fourcc, fps, (og_w, og_h))

    # Visualize segmentation results on output video file
    for idx, raw_frame in raw_frames.items():
        if idx == 0:
            continue
        draw = raw_frame.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        pred = 0 if smoothed_conf[idx-1] < 0.5 else 1
        cv2.putText(draw, idx_to_class[pred], (70, 140), font, 1.5, (255, 100, 255), 1, cv2.LINE_AA)
        cv2.putText(draw, 'conf:'+ str(round(smoothed_conf[idx-1],2)), (70, 180), font, 1.5, (255, 0, 255), 1, cv2.LINE_AA)
        out.write(draw)
    
    end = time.time() - start
    print('Total time taken for processing clip: ', round(end,2))

if __name__ == "__main__" :
    opts = parse_opts()
    opts.mean = get_mean(opts.norm_value, dataset=opts.mean_dataset)
    opts.std = get_std(opts.norm_value) 
    main(opts)