import torch
from torch.autograd import Variable
import torch.nn.functional as F
import time
import os
import numpy as np
import sys
import torch.nn as nn
import torchvision.transforms as transforms
import json
from cnn_lstm.mean import get_mean, get_std
from PIL import Image
import argparse
import cv2
from cnn_lstm.models import cnnlstmForSegEval as cnnlstm
from cnn_lstm.utils import AverageMeter
from cnn_lstm.opts import parse_opts
from cnn_lstm.spatial_transforms import (
    Compose, Normalize, Scale, CenterCrop, CornerCrop, MultiScaleCornerCrop,
    MultiScaleRandomCrop, RandomHorizontalFlip, ToTensor)
from cnn_lstm.temporal_transforms import LoopPadding, TemporalRandomCrop
from cnn_lstm.target_transforms import ClassLabel, VideoID
from cnn_lstm.target_transforms import Compose as TargetCompose
from sklearn.metrics import precision_score
from metadata import meta_in_crib, meta_in_wild

metadata = {
    'in-crib': meta_in_crib,
    'in-wild': meta_in_wild  
}

idx_to_class={0:'Non-NNS', 1:'NNS'}

def generate_model(opt, device):
	assert opt.model in [
		'cnnlstm'
	]

	if opt.model == 'cnnlstm':
		model = cnnlstm.CNNLSTM(num_classes=opt.n_classes)
	return model.to(device)

def pred_tolerance (pred, y_label, NNS):
    new_pred=pred
    for i in range (len(pred)):
        if pred[i]==NNS and y_label[i]!=NNS:
            if i >0:
                if y_label[i-1]==NNS:
                    new_pred[i-1]=NNS
                    new_pred[i]=y_label[i]
            if i<len(pred)-1:
                if y_label[i+1]==NNS:
                    new_pred[i+1]=NNS
                    new_pred[i]=y_label[i]
    return new_pred


def resume_model(opt, model, weight_path):
    # Loading model weights from path
    checkpoint = torch.load(weight_path, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])

def window(a, w = 4, o = 2, copy = False):
    # Generating windows with the given stride

    sh = (a.size - w + 1, w)
    st = a.strides * 2
    view = np.lib.stride_tricks.as_strided(a, strides = st, shape = sh)[0::o]
    if copy:
        return view.copy()
    else:
        return view

def predict(clip, model, opt):
    # Performing inference on a single window
    if opt.no_mean_norm and not opt.std_norm:
        norm_method = Normalize([0, 0, 0], [1, 1, 1])
    elif not opt.std_norm:
        norm_method = Normalize(opt.mean, [1, 1, 1])
    else:
        norm_method = Normalize(opt.mean, opt.std)

    spatial_transform = Compose([
        Scale((150, 150)),
        ToTensor(opt.norm_value), norm_method
    ])

    if spatial_transform is not None:
        clip = [spatial_transform(img) for img in clip]

    clip = torch.stack(clip, dim=0)
    clip = clip.unsqueeze(0)
    with torch.no_grad():
        outputs = model(clip)
    score=outputs
    if outputs< 0.5:
        outputs = 0
    else:
        outputs=1
    preds=(1-outputs,1-score.item())

    return preds

def process_video(video_frames, model, opts):
    #  Splitting a video into sliding windows and performing inference on each window

    overlap_pred=[]
    overlap_conf=[]
    save_indices = []
    clip_frames = np.array((range(len(video_frames))))
    sliding_window_indices = window(clip_frames, w=int(opts.sample_dur), o=opts.window_stride,copy=True)
    num_windows, l =(sliding_window_indices.shape)

    for i in range(num_windows):
        print(f" sliding window: {i+1}")
        indices = sliding_window_indices[i,:]
        save_indices.append([indices[0],indices[-1]])
        window_frames = []
        for ind in indices:
            window_frames.append(video_frames[ind])
        preds,score = predict(window_frames, model, opts)
        overlap_pred.append(preds)
        overlap_conf.append(score)
        print(idx_to_class[preds] , round(score, 4),preds)
        print('********************')
    overlap_pred.append(overlap_pred[-1])
    overlap_conf.append(overlap_conf[-1])

    return save_indices, overlap_conf, overlap_pred

def main():
    opt = parse_opts()
    print(opt)
    device = torch.device("cpu")
    ds = 'in-wild'
    model = generate_model(opt, device)

    model_path = opt.model_weights
    resume_model(opt, model, model_path)
    model.eval()

    meta = {k:v for k,v in metadata.items() if k in opt.datasets}

    raw_root_dir = opt.raw_vid_dir
    flow_root_dir = opt.opt_flow_dir
    results_root_dir = opt.results_dir
    sample_dur = opt.sample_dur

    for i, subj in enumerate(meta[ds].keys()):

        print('Processing subject: ', subj)
        optical_flow_dir = os.path.join(flow_root_dir, subj)
        raw_vid_dir = os.path.join(raw_root_dir, subj)
        results_dir = os.path.join(results_root_dir, subj)
        
        if not os.path.exists(results_dir):
            # Create a new directory because it does not exist
            os.makedirs(results_dir)


        for clipname in meta[ds][subj]:
            print('Processing clip: ', clipname)

            of_name = os.path.join(optical_flow_dir, 'video', str(clipname)+'.mp4')
            orig_name = os.path.join(raw_vid_dir, str(clipname)+'.mp4')
            out_no_name = os.path.join(results_dir, str(clipname)+'_inference.mp4')
            conf_overlap_name = os.path.join(results_dir, str(clipname)+'_conf_overlap_1node.npy')
            pred_overlap_name = os.path.join(results_dir, str(clipname)+'_pred_overlap_1node.npy')
            indices_name = os.path.join(results_dir, str(clipname)+'_index_tracker.npy')

            cam = cv2.VideoCapture(of_name)
            cam2 = cv2.VideoCapture(orig_name)
            fps = cam2.get(cv2.CAP_PROP_FPS)
            

            width = int(cam2.get(3)) # cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)   # float `width`
            height = int(cam2.get(4)) # cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)  # float `height`
            vid_length = int(cam2.get(cv2.CAP_PROP_FRAME_COUNT))
            fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
            out = cv2.VideoWriter(out_no_name, fourcc, fps, (width, height))
            
            optic_frames=[]

            opt.mean = get_mean(opt.norm_value, dataset=opt.mean_dataset)
            opt.std = get_std(opt.norm_value)

            frame_count = 0
            out_frames=[]
            ret=True
            
            # Reading frames
            while True and ret is True:
                ret, img_optic = cam.read()
                ret, img_orig = cam2.read()

                if img_optic is not None:
                    img_optic = Image.fromarray(img_optic.astype('uint8'), 'RGB')
                    out_frames.append(img_orig)
                    optic_frames.append(img_optic)
                    frame_count += 1

            cam.release()
            out.release()
            cv2.destroyAllWindows()
            save_indices, overlap_conf, overlap_pred = process_video(optic_frames, model, opt)
            np.save(indices_name, np.array(save_indices) )
            np.save(conf_overlap_name, overlap_conf)#, 'dtype=object' )
            np.save(pred_overlap_name, overlap_pred)#, 'dtype=object' )
            cam.release()
            out.release()


if __name__ == "__main__":
    main()