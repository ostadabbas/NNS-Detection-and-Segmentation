import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import argparse
from metadata import meta_in_crib, meta_in_wild
pd.set_option('display.precision', 1)

metadata = {
    'in-crib': meta_in_crib,
    'in-wild': meta_in_wild  
}

# Auxiliary functions.

# Conventions: A series of length n represents a time
# frame of 0 to n units of time. The index k entry, with 
# k = 0, ..., n-1, represents the time period from k to 
# k+1 units of time.

# Note: These functions are specifically designed for 
# discrete sequences. (Earlier versions I wrote were
# designed for continuous time events, which required
# more careful rounding and conversions.)




def intersection(a, b):
    # Computes length of intersection of two intervals.
    assert (a[0] <= a[1]) and (b[0] <= b[1]), 'Inverted interval'
    
    if ((a[0] <= b[0]) and (a[1] >= b[0])):
        return [b[0], min(a[1], b[1])]
    elif ((b[0] <= a[0]) and (b[1] >= a[0])):
        return [a[0], min(a[1], b[1])]
    else:
        return None
    
    
def IOU(a, b):
    intersect = intersection(a, b)
    
    if intersect is not None:
        union = max(a[1], b[1]) - min(a[0], b[0])
        return (intersect[1] - intersect[0]) / union
    else:
        return 0




def segments_to_series(length, events):
    # Converts segments to a time series.

    output = np.zeros(length)
    
    for event in events:
        for j in range(event[0], event[1]):
            output[j] = 1
        
    return np.array(output)




def series_to_segments(length, series):
    # Converts a time series to segments.

    events = []
    cur_seg = []
    
    for i in range(len(series)):
        if series[i] == 1 and len(cur_seg) == 0:
            cur_seg = [(i / len(series)) * length]
        if series[i] == 0 and len(cur_seg) == 1:
            events = events + [cur_seg + [(i / len(series)) * length]]
            cur_seg = []
        if i == len(series) - 1 and len(cur_seg) == 1:
            events = events + [cur_seg + [length]]
            
    return np.array(events)




def binarize(threshold, series):
    # Binarize a real-valued series based on a threshold.
    return np.array([(1 if x > threshold else 0) for x in series])
    

def organize_segments(segments, gap_len = 0):
    # Function which merges overlapping segments (optionally allowing 
    # for segments within gap_len to be merged as well).

    segments = np.array(segments)
    assert sum([seg[0] > seg[1] for seg in segments]) == 0 # Each segment is a valid interval
    
    # If there are fewer than two segments, do nothing.
    if len(segments) <= 1:
        return segments
    
    # First, sort the segments by start point.
    sorted_segs = [segments[i] for i in np.argsort([seg[0] for seg in segments])]
    
    # Next, merge segments that overlap. If gap_len > 0, then segments with 
    # gaps less than gap_len also get merged; this barely changes the algorithm.
    merged_segs = []
    start_i = 0
    cur_end = sorted_segs[0][1]
    for i in range(1, len(sorted_segs)):
        seg = sorted_segs[i]
        
        if seg[0] > cur_end + gap_len: # New segment; merge earlier ones.
            merged_segs = merged_segs + [[sorted_segs[start_i][0], cur_end]]
            start_i = i
            
            if i == len(sorted_segs) - 1: # Add last segment if at end.
                merged_segs = merged_segs + [seg]
                
        else: # Continuing segment; merge only if at end.
            if i == len(sorted_segs) - 1:
                merged_segs = merged_segs + [[sorted_segs[start_i][0], max(cur_end, seg[1])]]
                
        cur_end = max(cur_end, seg[1]) 
        
    return np.array(merged_segs)






def windows_to_frames(window_conf, window_stride, mode = 'simple'):
    # Funtion that aggregates window-wise confidence scores into 
    # framewise predictions.

    if mode == 'simple': # Simply take each confidence score to represent its middle window.
        agg_conf = np.repeat(window_conf, 5)
        agg_conf = np.pad(agg_conf, 10, mode = 'edge')
    elif mode == 'average': # Take window-wise moving average (convolution), then expand to frame-wise.
        avg_conf = np.convolve(window_conf, np.ones(window_stride), 'full') / window_stride
        agg_conf = np.repeat(avg_conf, 5)
    elif mode == 'non-overlap': # Take only a non-overlapping subsequence of the windows.
        nonoverlap_conf = window_conf[ : : window_stride]
        agg_conf = np.repeat(nonoverlap_conf, 25)
    
    return agg_conf




def precision_recall(gt, pred, threshold, verbose = False):
    
    # Precision-recall matched to THUMOS Challenge, with multiple 
    # hits tiebroken by IOU rather than confidence score. Verbose 
    # mode prints step-by-step reasoning.

    gt = organize_segments(gt)
    pred = organize_segments(pred)
    
    if verbose:
        print('Organized ground truth\n', gt, '\n')
        print('Organized predictions\n', pred, '\n')
        print('IOU Threshold', threshold, '\n')
    
    true_pos = 0
    false_pos = 0
    
    if verbose:
        print('Look through predictions\n')
        
    for pred_seg in pred:
        if verbose:
            print('Prediction', pred_seg)
        
        gt_matches = [gt_seg for gt_seg in gt if IOU(gt_seg, pred_seg) > threshold]
        if verbose:
            ('Thresholded matches with ground truth', gt_matches)
        
        if len(gt_matches) == 0:
            if verbose:
                print('False positive: no IOUs above threshold\n')
            false_pos += 1
        else:
            for gt_match in gt_matches:
                current_iou = IOU(gt_match, pred_seg)
                all_ious = [IOU(gt_match, pred_seg) for pred_seg in pred]
                if current_iou == max(all_ious):
                    if verbose:
                        print('True positive: IOU above threshold with', gt_match,
                              'and greater or equal to other predictions\n')
                    true_pos += 1 
                else:    
                    if verbose:
                        print('False positive: IOU above threshold with true event', gt_match,
                              'but less than another prediction\n')
                    false_pos += 1 
                    
    precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else None
    recall = true_pos / len(gt) if len(gt) > 0 else None
    if verbose:
        print('True positives', true_pos)
        print('False positives', false_pos)
        print('True events', len(gt))
        print('Precision', precision )
        print('Recall', recall)
        
    return precision, recall

def average_precision_recall(meta, dataset, frame_gt, window_conf, window_stride, iou_thres, conf_thres, mode):

    tot_precs = []
    tot_recs = []

    for i, subj in enumerate(meta[dataset].keys()):
        subj_precs = []
        subj_recs = []
        
        for j, clip in enumerate(meta[dataset][subj]):

            seg_gt = series_to_segments(601, frame_gt[dataset][subj, clip])
            frame_conf = windows_to_frames(window_conf[dataset][subj, clip], window_stride, mode)
            frame_pred = binarize(conf_thres, frame_conf)
            seg_pred = series_to_segments(600, frame_pred)

            prec, rec = precision_recall(seg_gt, seg_pred, iou_thres)
            if prec is not None:
                subj_precs += [prec]
            if rec is not None:
                subj_recs += [rec]
            
        if len(subj_precs) > 0:
            tot_precs = tot_precs + [(sum(subj_precs) / len(subj_precs))]
        if len(subj_recs) > 0:
            tot_recs = tot_recs + [(sum(subj_recs) / len(subj_recs))]

    avg_prec = sum(tot_precs) / len(tot_precs)
    avg_rec = sum(tot_recs) / len(tot_recs)

    return avg_prec, avg_rec

def load_dataset(meta, args):
    frame_gt = {} # Frame-wise ground truth.
    window_conf = {} # Window-wise confidence scores.

    for dataset in meta.keys():
        dataset_frame_gt = {}
        dataset_window_conf = {}

        for subj in meta[dataset].keys():
            for clip in meta[dataset][subj]:
                dataset_frame_gt[(subj, clip)] = np.load(args.result_dir + '/' + str(subj) + 
                    '/' + str(clip) + '.npy')[0:args.num_frames]
                dataset_window_conf[(subj, clip)] = np.load(args.result_dir + '/' + str(subj) + 
                    '/' + str(subj) + '_' + str(clip) + '_conf_overlap_1node.npy')
            
        assert dataset_frame_gt.keys() == dataset_window_conf.keys()
        
        frame_gt[dataset] = dataset_frame_gt
        window_conf[dataset] = dataset_window_conf
    
    return frame_gt, window_conf


def main(args):
    
    meta = {k:v for k,v in metadata.items() if k in args.datasets}

    num_frames = args.num_frames 
    num_windows = args.num_windows
    window_size = args.window_size
    window_stride = args.window_stride
    aggregation_methods = args.agg_methods

    num_subjs = {x: len(meta[x].keys()) for x in meta.keys()}
    num_clips = {x: max([len(meta[x][y]) for y in meta[x].keys()]) for x in meta.keys()} 

    # Load ground truth and inference confidence scores.

    frame_gt, window_conf = load_dataset(meta, args)

    for dataset in meta.keys():
        fig, ax = plt.subplots(num_subjs[dataset], num_clips[dataset], figsize = (4 * num_clips[dataset], num_subjs[dataset]), dpi = 200)

        for i, subj in enumerate(meta[dataset].keys()):
            ax[i, 0].set_ylabel(str(subj))

            for j, clip in enumerate(meta[dataset][subj]):
                ax[i, j].set_xlabel('Clip ' + str(clip))
                ax[i, j].set_ylim([0, 1])
                ax[i, j].fill_between(range(num_frames), [0] * num_frames, frame_gt[dataset][subj, clip], color = 'pink', alpha = 0.8)
                ax[i, j].plot(windows_to_frames(window_conf[dataset][subj, clip], 'simple'), c = 'r', alpha = 0.9)
                ax[i, j].plot(windows_to_frames(window_conf[dataset][subj, clip], 'non-overlap') - 0.05, c = 'c', alpha = 0.9)

        fig.tight_layout()
        fig.savefig(dataset+'_viz.png')

    IOU_thres = 0.1
    conf_thres = 0.9

    for ds in meta.keys():
        fig, ax = plt.subplots(num_subjs[ds], num_clips[ds], figsize = (num_clips[ds] * 5, num_subjs[ds] * 2), dpi = 200)
        tot_precs = []
        tot_recs = []

        for i, subj in enumerate(meta[ds].keys()):
            subj_precs = []
            subj_recs = []

            for j, clip in enumerate(meta[ds][subj]):

                seg_gt = series_to_segments(num_frames, frame_gt[ds][subj, clip])
                frame_conf = windows_to_frames(window_conf[ds][subj, clip], window_stride, 'average')
                frame_pred = binarize(conf_thres, frame_conf)
                seg_pred = series_to_segments(num_frames, frame_pred)

                prec, rec = precision_recall(seg_gt, seg_pred, IOU_thres)
                prec_str, rec_str = 'n/a', 'n/a'
                if prec is not None:
                    subj_precs += [prec]
                    prec_str = '%.2f'%prec
                if rec is not None:
                    subj_recs += [rec]
                    rec_str = '%.2f'%rec

                ax[i, j].set_xlabel('Clip ' + str(clip) + '\nPr ' + prec_str + ' | Rec ' + rec_str)
                if i + 1 < len(meta[ds].keys()):
                    ax[i, j].xaxis.set_ticklabels([])
        #         else:
        #             ax[i, j].set_xlabel('s')
        #             ax[i, j].xaxis.set_label_coords(1.02, -0.463)
                if j > 0:
                    ax[i, j].yaxis.set_ticklabels([])

                ax[i, j].set_ylim([-0.1, 1.1])
                ax[i, j].fill_between(np.linspace(0, 60, num_frames), [0] * num_frames, frame_gt[ds][subj, clip], color = 'pink', alpha = 0.8)
                ax[i, j].plot(np.linspace(0, 60, num_frames), frame_conf, c = 'purple', alpha = 0.3)
                ax[i, j].plot(np.linspace(0, 60, num_frames), binarize(conf_thres, frame_conf), c = 'r', alpha = 1)

            tot_precs = tot_precs + [sum(subj_precs) / len(subj_precs)]
            tot_recs =  tot_recs + [sum(subj_recs) / len(subj_recs)]
            
            ax[i, 0].set_ylabel(str(subj) + '\nPr ' + '%.1f'%(sum(subj_precs) / len(subj_precs))\
                                + ' | Rec ' + '%.1f'%(sum(subj_recs) / len(subj_recs)))

        avg_prec = tot_precs = 100 *  sum(tot_precs) / len(tot_precs)
        avg_rec = tot_recs = 100 * sum(tot_recs) / len(tot_recs)
        fig.suptitle('Dataset: ' + str.title(ds) + 
                    '\nClassifier Confidence Threshold ' + str(conf_thres) +
                    '\nPrecision-Recall IoU Threshold ' + str(IOU_thres) +
                    '\nAverage Precision ' + '%.1f'%avg_prec + 
                    ' | Average Recall ' + '%.1f'%avg_rec)

        fig.tight_layout()
        fig.savefig('segmentation-visualization_'+ds+'.png', bbox_inches = 'tight')
    # Visualization for publication

    if 'in-crib' in args.datasets:
        ds = 'in-crib'
            
        fig, ax = plt.subplots(num_subjs[ds], num_clips[ds], figsize = (10, 4), dpi = 200)

        tot_precs = []
        tot_recs = []

        for i, subj in enumerate(meta[ds].keys()):
            subj_precs = []
            subj_recs = []

            for j, clip in enumerate(meta[ds][subj]):

                seg_gt = series_to_segments(num_frames, frame_gt[ds][subj, clip])
                frame_conf = windows_to_frames(window_conf[ds][subj, clip], window_stride, 'simple')
                frame_pred = binarize(conf_thres, frame_conf)
                seg_pred = series_to_segments(num_frames, frame_pred)

                prec, rec = precision_recall(seg_gt, seg_pred, IOU_thres)
                prec_str, rec_str = 'n/a', 'n/a'
                if prec is not None:
                    subj_precs += [prec]
                    prec_str = '%.2f'%prec
                if rec is not None:
                    subj_recs += [rec]
                    rec_str = '%.2f'%rec

                # ax[i, j].set_xlabel('Clip ' + str(clip) + '\nPr ' + prec_str + ' | Rec ' + rec_str)
                if i + 1 < len(meta[ds].keys()):
                    ax[i, j].xaxis.set_ticklabels([])
                else:
                    ax[i, j].set_xlabel('s')
                    ax[i, j].xaxis.set_label_coords(1.03, -0.25)
                if j > 0:
                    ax[i, j].yaxis.set_ticklabels([])

                ax[i, j].set_ylim([-0.1, 1.1])
                ax[i, j].fill_between(np.linspace(0, 60, num_frames), [0] * num_frames, frame_gt[ds][subj, clip], color = 'pink', alpha = 0.8, label = 'Segmentation Ground Truth')
                ax[i, j].plot(np.linspace(0, 60, num_frames), frame_conf, c = 'purple', alpha = 0.5, label = 'Sliding Classifier Confidence')
                ax[i, j].plot(np.linspace(0, 60, num_frames), binarize(conf_thres, frame_conf), c = 'r', alpha = 0.8, label = 'Segmentation Prediction')
                
                ax[i, j].spines['top'].set_color('gray')
                ax[i, j].spines['left'].set_color('gray')
                ax[i, j].spines['right'].set_color('gray')
                ax[i, j].spines['bottom'].set_color('gray')
                ax[i, j].tick_params(left = False, color = 'gray')

            tot_precs = tot_precs + [sum(subj_precs) / len(subj_precs)]
            tot_recs =  tot_recs + [sum(subj_recs) / len(subj_recs)]

            ax[i, 0].set_ylabel(str(subj))

        avg_prec = tot_precs = 100 *  sum(tot_precs) / len(tot_precs)
        avg_rec = tot_recs = 100 * sum(tot_recs) / len(tot_recs)
        fig.suptitle('Dataset: ' + str.title(ds) + 
                    '\nClassifier Confidence Threshold ' + str(conf_thres) +
                    '\nPrecision-Recall IoU Threshold ' + str(IOU_thres) +
                    '\nAverage Precision ' + '%.1f'%avg_prec + 
                    ' | Average Recall ' + '%.1f'%avg_rec)

        handles, labels = ax[4, 4].get_legend_handles_labels()
        fig.legend(handles, labels, ncol = 3, bbox_to_anchor = (0.9, 1.1))

        fig.tight_layout()
        fig.savefig('segmentation-visualization-in-crib.png', bbox_inches = 'tight')


    # Function to compute average precision-recall for all 
    # subjects and clips.


    if 'in-wild' in args.datasets:
        dataset = 'in-wild'
        modes = ['non-overlap', 'simple', 'average']
        IOU_thress = [0.1, 0.3, 0.5]
        conf_thress = [0.1, 0.5, 0.9]

        results = np.zeros((len(IOU_thress), len(modes) * 2))

        for z, conf_thres in enumerate(conf_thress):
            for i, mode in enumerate(modes):
                for j, iou_thres in enumerate(IOU_thress):
                    prec, rec = average_precision_recall(meta, dataset, frame_gt, window_conf, window_stride, iou_thres, conf_thres, mode)
                    results[i, 2 * j] = prec * 100
                    results[i, 2 * j + 1] = rec * 100
                
            results_df = pd.DataFrame(results)
            results_df.columns = [metric + '@IoU' + str(thres) for thres in IOU_thress for metric in ['Pr', 'Rec']]
            results_df.index = ['Tiled Windows', 'Sliding Windows', 'Smoothed Sliding Windows']
            
            print('[' + str.title(dataset) + '] ' + 'Precision-Recall for Classifier Confidence Threshold @ ' + str(conf_thres))
            print(results_df)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_dir', default = './sliding_window_results', help = 'Directory containing segmentation results')
    parser.add_argument('--datasets', nargs='+', default=['in-wild'])
    parser.add_argument('--agg_methods', nargs='+', default=['simple'])
    parser.add_argument('--num_frames', default = 600, help = 'Total number of frames in the video')
    parser.add_argument('--num_windows', default = 116, help = 'Total number of windows')
    parser.add_argument('--window_size', default = 25, help = 'Size of the sliding window')
    parser.add_argument('--window_stride', default = 5, help = 'Stride of the sliding window')
    args = parser.parse_args()
    main(args)