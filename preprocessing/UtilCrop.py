import cv2
import numpy as np
import os
from retinaface import RetinaFace
from loguru import logger
from tqdm import tqdm


def extract_raw_frames_from_video(video, rotate=True):
    rawFrames = {}
    for frame_ndx in range(int(video.get(cv2.CAP_PROP_FRAME_COUNT))):
        ret, frame = video.read()
        assert ret
        if rotate:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        rawFrames[frame_ndx] = frame

    return rawFrames


def detectFace(raw_frames, cornerPadding, bboxSize):
    detector = RetinaFace
    bounding_boxes_dict = []
    for frameIndex, frame in raw_frames.items():
        # print(frameIndex)
        #   3.1. factial detection
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # frame_rgb_rotate = cv2.rotate(frame_rgb, cv2.ROTATE_180)
        faces = detector.detect_faces(frame_rgb)
        # assert len(faces) == 1
        if len(faces) != 1:
            bounding_boxes_dict.append(np.zeros(4))
            continue
        confidence, bbox, landmarks = faces['face_1'].values()
        assert confidence > 0.7
        x1, y1, x2, y2 = bbox
        bbox = np.array(
            [x1 - cornerPadding, y1 - cornerPadding, x2 - x1 + bboxSize, y2 - y1 + bboxSize]).round().astype(int)

        bounding_boxes_dict.append(bbox)

    return bounding_boxes_dict


def get_coordinates_v1(landmarks, width, mouthInflation):
    nose = landmarks['nose']
    ml = landmarks['mouth_left']
    mr = landmarks['mouth_right']
    # width: full size frame width
    min_x = min(nose[0], ml[0], mr[0])
    max_x = max(nose[0], ml[0], mr[0])
    min_y = min(nose[1], ml[1], mr[1])
    max_y = max(nose[1], ml[1], mr[1])

    min_x -= (mouthInflation * int(width))
    max_x += (mouthInflation * int(width))
    min_y -= (mouthInflation * int(width))
    max_y += (mouthInflation * int(width))
    factor = (max_x - min_x)

    return int(min_x), int(min_y), int(max_x), int(min_y + factor)


def find_mouth_in_frame_raw(frame, fail_mode):
    try:
        detector = RetinaFace
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = detector.detect_faces(frame_rgb)

        assert len(faces) == 1
        # assert isinstance(faces, tuple)
        confidence, bbox, landmarks = faces['face_1'].values()
        assert confidence > 0.7
        width = frame_rgb.shape[0]
        x1, y1, x2, y2 = get_coordinates_v1(landmarks, width, mouthInflation=0.001)
        bbox = np.array(
            [x1, y1, x2 - x1, y2 - y1]).round().astype(int)
        return bbox
    except AssertionError:
        if fail_mode:
            return None
        logger.error('Could Not Find Mouth, Please select the bounding box, press (c) if there is no face')
        bbox = cv2.selectROI(frame, False)
        return bbox


def find_face_in_frame_raw(frame, fail_mode):
    try:
        detector = RetinaFace
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = detector.detect_faces(frame_rgb)

        assert len(faces) == 1
        # assert isinstance(faces, tuple)
        confidence, bbox, landmarks = faces['face_1'].values()
        assert confidence > 0.7
        x1, y1, x2, y2 = bbox
        bbox = np.array(
            [x1, y1, x2 - x1, y2 - y1]).round().astype(int)
        return bbox
    except AssertionError:
        if fail_mode:
            return None
        logger.error('Could Not Find Face, Please select the bounding box, press (c) if there is no face')
        bbox = cv2.selectROI(frame, False)
        return bbox


def process_bbox_raw(raw_bbox, inflation_thresh=0):
    """
    some post processing. sometimes we want to expand the bouding box. this also ensures the bbox is square and of type int.
    """

    x, y, w, h = raw_bbox
    # fixWidth = 400
    # inflation_thresh = 0.

    new_x = max(0, x - inflation_thresh * w)
    new_y = max(0, y - inflation_thresh * h)

    new_h = h + inflation_thresh * h * 2
    new_w = w + inflation_thresh * w * 2

    new_dim = max(new_h, new_w)
    new_bbox = [new_x, new_y, new_dim, new_dim]

    return tuple([int(round(val)) for val in new_bbox])


def track_face_raw(frames_dict, inflation_thresh):
    """

    """

    # initialize
    bounding_boxes_dict = {frame_ndx: np.full(4, np.nan) for frame_ndx, _ in frames_dict.items()}

    # find first frame with reliable bounding box. detection will be None if not reliable
    # uses a stride to speed up computation
    seen_frame_indices = []
    for frame_ndx, frame in frames_dict.items():
        seen_frame_indices.append(frame_ndx)
        # if frame_ndx % 4 != 0:
        #     continue
        # initial_bbox = find_face_in_frame(frame, template_images=template_images, template_save_dir=template_save_dir)
        initial_bbox = find_face_in_frame_raw(frame, True)
        if initial_bbox is not None:
            break
        # else:
        #     raise RuntimeError('No Face found')

    if initial_bbox is None:
        # logger.error('Could Not Find Face, Please select the bounding box, press (c) if there is no face')
        # initial_bbox = cv2.selectROI(frames_dict[0], False)
        return None

    # initial_bbox = process_bbox_raw(initial_bbox, inflation_thresh)
    first_frame_found_ndx = seen_frame_indices[-1]
    first_frame_found = frames_dict[first_frame_found_ndx]

    # bounding_boxes_dict[first_frame_found_ndx] = process_bbox_raw(initial_bbox, inflation_thresh)
    bounding_boxes_dict[first_frame_found_ndx] = initial_bbox

    tracker = cv2.legacy.TrackerMOSSE_create()
    # tracker = cv2.legacy.TrackerCSRT_create()
    ok = tracker.init(first_frame_found,
                      initial_bbox)
    assert ok

    # next it tracks backward if the first reliable detection was not the first frame
    prev_frame = first_frame_found
    prev_bbox = initial_bbox
    for frame_ndx in reversed(seen_frame_indices[:-1]):
        curr_frame = frames_dict[frame_ndx]
        ok, bbox = tracker.update(curr_frame)
        # retinaResult = retinaBbox[frame_ndx]
        # if (retinaResult - [0, 0, 0, 0]).all:
        # iou = bb_intersection_over_union(bbox, process_bbox(retinaResult,bboxSize))
        if not ok:
            bbox = find_face_in_frame_raw(curr_frame, True)
            #                           template_images=template_images,
            #                           template_save_dir=template_save_dir
            if bbox is None:
                assert prev_bbox is not None
                bbox = prev_bbox
            # bbox = process_bbox_raw(bbox, inflation_thresh)
            tracker = cv2.legacy_TrackerMOSSE.create()
            # pip3 install opencv-contrib-python==4.5.5.62 if error comes up
            ok = tracker.init(curr_frame, bbox)
            assert ok
        # bbox = process_bbox_raw(bbox, inflation_thresh)
        bounding_boxes_dict[frame_ndx] = bbox

    # next it tracks forward to the frames that follow
    remaining_frames = sorted(list(set(list(frames_dict.keys())).difference(seen_frame_indices)))
    tracker = cv2.legacy.TrackerMOSSE_create()
    ok = tracker.init(first_frame_found,
                      initial_bbox)
    assert ok
    prev_frame = first_frame_found
    prev_bbox = initial_bbox
    for frame_ndx in remaining_frames:
        # print(frame_ndx)
        curr_frame = frames_dict[frame_ndx]
        ok, bbox = tracker.update(curr_frame)
        # retinaResult = retinaBbox[frame_ndx]
        # if (retinaResult - [0, 0, 0, 0]).all:
        #     iou = bb_intersection_over_union(bbox, process_bbox(retinaResult, bboxSize))
        if not ok:
            bbox = find_face_in_frame_raw(curr_frame, True)
            # template_images = template_images,
            # template_save_dir = template_save_dir
            if bbox is None:
                assert prev_bbox is not None
                bbox = prev_bbox

            # bbox = process_bbox_raw(bbox, inflation_thresh)

            tracker = cv2.legacy.TrackerMOSSE_create()
            ok = tracker.init(curr_frame, bbox)
            assert ok
        # bbox = process_bbox_raw(bbox, inflation_thresh)
        bounding_boxes_dict[frame_ndx] = bbox

    return bounding_boxes_dict


def track_mouth_raw(frames_dict, inflation_thresh):
    """

    """

    # initialize
    bounding_boxes_dict = {frame_ndx: np.full(4, np.nan) for frame_ndx, _ in frames_dict.items()}

    # find first frame with reliable bounding box. detection will be None if not reliable
    # uses a stride to speed up computation
    seen_frame_indices = []
    for frame_ndx, frame in frames_dict.items():
        seen_frame_indices.append(frame_ndx)
        # if frame_ndx % 4 != 0:
        #     continue
        # initial_bbox = find_face_in_frame(frame, template_images=template_images, template_save_dir=template_save_dir)
        initial_bbox = find_mouth_in_frame_raw(frame, True)
        if initial_bbox is not None:
            break
        # else:
        #     raise RuntimeError('No Face found')

    if initial_bbox is None:
        # logger.error('Could Not Find Face, Please select the bounding box, press (c) if there is no face')
        # initial_bbox = cv2.selectROI(frames_dict[0], False)
        return None

    # initial_bbox = process_bbox_raw(initial_bbox, inflation_thresh)
    first_frame_found_ndx = seen_frame_indices[-1]
    first_frame_found = frames_dict[first_frame_found_ndx]

    # bounding_boxes_dict[first_frame_found_ndx] = process_bbox_raw(initial_bbox, inflation_thresh)
    bounding_boxes_dict[first_frame_found_ndx] = initial_bbox

    tracker = cv2.legacy.TrackerMOSSE_create()
    # tracker = cv2.legacy.TrackerCSRT_create()
    ok = tracker.init(first_frame_found,
                      initial_bbox)
    assert ok

    # next it tracks backward if the first reliable detection was not the first frame
    prev_frame = first_frame_found
    prev_bbox = initial_bbox
    for frame_ndx in reversed(seen_frame_indices[:-1]):
        curr_frame = frames_dict[frame_ndx]
        ok, bbox = tracker.update(curr_frame)
        # retinaResult = retinaBbox[frame_ndx]
        # if (retinaResult - [0, 0, 0, 0]).all:
        # iou = bb_intersection_over_union(bbox, process_bbox(retinaResult,bboxSize))
        if not ok:
            bbox = find_mouth_in_frame_raw(curr_frame, True)
            #                           template_images=template_images,
            #                           template_save_dir=template_save_dir
            if bbox is None:
                assert prev_bbox is not None
                bbox = prev_bbox
            # bbox = process_bbox_raw(bbox, inflation_thresh)
            tracker = cv2.legacy_TrackerMOSSE.create()
            # pip3 install opencv-contrib-python==4.5.5.62 if error comes up
            ok = tracker.init(curr_frame, bbox)
            assert ok
        # bbox = process_bbox_raw(bbox, inflation_thresh)
        bounding_boxes_dict[frame_ndx] = bbox

    # next it tracks forward to the frames that follow
    remaining_frames = sorted(list(set(list(frames_dict.keys())).difference(seen_frame_indices)))
    tracker = cv2.legacy.TrackerMOSSE_create()
    ok = tracker.init(first_frame_found,
                      initial_bbox)
    assert ok
    prev_frame = first_frame_found
    prev_bbox = initial_bbox
    for frame_ndx in remaining_frames:
        # print(frame_ndx)
        curr_frame = frames_dict[frame_ndx]
        ok, bbox = tracker.update(curr_frame)
        # retinaResult = retinaBbox[frame_ndx]
        # if (retinaResult - [0, 0, 0, 0]).all:
        #     iou = bb_intersection_over_union(bbox, process_bbox(retinaResult, bboxSize))
        if not ok:
            bbox = find_mouth_in_frame_raw(curr_frame, True)
            # template_images = template_images,
            # template_save_dir = template_save_dir
            if bbox is None:
                assert prev_bbox is not None
                bbox = prev_bbox

            # bbox = process_bbox_raw(bbox, inflation_thresh)

            tracker = cv2.legacy.TrackerMOSSE_create()
            ok = tracker.init(curr_frame, bbox)
            assert ok
        # bbox = process_bbox_raw(bbox, inflation_thresh)
        bounding_boxes_dict[frame_ndx] = bbox

    return bounding_boxes_dict


def cropFace(raw_frames, bounding_boxes_dict):
    croppedFrame = []
    for i in range(len(bounding_boxes_dict)):
        x, y, w, h = bounding_boxes_dict[i]
        # test = raw_frames[i][y:y+h, x:x+w]
        # cv2.imshow("AI", test)
        # cv2.waitKey(1)
        croppedFrame.append(raw_frames[i][y:y + h, x:x + w])
    return croppedFrame


def trajectoryCal(croppedFrame):
    # 2. read a given video into frame list.
    # Get frame count
    n_frames = len(croppedFrame)

    # Read first frame
    prev = croppedFrame[0]
    # print(prev.shape)

    # Convert frame to grayscale
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    # print(prev_gray.shape)

    # Pre-define transformation-store array
    transforms = np.zeros((n_frames - 1, 3), np.float32)

    for i in range(1, n_frames):
        # Detect feature points in previous frame
        prev_pts = cv2.goodFeaturesToTrack(prev_gray,
                                           maxCorners=200,
                                           qualityLevel=0.001,
                                           minDistance=15,
                                           blockSize=3)
        # plt.figure("Image")  # 图像窗口名称
        # plt.imshow(prev)
        #
        # for ind in range(len(prev_pts)):
        #     x = prev_pts[ind, :, 0]
        #     y = prev_pts[ind, :, 1]
        #     plt.plot(x, y, 'o')
        #
        # plt.show()
        # Read next frame
        curr = croppedFrame[i]
        # print(curr.shape)

        # Convert to grayscale
        curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
        # print(curr_gray.shape)

        # Calculate optical flow (i.e. track feature points)
        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)

        # Sanity check
        assert prev_pts.shape == curr_pts.shape

        # Filter only valid points
        idx = np.where(status == 1)[0]
        prev_pts = prev_pts[idx]
        curr_pts = curr_pts[idx]

        # Find transformation matrix
        m, unknown = cv2.estimateAffinePartial2D(prev_pts, curr_pts)  # will only work with OpenCV-3 or less
        # cv::estimateAffinePartial2D(InputArray_from, InputArray_to)
        # which means: prev_pts * m -> curr_pts

        # Extract traslation
        dx = m[0, 2]
        dy = m[1, 2]

        # Extract rotation angle
        da = np.arctan2(m[1, 0], m[0, 0])

        # Store transformation
        transforms[i - 1] = [dx, dy, da]

        # Move to next frame
        prev_gray = curr_gray

    return transforms


def movingAverage(curve, radius):
    window_size = 2 * radius + 1
    # Define the filter
    f = np.ones(window_size) / window_size
    # Add padding to the boundaries
    curve_pad = np.lib.pad(curve, (radius, radius), 'edge')
    # Apply convolution
    curve_smoothed = np.convolve(curve_pad, f, mode='same')
    # Remove padding
    curve_smoothed = curve_smoothed[radius:-radius]
    # return smoothed curve
    return curve_smoothed


def smooth(trajectory, SMOOTHING_RADIUS):
    smoothed_trajectory = np.copy(trajectory)
    # Filter the x, y and angle curves
    for i in range(3):
        smoothed_trajectory[:, i] = movingAverage(trajectory[:, i], radius=SMOOTHING_RADIUS)

    return smoothed_trajectory


def fixBorder(frame, inflationFactor):
    s = frame.shape
    # Scale the image 4% without moving the center
    T = cv2.getRotationMatrix2D((s[1] / 2, s[0] / 2), 0, inflationFactor)
    frame = cv2.warpAffine(frame, T, (s[1], s[0]))
    return frame


def trajectorySmooth(transforms, SMOOTHING_RADIUS):
    trajectory = np.cumsum(transforms, axis=0)

    # Create variable to store smoothed trajectory
    smoothed_trajectory = smooth(trajectory, SMOOTHING_RADIUS)

    # Calculate difference in smoothed_trajectory and trajectory
    difference = smoothed_trajectory - trajectory

    # Calculate newer transformation array
    transforms_smooth = transforms + difference

    return transforms_smooth


def bboxAffineTran(transforms_smooth_item, frame, inflationFactor):
    # Extract transformations from the new transformation array
    dx = transforms_smooth_item[0]
    dy = transforms_smooth_item[1]
    da = transforms_smooth_item[2]

    # Reconstruct transformation matrix accordingly to new values
    m = np.zeros((2, 3), np.float32)
    m[0, 0] = np.cos(da)
    m[0, 1] = -np.sin(da)
    m[1, 0] = np.sin(da)
    m[1, 1] = np.cos(da)
    m[0, 2] = dx
    m[1, 2] = dy

    # Apply affine wrapping to the given frame
    w = frame.shape[0]
    h = frame.shape[1]
    frame_stabilized = cv2.warpAffine(frame, m, (h, w))

    # Fix border artifacts
    frame_stabilized = fixBorder(frame_stabilized, inflationFactor)

    return frame_stabilized


def chop2VideoVer2(croppedFrame, facial_bounding_boxes, transforms_smooth, save_dir, fps, filename, bboxSize,
                   inflationFactor):
    _, _, w, h = facial_bounding_boxes[0]

    filename = filename.replace('.mp4', '')
    chopped_frames = []
    if save_dir is not None:
        clip_path = os.path.join(save_dir, f'{filename}.mp4')
        out = cv2.VideoWriter(clip_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (bboxSize, bboxSize))

    for i in range(len(croppedFrame) - 1):
        frame = croppedFrame[i]

        frame_stabilized = bboxAffineTran(transforms_smooth[i], frame, inflationFactor)
        frame_stabilized = cv2.resize(frame_stabilized, (int(bboxSize), int(bboxSize)), interpolation=cv2.INTER_AREA)
        chopped_frames.append(frame_stabilized)
        # print(f'output frame size: {frame_stabilized.shape}')
        if save_dir is not None:
            out.write(frame_stabilized)
    
    if save_dir is not None:
        out.release()

    return chopped_frames



def get_corners(bboxes):
    width = (bboxes[2]).reshape(-1, 1)
    height = (bboxes[3]).reshape(-1, 1)

    x1 = bboxes[0].reshape(-1, 1)
    y1 = bboxes[1].reshape(-1, 1)

    x2 = x1 + width
    y2 = y1

    x3 = x1
    y3 = y1 + height

    x4 = (x1 + width).reshape(-1, 1)
    y4 = (y1 + height).reshape(-1, 1)

    corners = np.hstack((x1, y1, x2, y2, x3, y3, x4, y4))

    return corners


def rotate_im(image, angle):
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    image = cv2.warpAffine(image, M, (nW, nH))

    return image


def rotate_box(corners, angle, cx, cy, h, w):
    corners = corners.reshape(-1, 2)
    corners = np.hstack((corners, np.ones((corners.shape[0], 1), dtype=type(corners[0][0]))))

    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)

    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    M[0, 2] += (nW / 2) - cx
    M[1, 2] += (nH / 2) - cy
    calculated = np.dot(M, corners.T).T

    calculated = calculated.reshape(-1, 8)

    return calculated


def get_enclosing_box(corners):
    x_ = corners[:, [0, 2, 4, 6]]
    y_ = corners[:, [1, 3, 5, 7]]

    xmin = np.min(x_, 1).reshape(-1, 1)
    ymin = np.min(y_, 1).reshape(-1, 1)
    xmax = np.max(x_, 1).reshape(-1, 1)
    ymax = np.max(y_, 1).reshape(-1, 1)

    final = np.hstack((xmin, ymin, xmax, ymax, corners[:, 8:]))

    return final


def bbox_area(bbox):
    return (bbox[:, 2] - bbox[:, 0]) * (bbox[:, 3] - bbox[:, 1])


def clip_box(bbox, clip_box, alpha):
    ar_ = (bbox_area(bbox))
    x_min = np.maximum(bbox[:, 0], clip_box[0]).reshape(-1, 1)
    y_min = np.maximum(bbox[:, 1], clip_box[1]).reshape(-1, 1)
    x_max = np.minimum(bbox[:, 2], clip_box[2]).reshape(-1, 1)
    y_max = np.minimum(bbox[:, 3], clip_box[3]).reshape(-1, 1)

    bbox = np.array([x_min, y_min, x_max, y_max])

    return bbox


def rotation(img, bboxes, angle, inflation_thresh):
    w, h = img.shape[1], img.shape[0]
    # print(w)
    cx, cy = w // 2, h // 2
    corners = get_corners(bboxes)
    # print('corner')
    # print(corners)
    img = rotate_im(img, angle)
    corners[:, :8] = rotate_box(corners[:, :8], angle, cx, cy, h, w)
    new_bbox = get_enclosing_box(corners)
    bboxes = new_bbox
    # print('new_bbox')
    # print(bboxes)
    bboxes = clip_box(bboxes, [0, 0, w, h], 0.25).squeeze()
    # print('bboxes')
    # print(bboxes)
    bboxes[2] = bboxes[2] - bboxes[0]
    bboxes[3] = bboxes[3] - bboxes[1]

    # bboxes = process_bbox(bboxes, fixWidth)
    bboxes = process_bbox_raw(bboxes, inflation_thresh)
    return img, bboxes


def horizontalFlip(img, bboxes):
    bboxes = np.array(bboxes)
    img_center = np.array(img.shape[:2])[::-1] / 2
    img = img[:, ::-1, :]
    bboxes[0] = 2 * img_center[0] - bboxes[0] - bboxes[2]

    return img, tuple(bboxes)


def chop2Video(raw_frames, facial_bounding_boxes, save_dir, fps, filename):
    frame_indices = list(raw_frames.keys())
    filename = filename.replace('.mp4', '')
    clip_path = os.path.join(save_dir, f'{filename}.mp4')
    _, _, w, h = facial_bounding_boxes[0]
    out = cv2.VideoWriter(clip_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (h, w))
    for i in tqdm(frame_indices):
        frame = raw_frames[i]
        bbox = facial_bounding_boxes[i]
        x, y, w, h = bbox
        frame = frame[y:y + h, x:x + w]
        out.write(frame)
    out.release()


def adjustBboxDim(bounding_boxes_dict, raw_frames):
    sumup = np.zeros((4,))
    for i in range(len(bounding_boxes_dict)):
        temp = np.array(bounding_boxes_dict[i])
        sumup = sumup + temp
    sumup = sumup / len(bounding_boxes_dict)

    frame_h = raw_frames[0].shape[0]
    frame_w = raw_frames[0].shape[1]

    dim = max(int(sumup[2]), int(sumup[3]))
    # dim = 400
    for i in range(len(bounding_boxes_dict)):
        bounding_boxes_dict[i] = list(bounding_boxes_dict[i])
        # adjust upper left (x,y) into the right range
        bounding_boxes_dict[i][0] = max(min((frame_w - dim - 1), bounding_boxes_dict[i][0]), 0)
        bounding_boxes_dict[i][1] = max(min((frame_h - dim - 1), bounding_boxes_dict[i][1]), 0)

        # adjust bbox dimension into right size
        bounding_boxes_dict[i][2] = dim
        bounding_boxes_dict[i][3] = dim
        bounding_boxes_dict[i] = tuple(bounding_boxes_dict[i])
    return bounding_boxes_dict