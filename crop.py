import os
import cv2
import glob
import argparse
import numpy as np
from collections import deque

from utils import *
from transform import *


def load_args(default_config=None):
    parser = argparse.ArgumentParser(description='Lipreading Pre-processing')
    # -- utils
    parser.add_argument('--video-direc', default=None, help='raw video directory')
    parser.add_argument('--landmark-direc', default=None, help='landmark directory')
    parser.add_argument('--filename-path', default='./lrw500_detected_face.csv', help='list of detected video and its subject ID')
    parser.add_argument('--save-direc', default=None, help='the directory of saving mouth ROIs')
    # -- mean face utils
    parser.add_argument('--mean-face', default='./20words_mean_face.npy', help='mean face pathname')
    # -- mouthROIs utils
    parser.add_argument('--crop-width', default=96, type=int, help='the width of mouth ROIs')
    parser.add_argument('--crop-height', default=96, type=int, help='the height of mouth ROIs')
    parser.add_argument('--start-idx', default=48, type=int, help='the start of landmark index')
    parser.add_argument('--stop-idx', default=68, type=int, help='the end of landmark index')
    parser.add_argument('--window-margin', default=12, type=int, help='window margin for smoothed_landmarks')
    # -- convert to gray scale
    parser.add_argument('--convert-gray', default=True, action='store_true', help='convert2grayscale')
    # -- test set only
    parser.add_argument('--testset-only', default=False, action='store_true', help='process testing set only')

    args = parser.parse_args()
    return args

args = load_args()

# -- mean face utils
STD_SIZE = (256, 256)
mean_face_landmarks = np.load('20words_mean_face.npy')
stablePntsIDs = [33, 36, 39, 42, 45]


def crop_patch( video_pathname, landmarks):

    """Crop mouth patch
    :param str video_pathname: pathname for the video_dieo
    :param list landmarks: interpolated landmarks
    """

    frame_idx = 0
    frame_gen = read_video(video_pathname)
    while True:
        try:
            frame = frame_gen.__next__() ## -- BGR
        except Exception as ee:
            print("error",ee)
            break
        if frame_idx == 0:
            q_frame, q_landmarks = deque(), deque()
            sequence = []

        q_landmarks.append(landmarks[frame_idx])
        q_frame.append(frame)
        if len(q_frame) == args.window_margin:
            smoothed_landmarks = np.mean(q_landmarks, axis=0)
            cur_landmarks = q_landmarks.popleft()
            cur_frame = q_frame.popleft()
            # -- affine transformation
            trans_frame, trans = warp_img( smoothed_landmarks[stablePntsIDs, :],
                                           mean_face_landmarks[stablePntsIDs, :],
                                           cur_frame,
                                           STD_SIZE)
            trans_landmarks = trans(cur_landmarks)
            # -- crop mouth patch
            sequence.append( cut_patch( trans_frame,
                                        trans_landmarks[args.start_idx:args.stop_idx],
                                        args.crop_height//2,
                                        args.crop_width//2,))
        if frame_idx == len(landmarks)-1:
            while q_frame:
                cur_frame = q_frame.popleft()
                # -- transform frame
                trans_frame = apply_transform( trans, cur_frame, STD_SIZE)
                # -- transform landmarks
                trans_landmarks = trans(q_landmarks.popleft())
                # -- crop mouth patch
                sequence.append( cut_patch( trans_frame,
                                            trans_landmarks[args.start_idx:args.stop_idx],
                                            args.crop_height//2,
                                            args.crop_width//2,))
            return np.array(sequence)
        frame_idx += 1
    return None


def landmarks_interpolate(landmarks):
    
    """Interpolate landmarks
    param list landmarks: landmarks detected in raw videos
    """

    valid_frames_idx = [idx for idx, _ in enumerate(landmarks) if _ is not None]
    if not valid_frames_idx:
        return None
    for idx in range(1, len(valid_frames_idx)):
        if valid_frames_idx[idx] - valid_frames_idx[idx-1] == 1:
            continue
        else:
            landmarks = linear_interpolate(landmarks, valid_frames_idx[idx-1], valid_frames_idx[idx])
    valid_frames_idx = [idx for idx, _ in enumerate(landmarks) if _ is not None]
    # -- Corner case: keep frames at the beginning or at the end failed to be detected.
    if valid_frames_idx:
        landmarks[:valid_frames_idx[0]] = [landmarks[valid_frames_idx[0]]] * valid_frames_idx[0]
        landmarks[valid_frames_idx[-1]:] = [landmarks[valid_frames_idx[-1]]] * (len(landmarks) - valid_frames_idx[-1])
    valid_frames_idx = [idx for idx, _ in enumerate(landmarks) if _ is not None]
    assert len(valid_frames_idx) == len(landmarks), "not every frame has landmark"
    return landmarks



if True:

    
    video_pathname =  'test.mp4'
    landmarks_pathname ="ltmp.npz"
    dst_pathname = 'crop.npz'

   

    

    multi_sub_landmarks = np.load( landmarks_pathname, allow_pickle=True)['data']
    print(len(multi_sub_landmarks))
    landmarks = [None] * len( multi_sub_landmarks)
    for frame_idx in range(len(landmarks)):
        try:
            landmarks[frame_idx] = multi_sub_landmarks[frame_idx]['facial_landmarks']
        except IndexError:
            continue

    
    
    # -- crop
    sequence = crop_patch(video_pathname, landmarks)
    

    # -- save
    data = convert_bgr2gray(sequence) if args.convert_gray else sequence[...,::-1]
    save2npz("cropped/"+dst_pathname, data=data)

print('Done.')
