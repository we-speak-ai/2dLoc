import os
import sys
import cv2
import time
import numpy as np
import os

THIS_PATH = os.path.dirname(os.path.abspath(__file__))


from localizer import Localizer
from utils import preprocess_frame, imshow, save_training_data
from plot_utils import draw_point, draw_bbox

# Set input/output folders
output_dir = f'{THIS_PATH}/output/'
os.makedirs(output_dir, exist_ok=True)
input_dir = f'{THIS_PATH}/../../../recordings/'


# video_filename = f"V_20240102_134418_OC0.mp4"
# f_roi = (340, 2600)
# v_roi = [900, 3000, -1, -1]
# proc_freq = 2
# res_ratio = 1
#
video_filename = "color/color_ISO1600_EXP800.mp4"
#video_filename = "/mono/mono_ISO1600_EXP800.mp4"
f_roi = (0, 5000)
v_roi = [-1, np.inf, -1, np.inf]
proc_freq = 1
res_ratio = 1

# video_filename = "V_20240109_173420_OC0.mp4"
# video_filename = "V_20240111_123540_OC0.mp4"
# f_roi = (50, 3300)
# v_roi = [600, 2600, 190, 2000]
# proc_freq = 2
# res_ratio = 1

use_resized_frame = res_ratio != 1
last_good_bbox = None


train_data_dir = output_dir + '/train_ds/' + video_filename[:-4] + '/'
os.makedirs(train_data_dir, exist_ok=True)
save_freq = 2

# Initialize Localizer
rough_tracker_model_dir = f'{THIS_PATH}/models/rough_tracker/'
marker_file_path = f'{input_dir}/7x7_1000-18.png'
localizer = Localizer(marker_img_path=marker_file_path, rough_tracker_model_dir=rough_tracker_model_dir)

# Read video
video = cv2.VideoCapture(f"{input_dir}/{video_filename}")
# Exit if video not opened
if not video.isOpened():
     print("Could not open video")
     sys.exit()
# Read first frame
ok, frame = video.read()
if not ok:
     print("Cannot read video file")
     sys.exit()

pp_frame = preprocess_frame(frame, res_ratio, v_roi)

# Run through the boring frames at the beginning of the video
for _ in range(f_roi[0]):
    ok, frame = video.read()

bbox = None
while bbox is None:
    ok, frame = video.read()
    # get initial bounding box
    pp_frame = preprocess_frame(frame, res_ratio, v_roi)
    bbox, box_corners = localizer.find_marker_based_on_kps(pp_frame)

cv2.namedWindow('frm', cv2.WINDOW_NORMAL)
cv2.resizeWindow('frm', 640, 320)
cv2.imshow('frm', pp_frame)
cv2.waitKey()

bbox, box_corners = localizer.update_marker_location(pp_frame)
bbox, box_corners = localizer.update_marker_location(pp_frame)

for fr_idx in range(f_roi[1]):
    # Read a new frame
    video_read_ok, frame = video.read()
    if not video_read_ok:
        break
    pp_frame = preprocess_frame(frame, res_ratio, v_roi)

    # frame skipping
    if fr_idx % proc_freq != 0:
        continue
    frame_cntr = fr_idx//proc_freq

    # Update tracker
    bbox_feat, box_corners = localizer.update_marker_location(pp_frame)
    tracking_feat_ok = bbox_feat is not None
    if tracking_feat_ok:
        last_tracked_frame = np.array(pp_frame)
        last_good_bbox = bbox_feat
        nano_tracking_ok = False
        nano_initialized = False
    elif last_good_bbox is not None:
        if not nano_initialized:
            s1 = 0.5
            s2 = s1 * 2 + 1
            p1 = int(np.max([last_good_bbox[0] - s1 * last_good_bbox[2], 0])), int(np.max([last_good_bbox[1] - s1 * last_good_bbox[3], 0]))
            w, h = int(last_good_bbox[2] * s2), int(last_good_bbox[3] * s2)
            p2 = (int(np.min([p1[0] + w, last_tracked_frame.shape[1]])), int(np.min([p1[1] + h, last_tracked_frame.shape[0]])))
            last_good_bbox = (p1[0], p1[1], w, h)
            nano_tracking_ok = localizer.tracker.init(last_tracked_frame, last_good_bbox)
            nano_initialized = True
        nano_tracking_ok, bbox = localizer.tracker.update(pp_frame)
        localizer.marker_location = bbox
        localizer.age_of_marker_location = np.min([localizer.age_of_marker_location, 1])
        if nano_tracking_ok:
            bbox_feat = bbox

    # Draw bounding box
    if tracking_feat_ok:
        if fr_idx % save_freq == 0:
            save_training_data(pp_frame, bbox_feat, train_data_dir + f'frame_{frame_cntr}')

