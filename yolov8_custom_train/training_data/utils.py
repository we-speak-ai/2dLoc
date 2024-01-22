import numpy as np
import cv2


def preprocess_frame(frame, res_ratio, roi):
    frame = get_video_roi(frame, roi)
    if 0 < res_ratio < 1.0:
        frame = cv2.resize(frame, dsize=[int(frame.shape[1] * res_ratio), int(frame.shape[0] * res_ratio)])
    return frame


def get_video_roi(frame, roi):
    if roi is not None:
        roi[0] = int(np.max([roi[0], 0]))
        roi[1] = int(np.min([roi[1], frame.shape[1]]))
        roi[2] = int(np.max([roi[2], 0]))
        roi[3] = int(np.min([roi[3], frame.shape[0]]))
        frame = frame[roi[2]:roi[3], roi[0]: roi[1]]
    return frame


def imshow(window_title, img, max_size=(900, 1600)):
    res_ratio = np.max([img.shape[0]/max_size[0], img.shape[1]/max_size[1]])
    if res_ratio > 1:
        new_size = (int(img.shape[1]/res_ratio), int(img.shape[0]/res_ratio))
        img = cv2.resize(img, new_size)
    cv2.imshow(window_title, img)


def save_training_data(img, bbox, base_output_path):
    cv2.imwrite(base_output_path + '.jpg', img)
    with open(base_output_path + '.txt', 'w') as f:
        f.write(f"{bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}")




