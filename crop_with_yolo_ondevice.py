#!/usr/bin/env python3

# based on  : https://github.com/luxonis/depthai-experiments/blob/master/gen2-lossless-zooming/main.py
# sync      : https://github.com/luxonis/depthai-experiments/blob/master/gen2-syncing/host-nn-sync.py

from pathlib import Path
import cv2
import depthai as dai
import time
import os
import json
import numpy as np
from collections import deque


class HostSync:
    def __init__(self):
        self.msgs = {
            'det': deque(),
            'crop': deque()
        }
    def add_det(self, msg):
        self.msgs['det'].append(msg)
    def add_crop(self, msg):
        self.msgs['crop'].append(msg)

    def get_msgs(self):
        if not self.msgs['det'] or not self.msgs['crop']:
            # empty
            return None, None
        # get seq num of msgs
        seq_det = self.msgs['det'][0].getSequenceNum()
        seq_crop = self.msgs['crop'][0].getSequenceNum()
        
        if seq_det == seq_crop:
            # sync
            return self.msgs['det'].popleft(), self.msgs['crop'].popleft()
        if seq_det + 5 < seq_crop or seq_det > seq_crop:
            # no detection
            return None, self.msgs['crop'].popleft()

THIS_PATH = os.path.dirname(os.path.abspath(__file__))
# Start defining a pipeline
pipeline = dai.Pipeline()

# Define a source - color camera
cam = pipeline.create(dai.node.ColorCamera)
cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)

# Create Yolo detection network
# 10k image train, 480x480 nn input
blob_path = f'{THIS_PATH}/yolov8_custom_train/20240119_train_yolov8n_10kimg_480x/blob_320x192/'
model_path = f'{blob_path}/best_openvino_2022.1_6shave.blob'
config_path = f'{blob_path}/best.json'

# parse config
with Path(config_path).open() as f:
    config = json.load(f)
nnConfig = config.get("nn_config", {})

# parse input shape
if "input_size" in nnConfig:
    nn_w, nn_h = tuple(map(int, nnConfig.get("input_size").split('x')))
    print(f'NN input size = {nn_w}x{nn_h}')

# Constants
ORIGINAL_SIZE = (int(2160/nn_h*nn_w), 2160) # 4K (3840 x 2160) - should be the same AR as nn input
if ORIGINAL_SIZE[0] > 3840:
    ORIGINAL_SIZE = (3840, int(3840/nn_w*nn_h))
CROP_SIZE = (500, 500)      # crop size in pixels

# extract metadata
metadata = nnConfig.get("NN_specific_metadata", {})
classes = metadata.get("classes", {})
coordinates = metadata.get("coordinates", {})
anchors = metadata.get("anchors", {})
anchorMasks = metadata.get("anchor_masks", {})
iouThreshold = metadata.get("iou_threshold", {})
confidenceThreshold = metadata.get("confidence_threshold", {})
print('NN parameters:')
print(f'metadata = {metadata}')
print(f'classes = {classes}')
print(f'coordinates = {coordinates}')
print(f'anchors = {anchors}')
print(f'anchorMasks = {anchorMasks}')
print(f'iouThreshold = {iouThreshold}')
print(f'confidenceThreshold = {confidenceThreshold}')

# parse labels
nnMappings = config.get("mappings", {})
labels = nnMappings.get("labels", {})

nnMappings.get("labels", {})

# get model path
if not Path(model_path).exists():
    print("No blob found at {}".format(model_path))
    exit()
detectionNetwork = pipeline.create(dai.node.YoloDetectionNetwork)
xoutDet = pipeline.create(dai.node.XLinkOut)
xoutDet.setStreamName('det')

# Network specific settings
detectionNetwork.setConfidenceThreshold(confidenceThreshold)
detectionNetwork.setNumClasses(classes)
detectionNetwork.setCoordinateSize(coordinates)
detectionNetwork.setAnchors(anchors)
detectionNetwork.setAnchorMasks(anchorMasks)
detectionNetwork.setIouThreshold(iouThreshold)
detectionNetwork.setBlobPath(model_path)
detectionNetwork.setNumInferenceThreads(2)
detectionNetwork.input.setBlocking(True)

# Squash whole 4K frame into nn size
cam.setVideoSize(ORIGINAL_SIZE[0], ORIGINAL_SIZE[1])
cam.setPreviewSize(nn_w, nn_h)
cam.setInterleaved(False)
cam.setPreviewKeepAspectRatio(True)
# cam.initialControl.setManualFocus(130)

cam.preview.link(detectionNetwork.input)

def frameNorm(frame_shape_1, frame_shape_0, bbox):
    normVals = np.full(len(bbox), frame_shape_0)
    normVals[::2] = frame_shape_1
    return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

def get_best_detection(dets):
    # find detection with highest confidence
    best_detection = None
    for detection in dets:
        if best_detection is None or detection.confidence > best_detection.confidence:
            best_detection = detection
    return best_detection
    
def get_bbox_on_original_imgs(dets):
    bd = get_best_detection(dets)
    if bd is not None:    
        # Get detection center
        center_x = int((bd.xmin + bd.xmax) / 2 * ORIGINAL_SIZE[0])
        center_y = int((bd.ymin + bd.ymax) / 2 * ORIGINAL_SIZE[1])
        bbox = frameNorm(ORIGINAL_SIZE[0], ORIGINAL_SIZE[1], (bd.xmin, bd.ymin, bd.xmax, bd.ymax))
        return (center_x, center_y, bbox)
    return None

# cam control
controlIn = pipeline.create(dai.node.XLinkIn)
controlIn.setStreamName('control')
controlIn.out.link(cam.inputControl)

# Script node
script = pipeline.create(dai.node.Script)
detectionNetwork.out.link(script.inputs['dets'])
detectionNetwork.out.link(xoutDet.input)

script.setScript(f"""
ORIGINAL_SIZE = {ORIGINAL_SIZE} # 4K
CROP_SIZE = {CROP_SIZE} # crop

cfg = ImageManipConfig()
size = Size2f(CROP_SIZE[0], CROP_SIZE[1])
# initial crop to center
rect = RotatedRect()
rect.size = size
rect.center = Point2f(ORIGINAL_SIZE[0]//2, ORIGINAL_SIZE[1]//2)

while True:
    dets = node.io['dets'].get().detections
    if len(dets) == 0:
        # use old rect
        cfg.setCropRotatedRect(rect, False)
        node.io['cfg'].send(cfg)
        continue
    
    # find detection with most confidence
    best_detection = None
    for detection in dets:
        if best_detection is None or detection.confidence > best_detection.confidence:
            best_detection = detection
    if best_detection is not None:    
        coords = best_detection
        # Get detection center
        x = int((coords.xmin + coords.xmax) / 2 * ORIGINAL_SIZE[0])
        y = int((coords.ymin + coords.ymax) / 2 * ORIGINAL_SIZE[1])

        rect.size = size
        rect.center = Point2f(x, y)
        cfg.setCropRotatedRect(rect, False)
        node.io['cfg'].send(cfg)
""")
crop_manip = pipeline.create(dai.node.ImageManip)
crop_manip.setMaxOutputFrameSize(CROP_SIZE[0] * CROP_SIZE[1] * 3)
crop_manip.setWaitForConfigInput(True)

script.outputs['cfg'].link(crop_manip.inputConfig)
cam.video.link(crop_manip.inputImage)

xout = pipeline.create(dai.node.XLinkOut)
xout.setStreamName('crop')
crop_manip.out.link(xout.input)

xoutNNin = pipeline.create(dai.node.XLinkOut)
xoutNNin.setStreamName('nnin')
cam.preview.link(xoutNNin.input)

fpss = {}

def clamp(num, v0, v1):
    return max(v0, min(num, v1))
def displayFrame(name, frame, dets = []):
    if len(dets):
        detection_center_x, detection_center_y, bbox = get_bbox_on_original_imgs(dets)
        crop_tl_x = detection_center_x - CROP_SIZE[0] // 2
        crop_tl_y = detection_center_y - CROP_SIZE[1] // 2
        # print(crop_tl_x, crop_tl_y)
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

    frm_size_based_scale = frame.shape[0]/640
    if name in fpss:
        fps = fpss[name]['fps']
        cv2.putText(frame, "fps: {:.2f}".format(fps), (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, fontScale=frm_size_based_scale, color=(255, 0, 0), thickness=int(frm_size_based_scale))
    # Show the frame
    cv2.imshow(name, frame)
def calcFps(name):
    if name not in fpss:
        fpss[name] = {
            'fps': 0,
            'counter': 0,
            'startTime': time.monotonic()
        }
    fpss[name]['counter']+=1
    current_time = time.monotonic()
    if (current_time - fpss[name]['startTime']) > 1 :
        fpss[name]['fps'] = fpss[name]['counter'] / (current_time - fpss[name]['startTime'])
        fpss[name]['counter'] = 0
        fpss[name]['startTime'] = current_time
        

cv2.namedWindow('nnin', cv2.WINDOW_NORMAL)
cv2.resizeWindow('nnin', 888, 500)

cv2.namedWindow('paint', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
cv_paint_img = np.zeros(shape=(ORIGINAL_SIZE[1],ORIGINAL_SIZE[0],3), dtype=np.uint8)
cv2.resizeWindow('paint', 1280, 724)

with dai.Device(pipeline) as device:
    sync = HostSync()

    qCrop = device.getOutputQueue(name='crop')
    qNNin = device.getOutputQueue(name='nnin')
    qDet = device.getOutputQueue(name="det")
    qControl = device.getInputQueue('control')
    EXP_STEP = 500  # us
    ISO_STEP = 50
    expTime = 6000
    sensIso = 1500  
    
    # Main loop
    while True:
        if qCrop.has():
            crop_elem = qCrop.get()
            sync.add_crop(crop_elem)
            calcFps('crop')
            # crop_frame = crop_elem.getCvFrame()
            # displayFrame('crop', crop_frame)

        if qNNin.has():
            NNin_elem = qNNin.get()
            # print('nnin: ', NNin_elem.getSequenceNum())
            NNin_frame = NNin_elem.getCvFrame()
            calcFps('nnin')
            displayFrame('nnin', NNin_frame)

        if qDet.has():
            detIn = qDet.get()
            sync.add_det(detIn)

        # get synchronized msgs
        det_msg, crop_msg = sync.get_msgs()
        if crop_msg is not None:
            crop_frame = crop_msg.getCvFrame()
            #displayFrame('crop', crop_frame, det_msg.detections if det_msg else None)
            displayFrame('crop', crop_frame)

            # test bbox-image synch with inpaint
            if len(det_msg.detections):
                detection_center_x, detection_center_y, bbox = get_bbox_on_original_imgs(det_msg.detections)
                crop_tl_x = detection_center_x - CROP_SIZE[0] // 2
                crop_tl_y = detection_center_y - CROP_SIZE[1] // 2
                cv_paint_img[crop_tl_y:crop_tl_y+crop_frame.shape[0], crop_tl_x:crop_tl_x+crop_frame.shape[1]] = crop_frame
                cv2.imshow('paint', cv_paint_img)

        # Update screen (1ms pooling rate)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key in [ord('i'), ord('o'), ord('k'), ord('l')]:
            if key == ord('i'): expTime -= EXP_STEP
            if key == ord('o'): expTime += EXP_STEP
            if key == ord('k'): sensIso -= ISO_STEP
            if key == ord('l'): sensIso += ISO_STEP
            expTime = clamp(expTime, 1, 33000)
            sensIso = clamp(sensIso, 100, 1600)
            print("Setting manual exposure, time: ", expTime, "iso: ", sensIso)
            ctrl = dai.CameraControl()
            ctrl.setManualExposure(expTime, sensIso)
            qControl.send(ctrl)