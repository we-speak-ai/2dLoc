#!/usr/bin/env python3

from pathlib import Path
import cv2
import depthai as dai
import numpy as np
import time
import os
import json

THIS_PATH = os.path.dirname(os.path.abspath(__file__))
USE_MONO_FOR_DETECTION = False  # using mono camera fro detection
CROP_EN = True                  # crop or send the whole frame
cropWpix, cropHpix = 300, 300   # 4k crop roi
rgb_w, rgb_h = 2160, 2160       # should be the same AR as crop roi

##### NN

# parse arguments
# # 1k image train
# blob_path = f'{THIS_PATH}/yolov8_custom_train/20240115_train_yolov8n_marker/blob_RVC2_shaves6_320x/'
# model_path = f'{blob_path}yolov8n_marker_trained_openvino_2022.1_6shave.blob'
# config_path = f'{blob_path}yolov8n_marker_trained.json'

# # 10k image train, 320x320 nn input
# blob_path = f'{THIS_PATH}/yolov8_custom_train/20240119_train_yolov8n_10kimg_320x/blob_320x/'
# model_path = f'{blob_path}/best_openvino_2022.1_6shave.blob'
# config_path = f'{blob_path}/best.json'

# 10k image train, 480x480 nn input
blob_path = f'{THIS_PATH}/yolov8_custom_train/20240119_train_yolov8n_10kimg_480x/blob_256x/'
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

# extract metadata
metadata = nnConfig.get("NN_specific_metadata", {})
classes = metadata.get("classes", {})
coordinates = metadata.get("coordinates", {})
anchors = metadata.get("anchors", {})
anchorMasks = metadata.get("anchor_masks", {})
iouThreshold = metadata.get("iou_threshold", {})
confidenceThreshold = metadata.get("confidence_threshold", {})

# parse labels
nnMappings = config.get("mappings", {})
labels = nnMappings.get("labels", {})

nnMappings.get("labels", {})

# get model path
if not Path(model_path).exists():
    print("No blob found at {}".format(model_path))
    exit()

######
# Create pipeline
pipeline = dai.Pipeline()

# Define sourcesa)
camRgb = pipeline.create(dai.node.ColorCamera)
if USE_MONO_FOR_DETECTION:
    camMono = pipeline.create(dai.node.MonoCamera)
    manipMono = pipeline.create(dai.node.ImageManip)
if CROP_EN:
    cropRgb = pipeline.create(dai.node.ImageManip)
    configIn = pipeline.create(dai.node.XLinkIn)
    configIn.setStreamName('config')

detectionNetwork = pipeline.create(dai.node.YoloDetectionNetwork)
# Define outputs
xoutNNin = pipeline.create(dai.node.XLinkOut)
xoutOut = pipeline.create(dai.node.XLinkOut)
xoutNNout = pipeline.create(dai.node.XLinkOut)

xoutNNin.setStreamName("nnin")
xoutNNout.setStreamName("nn")
xoutOut.setStreamName("out")

# Network specific settings
detectionNetwork.setConfidenceThreshold(confidenceThreshold)
detectionNetwork.setNumClasses(classes)
detectionNetwork.setCoordinateSize(coordinates)
detectionNetwork.setAnchors(anchors)
detectionNetwork.setAnchorMasks(anchorMasks)
detectionNetwork.setIouThreshold(iouThreshold)
detectionNetwork.setBlobPath(model_path)
detectionNetwork.setNumInferenceThreads(2)
detectionNetwork.input.setBlocking(False)

# Properties
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)    # 3840x2160
if USE_MONO_FOR_DETECTION:
    camMono.setCamera("left")
    camMono.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
    manipMono.initialConfig.setResize(nn_w, nn_h)
    manipMono.initialConfig.setKeepAspectRatio(True)
    # The NN model expects BGR input. By default ImageManip output type would be same as input (gray in this case)
    manipMono.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)
else:
    camRgb.setPreviewSize(nn_w, nn_h)    # NN input
    camRgb.setInterleaved(False)
    camRgb.setPreviewKeepAspectRatio(True)
camRgb.setVideoSize(rgb_w, rgb_h)

camW, camH = camRgb.getVideoWidth(), camRgb.getVideoHeight()
maxFrameSize = cropHpix * cropWpix * 3
cropRgb.setMaxOutputFrameSize(maxFrameSize)
cropW, cropH = float(f'{cropWpix / camW:0.3f}'), float(f'{cropHpix / camH:0.3f}')

def get_crop_params(cropCenterX=0.1, cropCenterY=0.1):
    cropRectXmin, cropRectYmin = cropCenterX - cropW/2, cropCenterY - cropH/2
    cropRectXmax, cropRectYmax = cropCenterX + cropW/2, cropCenterY + cropH/2
    cropRectXmin = max(cropRectXmin, 0)
    cropRectYmin = max(cropRectYmin, 0)
    cropRectXmax = min(cropRectXmax, 1)
    cropRectYmax = min(cropRectYmax, 1)
    if cropRectXmin == 0:
        cropRectXmax = cropW
    if cropRectYmin == 0:
        cropRectYmax = cropH
    if cropRectXmax == 0:
        cropRectXmin = 1.0-cropW
    if cropRectYmax == 0:
        cropRectYmin = 1.0-cropH
    # if (cropRectXmax - cropRectXmin) % 2:
    #     cropRectXmax -= 1
    # if (cropRectYmax - cropRectYmin) % 3:
    #     cropRectYmax -= ((cropRectYmax - cropRectYmin) % 3)
        
    return cropRectXmin, cropRectYmin, cropRectXmax, cropRectYmax

if CROP_EN:
    cropRectXmin, cropRectYmin, cropRectXmax, cropRectYmax = get_crop_params()
    cropRgb.initialConfig.setCropRect(cropRectXmin, cropRectYmin, cropRectXmax, cropRectYmax)

# Linking
if USE_MONO_FOR_DETECTION:
    camMono.out.link(manipMono.inputImage)
    manipMono.out.link(xoutNNin.input)
    manipMono.out.link(detectionNetwork.input)
else:
    camRgb.preview.link(xoutNNin.input)
    camRgb.preview.link(detectionNetwork.input)
detectionNetwork.out.link(xoutNNout.input)
if CROP_EN:
    camRgb.video.link(cropRgb.inputImage)
    cropRgb.out.link(xoutOut.input)
    configIn.out.link(cropRgb.inputConfig)
else:
    camRgb.video.link(xoutOut.input)

# Connect to device and start pipeline
with dai.Device(pipeline) as device:
    if USE_MONO_FOR_DETECTION:
        calibData = device.readCalibration()
        l_rgb_extrinsics = np.array(calibData.getCameraExtrinsics(dai.CameraBoardSocket.CAM_B, dai.CameraBoardSocket.CAM_A))
        print("Transformation matrix of where left Camera is W.R.T RGB Camera's optical center")
        print(l_rgb_extrinsics)

    if CROP_EN:
        configQueue = device.getInputQueue(configIn.getStreamName())

    # Output queues will be used to get the frames and nn data from the outputs defined above
    qNNin = device.getOutputQueue(name="nnin", maxSize=4, blocking=False)
    qOut = device.getOutputQueue(name="out", maxSize=4, blocking=False)
    qDet = device.getOutputQueue(name="nn", maxSize=4, blocking=False)
    cfg = dai.ImageManipConfig()
                
    nninFrame = None
    color = (255, 0, 0)
    detections = []
    fpss = {}

    # nn data, being the bounding box locations, are in <0..1> range - they need to be normalized with frame width/height
    def frameNorm(frame_shape_0, frame_shape_1, bbox):
        normVals = np.full(len(bbox), frame_shape_0)
        normVals[::2] = frame_shape_1
        return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

    def displayFrame(name, frame):
        frm_size_based_scale = frame.shape[0]/640
        #print(frame.shape)
        if name in fpss:
            fps = fpss[name]['fps']
            cv2.putText(frame, "fps: {:.2f}".format(fps), (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, fontScale=frm_size_based_scale, color=color, thickness=int(frm_size_based_scale))
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

    cv2.namedWindow('out', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('out', 1280, 720)

    while True:
        # Instead of get (blocking), we use tryGet (non-blocking) which will return the available data or None otherwise
        nnin_ = qNNin.tryGet()
        out_rgb_ = qOut.tryGet()
        nnout_ = qDet.tryGet()

        if nnin_ is not None:
            NNinFrame = nnin_.getCvFrame()
            calcFps('nnin')
            for detection in detections:
                bbox = frameNorm(nn_w, nn_h, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
                cv2.putText(NNinFrame, f"{int(detection.confidence * 100)}%", (bbox[0] + 10, bbox[1] + 40), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                cv2.rectangle(NNinFrame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            displayFrame("nnin", NNinFrame)

        if out_rgb_ is not None:
            outFrame = out_rgb_.getCvFrame()
            calcFps('out')
            displayFrame('out', outFrame)
        
        if nnout_ is not None:
            detections = nnout_.detections
            best_detection = None
            for detection in detections:
                if best_detection is None or detection.confidence > best_detection.confidence:
                    best_detection = detection
            if best_detection is not None:
                # # AR correction
                # camVW, camVH = camRgb.getVideoWidth(), camRgb.getVideoHeight()
                # camPW, camPH = camRgb.getPreviewWidth(), camRgb.getPreviewHeight()
                # ar_corr = (camVW / camVH) / (camPW / camPH)
                # detection.ymin /= ar_corr
                # detection.ymax /= ar_corr
                
                if CROP_EN:
                    cropCenterX = (detection.xmin + detection.xmax) / 2
                    cropCenterY = (detection.ymin + detection.ymax) / 2
                    
                    cropRectXmin, cropRectYmin, cropRectXmax, cropRectYmax = get_crop_params(cropCenterX, cropCenterY)
                    # print(cropCenterX, cropCenterY)
                    
                    cfg.setCropRect(cropRectXmin, cropRectYmin, cropRectXmax, cropRectYmax)
                    configQueue.send(cfg)
                    
            calcFps('nnout')

        if cv2.waitKey(1) == ord('q'):
            break