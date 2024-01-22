#!/usr/bin/env python3

import depthai as dai
import subprocess

# Create pipeline
pipeline = dai.Pipeline()

# Define sources and outputs
IS_COLOR = True
USE_PREVIEW = True
if IS_COLOR:
    cam = pipeline.create(dai.node.ColorCamera)
    cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)
    cam.setPreviewSize(320, 320)    # NN input
    cam.setInterleaved(False)
    cam.setPreviewKeepAspectRatio(True)
else:
    cam = pipeline.create(dai.node.MonoCamera)
    cam.setCamera("left")
    cam.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)

ve1 = pipeline.create(dai.node.VideoEncoder)
ve1Out = pipeline.create(dai.node.XLinkOut)
ve1Out.setStreamName('ve1Out')

controlIn = pipeline.create(dai.node.XLinkIn)
controlIn.setStreamName('control')
controlIn.out.link(cam.inputControl)

# Properties

# Setting to 26fps will trigger error
ve1.setDefaultProfilePreset(30, dai.VideoEncoderProperties.Profile.H264_MAIN)

# Linking
if IS_COLOR:
    if USE_PREVIEW:
        cam.preview.link(ve1.input)
    else:
        cam.video.link(ve1.input)
else:
    cam.out.link(ve1.input)

ve1.bitstream.link(ve1Out.input)

# Connect to device and start pipeline
with dai.Device(pipeline) as dev:
    # set ISO, EXP

    controlQueue = dev.getInputQueue('control')
    ctrl = dai.CameraControl()
    expTime = 800  #20000  expTime = clamp(expTime, 1, 33000)
    sensIso = 1600  #800     sensIso = clamp(sensIso, 100, 1600)
    ctrl.setManualExposure(expTime, sensIso)
    controlQueue.send(ctrl)

    # Output queues will be used to get the encoded data from the output defined above
    outQ1 = dev.getOutputQueue('ve1Out', maxSize=30, blocking=True)

    # Processing loop
    with open('video.h265', 'wb') as fileColorH265:
        print("Press Ctrl+C to stop encoding...")
        while True:
            try:
                # Empty each queue
                while outQ1.has():
                    outQ1.get().getData().tofile(fileColorH265)

            except KeyboardInterrupt:
                break

    fname_prefix = 'mono'
    if IS_COLOR:
        fname_prefix = 'color'

    command = f"ffmpeg -y -framerate 30 -i video.h265 -c copy ./recordings/{fname_prefix}/{fname_prefix}_ISO{sensIso}_EXP{expTime}.mp4"
    subprocess.run(command, shell=True)
