# 2dLoc
Train and run yolo detection on OAK camera

## Train
see yolo8_custom_train/README.md

## Infer
### Host process
Yolo detection results received in the host, send crop config back to device - problem: delay
```
$ python crop_with_yolo_host_proc.py
```

### Device only
Using on-device scripting it is possible to connect the yolo detection ROI to crop config without host processing
```
$ python crop_with_yolo_ondevice.py
```