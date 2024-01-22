# YOLO8 training

## record video
Record 4k video from OAK camera for training
```
$ python video_saver.py
```

## extract every 10th frames from recordings
```
$ ffmpeg -i <input_video.mp4> -vf "select=not(mod(n\,10))" -vsync vfr <./output_location/output_file_name_%03d.jpg>
```

## detect template on frames
Creating img, bbox files
```
$ python dataset_creator.py
```

## create masks based on bboxs
```
$ python dataset_mask.py
```

## generate yolo training data
Put template image into random backgrounds, with random scale, orientation and noise
Code based on https://github.com/alexppppp/synthetic-dataset-object-detection
### python file from notebook
```
$ jupyter nbconvert --to python generator_for_yolov5.ipynb
```
### exec python
```
$ python generator_for_yolov5.py
```

## train
Training using ultralytics package.
Source:
https://github.com/luxonis/depthai-ml-training/tree/master/colab-notebooks
https://github.com/luxonis/depthai-ml-training/blob/master/colab-notebooks/YoloV8_training.ipynb

```
$ export LD_LIBRARY_PATH=

$ yolo train model=yolov8s.pt data=./src/data/dataset/yolo_marker.yaml epochs=500 imgsz=640 batch=64 device=0
$ yolo train model=yolov8n.pt data=./src/data/dataset/yolo_marker.yaml epochs=500 imgsz=[640,360] batch=64 device=0
$ yolo train model=yolov8n.pt data=./src/data/dataset/yolo_marker.yaml epochs=1000 imgsz=320 batch=640 device=0
$ yolo train model=yolov8n.pt data=./src/data/dataset/yolo_marker.yaml epochs=1000 imgsz=480 batch=256 device=0

```

## test infer
```
$ yolo predict model=runs/detect/train/weights/best.pt source='src/data/dataset/valid/images/20.jpg'
```

## generate blob file for OAK camera
https://tools.luxonis.com/

## device run
see ../README.md