import cv2
from utils import imshow
import os

THIS_PATH = os.path.dirname(os.path.abspath(__file__))


img_folder = f'{THIS_PATH}/output/train_ds/mono/mono_ISO1600_EXP800/'
img_files = []
# Iterate directory
for file in os.listdir(img_folder):
    # check only text files
    if file.endswith('.jpg'):
        img_files.append(file)

for ifile in img_files:
    img_file = f'{img_folder}/{ifile}'
    bbox_file = f'{img_folder}/{ifile.split(".")[0]}.txt'

    img = cv2.imread(img_file)
    with open(bbox_file, 'r')as f:
        bbox = f.readline().split(' ')

    bbox = [int(i) for i in bbox]

    p1 = (int(bbox[0]), int(bbox[1]))
    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
    cv2.rectangle(img, p1, p2, (255, 255, 0), 2, 1)
    imshow('img with rect', img)
    cv2.waitKey()