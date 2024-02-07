import cv2
import os

dir_2_inv = os.path.dirname(os.path.realpath(__file__)) + '/data/ppf/masks'
inverter_dir = os.path.dirname(os.path.realpath(__file__)) + '/data/ppf/masks_inv'
os.makedirs(inverter_dir, exist_ok=True)

for f in os.listdir(dir_2_inv):
    if f.endswith('.png'):
        img = cv2.imread(dir_2_inv + '/' + f)
        img_not = cv2.bitwise_not(img)
        cv2.imwrite(f'{inverter_dir}/{f}', img_not)