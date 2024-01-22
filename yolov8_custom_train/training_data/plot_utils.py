import cv2
import numpy as np


def draw_point(img, pt, pt_size=15, color=(0, 0, 0)):
    img_with_pt = np.array(img)
    cv2.line(img_with_pt, (int(pt[0]-pt_size), int(pt[1])), (int(pt[0]+pt_size), int(pt[1])), (0,0,0), 5)
    cv2.line(img_with_pt, (int(pt[0]), int(pt[1]-pt_size)), (int(pt[0]), int(pt[1]+pt_size)), (0,0,0), 5)
    cv2.line(img_with_pt, (int(pt[0]-pt_size), int(pt[1])), (int(pt[0]+pt_size), int(pt[1])), color, 3)
    cv2.line(img_with_pt, (int(pt[0]), int(pt[1]-pt_size)), (int(pt[0]), int(pt[1]+pt_size)), color, 3)
    return img_with_pt


def draw_bbox(img, bbox, color=(255, 0, 0), thickness=2):
    img_with_bbox = np.array(img)
    p1 = (int(bbox[0]), int(bbox[1]))
    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
    cv2.rectangle(img_with_bbox, p1, p2, color=color, thickness=thickness, lineType=1)
    return img_with_bbox
