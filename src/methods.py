import numpy as np
import matplotlib.pyplot as plt
import cv2
from collections import defaultdict

def get_canny_edges(target_img: np.ndarray, base_img: np.ndarray, normalize = False) -> np.ndarray:
    
    x = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY) # x is the target image
    y = cv2.cvtColor(base_img, cv2.COLOR_BGR2GRAY)   # y is the base image

    x = cv2.GaussianBlur(x, (5,5), 0)
    
    # Adaptive Histogram Equalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    x = clahe.apply(x)

    # Filter the target to smooth sunny artifacts 
    x = cv2.bilateralFilter(x,9,127,127)

    x = cv2.absdiff(x, y)

    if normalize:
        x = ((x - np.min(x) / np.max(x) - np.min(x)) * 255.0).astype('uint8')

    x= cv2.Canny(x, 100, 200)
    x = cv2.dilate(x, np.ones((3,3)))
    x = cv2.morphologyEx(x, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1,1)))
    
    return x

def get_average_image(target_img: np.ndarray, base_img: np.ndarray, alpha = 0.5) -> np.ndarray:
    
    beta = 1 - alpha
    avg_img = cv2.addWeighted(target_img, alpha, base_img, beta, 0.0)
    
    return avg_img

def rgb_distance_mask(target_img: np.ndarray, base_img: np.ndarray, thresh: int, morph = False) -> np.ndarray:
    
    diff = np.zeros_like(target_img)
    diff_single= np.zeros_like(target_img[:,:,0])
    for i in range(3):
        diff[:,:,i] = cv2.absdiff(target_img[:,:,i], base_img[:,:,i])
        diff[:,:,i][diff[:,:,i] > thresh] = 255
        diff[:,:,i][diff[:,:,i] < thresh] = 0

    diff_single = cv2.bitwise_and(diff[:,:,0], diff[:,:,1])
    diff_single = cv2.bitwise_and(diff[:,:,1], diff[:,:,2])
    if morph:
        # Helps remove noise
        mask_morph = cv2.morphologyEx(diff_single, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        mask_morph = cv2.morphologyEx(mask_morph, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        output = cv2.bitwise_and(diff_single, diff_single, None, mask_morph)
        
        return output
    
    return diff_single

def create_dictionary_image_points(annotations) -> defaultdict:

    data_manual_points = defaultdict(lambda: {'coordinates': []})

    for index, row in annotations.iterrows():
        image_name=row['image_name']
        point_x=int(row['point_x'])
        point_y=int(row['point_y'])
        data_manual_points[image_name]['coordinates'].append((point_x, point_y))
    return data_manual_points