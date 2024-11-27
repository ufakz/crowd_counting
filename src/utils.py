import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import time
import tracemalloc
import psutil
from functools import wraps
import csv
from PIL import Image
import os
import pandas as pd


def read_annotations_data(data_path, show_img=False, show_annotations=False):
    image_files = sorted([f for f in os.listdir(data_path) if f.endswith('.jpg')])
    input_images = [cv2.imread(os.path.join(data_path, f)) for f in image_files]
    image_names = image_files
    
    annotations = pd.read_csv(f'{data_path}/manual_annotations.csv')
    annotations['image_id'] = ['image_' + str(i) for i in pd.factorize(annotations['image_name'])[0]]
    
    if show_img:
        fig, axes = plt.subplots(3, len(input_images) // 3, figsize=(15, 15))
        for ax, img, name in zip(axes.flatten(), input_images, image_names):
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            ax.set_title(name)
            ax.axis('off')
        plt.show()
        
    if show_annotations:
        fig, axes = plt.subplots(3, len(input_images) // 3, figsize=(15, 15))
        for ax, img, name in zip(axes.flatten(), input_images, image_names):
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            
            for _, row in annotations[annotations['image_name'] == name].iterrows():
                x, y = row['point_x'], row['point_y']
                ax.plot(x, y, 'ro', markersize=5)
            
            ax.set_title(name)
            ax.axis('off')
        plt.show()
        
    return annotations
    
def show_data_distribution(annotations):
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    annotations['image_name'].value_counts().sort_index().plot(kind='bar', ax=ax[0])
    ax[0].set_title('Annotations per Image')
    ax[0].set_xlabel('Image Name')
    ax[0].set_ylabel('Number of Annotations')
    
    annotations['image_name'].value_counts().plot(kind='pie', ax=ax[1], autopct='%1.1f%%')
    ax[1].set_title('Annotations Distribution (%)')
    plt.show()
    
    
def plot_validation_results(image_level_eval, person_level_eval):
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))

    # Changed to bar chart for detected vs ground truth
    x = range(len(image_level_eval['image_name']))
    width = 0.35
    
    rects1 = axs[0, 0].bar(x, image_level_eval['people_gt'], width, label='Ground Truth')
    rects2 = axs[0, 0].bar([i + width for i in x], image_level_eval['people'], width, label='Detected')
    
    for rect in rects1:
        height = rect.get_height()
        axs[0, 0].text(rect.get_x() + rect.get_width()/2., height,
                       f'{int(height)}',
                       ha='center', va='bottom')
    
    for rect in rects2:
        height = rect.get_height()
        axs[0, 0].text(rect.get_x() + rect.get_width()/2., height,
                       f'{int(height)}',
                       ha='center', va='bottom')
    
    axs[0, 0].set_title('Number of People Detected vs Ground Truth')
    axs[0, 0].set_xlabel('Image ID')
    axs[0, 0].set_ylabel('Number of People')
    axs[0, 0].set_xticks([i + width/2 for i in x])
    axs[0, 0].set_xticklabels(image_level_eval['image_name'], rotation=45)
    axs[0, 0].legend()

    axs[0, 1].plot(image_level_eval['image_name'], image_level_eval['mse'], label='MSE', marker='o')
    axs[0, 1].set_title('Mean Squared Error (MSE)')
    axs[0, 1].set_xlabel('Image ID')
    axs[0, 1].set_ylabel('MSE')
    axs[0, 1].tick_params(axis='x', rotation=45)
    axs[0, 1].legend()

    axs[1, 0].plot(person_level_eval['image_name'], person_level_eval['precision'], label='Precision', marker='o')
    axs[1, 0].plot(person_level_eval['image_name'], person_level_eval['recall'], label='Recall', marker='x')
    axs[1, 0].plot(person_level_eval['image_name'], person_level_eval['f1score'], label='F1 Score', marker='s')
    axs[1, 0].set_title('Precision, Recall, and F1 Score')
    axs[1, 0].set_xlabel('Image Name')
    axs[1, 0].set_ylabel('Score')
    axs[1, 0].tick_params(axis='x', rotation=45)
    axs[1, 0].legend()

    axs[1, 1].plot(person_level_eval['image_name'], person_level_eval['accuracy'], label='Accuracy', marker='o')
    axs[1, 1].set_title('Accuracy')
    axs[1, 1].set_xlabel('Image Name')
    axs[1, 1].set_ylabel('Accuracy')
    axs[1, 1].tick_params(axis='x', rotation=45)
    axs[1, 1].legend()

    plt.tight_layout()
    plt.show()