import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import pandas as pd

from src.PerformanceMonitor import PerformanceMonitor

class ConnectedComponents:
    def __init__(self, manual_annotations):
        self.performance = PerformanceMonitor()
        self.annotations_df = manual_annotations
        self.predictions_df = None
        
    @property
    def metrics(self):
        return self.performance.get_metrics()

    def load_image(self, image_name, show_image=False):
        image_path = f"data/{image_name}"
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            print(f"The image cannot be loaded: {image_path}")
            return None
        if show_image:
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.title('Loaded Image')
            plt.show()
        return image
    
    def get_canny_edges(self, target_img: np.ndarray, base_img: np.ndarray, show_image=False) -> np.ndarray:
        x = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY) # x is the target image
        y = cv2.cvtColor(base_img, cv2.COLOR_BGR2GRAY)   # y is the base image

        x = cv2.GaussianBlur(x, (5,5), 0)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        x = clahe.apply(x)

        x = cv2.bilateralFilter(x,9,127,127)
        x = cv2.absdiff(x, y)

        x= cv2.Canny(x, 100, 200)
        x = cv2.dilate(x, np.ones((3,3)))
        x = cv2.morphologyEx(x, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1,1)))
        
        if show_image:
            plt.imshow(x, cmap='gray')
            plt.title('Canny Edges')
            plt.show()
        
        return x
    
    def rgb_distance_mask(self, target_img: np.ndarray, base_img: np.ndarray, thresh: int, morph = False, show_image=False) -> np.ndarray:
        diff = np.zeros_like(target_img)
        diff_single= np.zeros_like(target_img[:,:,0])
        for i in range(3):
            diff[:,:,i] = cv2.absdiff(target_img[:,:,i], base_img[:,:,i])
            diff[:,:,i][diff[:,:,i] > thresh] = 255
            diff[:,:,i][diff[:,:,i] < thresh] = 0

        diff_single = cv2.bitwise_and(diff[:,:,0], diff[:,:,1])
        diff_single = cv2.bitwise_and(diff[:,:,1], diff[:,:,2])
        
        if morph:
            mask_morph = cv2.morphologyEx(diff_single, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
            mask_morph = cv2.morphologyEx(mask_morph, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
            output = cv2.bitwise_and(diff_single, diff_single, None, mask_morph)
            
            return output

        if show_image:
            plt.imshow(diff_single, cmap='gray')
            plt.title('RGB Distance Mask')
            plt.show()
        
        return diff_single
    
    def process_image(self, image_name, image_id, binary_image, bckg_image, thresh, morph, show_image=False):
        return self.performance.measure_performance(self._process_image, image_name)(image_id, image_name, binary_image, bckg_image, thresh, morph, show_image)
    
    def _process_image(self, image_id, image_name, binary_image, bckg_image, thresh, morph, show_image=False):
        points = []
        
        bel1 = self.get_canny_edges(binary_image, bckg_image, show_image)
        bel2 = self.rgb_distance_mask(binary_image, bckg_image, thresh, morph, show_image)
        
        mask = cv2.bitwise_or(bel1, bel2)
        mask = cv2.erode(mask, np.ones((5,3)))
        
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        
        for i in range(1, num_labels):
            area = stats[i][cv2.CC_STAT_AREA]
            centroid = centroids[i]
            
            if (50 < area < 1000 and centroid[1] > 450):
                points.append({'point_x': int(centroid[0]), 'point_y': int(centroid[1]), 'image_id': f'image_{image_id}', 'image_name': image_name})
        
        return points

    def process_images(self, thresh=80, morph=False, show_image=False):
        processed_images = []
        for idx, name in enumerate(self.annotations_df['image_name'].unique()[1:], start=1):
            image = self.load_image(name, show_image)
            background = self.load_image(self.annotations_df.iloc[0]['image_name'], show_image)
            points = self.process_image(name, idx, image, background, thresh, morph, show_image)
            
            processed_images.extend(points)
                
        self.predictions_df = pd.DataFrame(processed_images)
        return self.predictions_df
    
    def image_level_validation(self):
        pred_counts = self.predictions_df.groupby('image_id')['image_name'].agg(['count', 'first']).reset_index()
        pred_counts.columns = ['image_id', 'people', 'image_name']
        
        annot_counts = self.annotations_df.groupby('image_id')['image_name'].agg(['count', 'first']).reset_index()
        annot_counts.columns = ['image_id', 'people_gt', 'image_name_gt']
        
        eval_df = pd.merge(pred_counts, annot_counts, on='image_id', how='outer')
        eval_df = eval_df.drop(columns=['image_name'])
        eval_df = eval_df.rename(columns={'image_name_gt': 'image_name'})
        eval_df = eval_df.sort_index()
        eval_df = eval_df.fillna(0)
        
        eval_df['mse'] = np.power(eval_df['people'] - eval_df['people_gt'], 2)
        eval_df['%\error'] = np.abs(100 - np.divide(eval_df['people'], eval_df['people_gt']) * 100)
        
        mean_mse = eval_df['mse'].mean()
        
        return eval_df, mean_mse
    
    def is_within_radius(self, p1, p2, radius):
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2) <= radius
    
    def person_level_validation(self, radius=30, show_images=False):
        true_positives = defaultdict(int)
        false_positives = defaultdict(int)
        false_negatives = defaultdict(int)
        image_names = {}
        
        for image_id in self.annotations_df['image_id'].unique():
            if image_id == 'image_0':
                continue
            
            pred_points = self.predictions_df[self.predictions_df['image_id'] == image_id]
            annot_points = self.annotations_df[self.annotations_df['image_id'] == image_id]
            
            image_names[image_id] = annot_points.iloc[0]['image_name']
            
            matched_annotations = set()
            
            if show_images:
                image = self.load_image(image_names[image_id])
                out = image.copy()
            
            for _, pred in pred_points.iterrows():
                found_match = False
                for _, annot in annot_points.iterrows():
                    if self.is_within_radius(
                    (pred['point_x'], pred['point_y']),
                    (annot['point_x'], annot['point_y']),
                    radius
                    ):
                        if annot.name not in matched_annotations:
                            true_positives[image_id] += 1
                            matched_annotations.add(annot.name)
                            found_match = True
                            
                            if show_images:
                                cv2.circle(out, (annot['point_x'], annot['point_y']), radius, (127, 255, 25), 3)
                                cv2.circle(out, (pred['point_x'], pred['point_y']), 5, (0, 0, 255), -1)
                            break
                
                if not found_match:
                    false_positives[image_id] += 1
                
                false_negatives[image_id] = len(annot_points) - true_positives[image_id]
            
            if show_images:
                plt.figure(figsize=(6, 8))
                plt.imshow(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
                plt.title(f'{image_names[image_id]}')
                plt.show()
        
        results = pd.DataFrame({
            'image_name': pd.Series(image_names),
            'tp': pd.Series(true_positives),
            'fp': pd.Series(false_positives),
            'fn': pd.Series(false_negatives)
        }).fillna(0)
        
        results['precision'] = results['tp'] / (results['tp'] + results['fp'])
        results['recall'] = results['tp'] / (results['tp'] + results['fn'])
        results['f1score'] = 2 * (results['precision'] * results['recall']) / (results['precision'] + results['recall'])
        results['accuracy'] = results['tp'] / (results['tp'] + results['fp'] + results['fn'])
        
        return results.fillna(0)