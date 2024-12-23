import cv2
import numpy as np
import matplotlib.pyplot as plt
from functools import wraps
import pandas as pd

from src.PerformanceMonitor import PerformanceMonitor

class ContourDetector:
    def __init__(self, manual_annotations):
        self.performance = PerformanceMonitor()
        self.annotations_df = manual_annotations
        self.predictions_df = None

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

    def clahe(self, image: np.ndarray, show_image=False, clip_limit=2.0, grid_size=(8, 8)) -> np.ndarray:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
        clahe_image = clahe.apply(image)
        if show_image:
            plt.imshow(clahe_image, cmap='gray')
            plt.title('CLAHE Image')
            plt.show()
        return clahe_image

    def convert_to_lab(self, image: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    def apply_clahe_to_lab(self, lab_image: np.ndarray, show_image=False) -> np.ndarray:
        l, a, b = cv2.split(lab_image)
        l_clahe = self.clahe(l, show_image)
        lab_clahe = cv2.merge((l_clahe, a, b))
        return cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

    def apply_truncated_threshold(self, image: np.ndarray, show_image=False) -> np.ndarray:
        blurred_image = cv2.GaussianBlur(image, (3, 3), 0)
        _, r_channel = cv2.threshold(blurred_image[:, :, 2], 127, 255, cv2.THRESH_TRUNC)
        _, g_channel = cv2.threshold(blurred_image[:, :, 1], 127, 255, cv2.THRESH_TRUNC)
        _, b_channel = cv2.threshold(blurred_image[:, :, 0], 127, 255, cv2.THRESH_TRUNC)
        truncated_image = cv2.merge((b_channel, g_channel, r_channel))
        if show_image:
            plt.imshow(cv2.cvtColor(truncated_image, cv2.COLOR_BGR2RGB))
            plt.title('Preprocessed Image')
            plt.show()
        return truncated_image

    def preprocess_image(self, image_name, show_image=False):
        image = self.load_image(image_name, show_image)
        if image is None:
            return None
        lab_image = self.convert_to_lab(image)
        clahe_image = self.apply_clahe_to_lab(lab_image, show_image)
        truncated_image = self.apply_truncated_threshold(clahe_image, show_image)
        return truncated_image

    def segmentation_foreground(self, truncated_image, truncated_bckg_image, show_image=False):
        segmented_image = cv2.absdiff(truncated_image, truncated_bckg_image)
        kernel = np.ones((5, 5), np.uint8)
        opening = cv2.morphologyEx(segmented_image, cv2.MORPH_OPEN, kernel)
        if show_image:
            plt.imshow(opening)
            plt.title('Segmented Foreground')
            plt.show()
        return opening

    def color_edge_detection(self, image: np.ndarray, show_image=False, min_threshold=100, max_threshold=200):
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        color_ranges = {
            'black': ((0, 0, 0), (180, 255, 50)), 
            'sand': ((20, 40, 150), (30, 100, 200))
        }
        
        edges = np.zeros_like(image[:, :, 0])
        for color, (lower, upper) in color_ranges.items():
            mask = cv2.inRange(hsv_image, lower, upper)
            color_edges = cv2.Canny(mask, min_threshold, max_threshold)
            edges = cv2.bitwise_or(edges, color_edges)
            
        intensified_edges = cv2.Laplacian(edges, cv2.CV_64F)
        laplacian = np.uint8(np.absolute(intensified_edges))
        if show_image:
            plt.imshow(laplacian, cmap='gray')
            plt.title('Color Edge Detection')
            plt.show()
        return laplacian

    def obtain_contours(self, image: np.ndarray, show_image=False):
        contours, _ = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        min_perimeter = 50
        filtered_contours = [cnt for cnt in contours if cv2.arcLength(cnt, True) > min_perimeter]
        
        if show_image:
            contour_image = np.zeros_like(image)
            cv2.drawContours(contour_image, filtered_contours, -1, (255, 255, 255), 1)
            plt.imshow(contour_image, cmap='gray')
            plt.title('Contours')
            plt.show()
        return filtered_contours

    def obtain_center_points(self, filtered_contours):
        y_min = 420
        contour_points = []
        for contorno in filtered_contours:
            M = cv2.moments(contorno)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                if cy > y_min:
                    contour_points.append((cx, cy))
        return contour_points

    @property
    def metrics(self):
        return self.performance.get_metrics()

    def calculate_tp_fp_fn(self, manual_points, detected_points, y_thresh=30, x_thresh=12):
        manual_points = np.array(manual_points)
        detected_points = np.array(detected_points)
        tp = 0
        fp = 0
        fn = 0
        detected_used = np.zeros(len(detected_points), dtype=bool)
        
        for manual_point in manual_points:
            detected = False
            closest_index = None
            closest_distance = float('inf')
            
            for i, detected_point in enumerate(detected_points):
                if not detected_used[i]:
                    distance_y = abs(manual_point[1] - detected_point[1])
                    distance_x = abs(manual_point[0] - detected_point[0])
                    if distance_y <= y_thresh and distance_x <= x_thresh:
                        distance = np.sqrt(distance_y ** 2 + distance_x ** 2)
                        if distance < closest_distance:
                            closest_distance = distance
                            closest_index = i
                            detected = True
            
            if detected and closest_index is not None:
                tp += 1
                detected_used[closest_index] = True
            else:
                fn += 1

        fp = np.sum(~detected_used)
        return tp, fp, fn

    def calculate_F1score_recall_precision(self, manual_points, detected_points):
        tp, fp, fn = self.calculate_tp_fp_fn(manual_points, detected_points)
        precision = tp/(tp+fp) if tp+fp > 0 else 0
        recall = tp/(tp+fn) if tp+fn > 0 else 0
        f1score = 2*((precision*recall)/(precision+recall)) if recall > 0 and precision > 0 else 0
        return tp, fp, fn, precision, recall, f1score

    def process_image(self, image_name, binary_image, bckg_image, show_image=False):
        return self.performance.measure_performance(self._process_image, image_name)(binary_image, bckg_image, show_image)
    
    def _process_image(self, binary_image, bckg_image, show_image=False):
        segmented_image = self.segmentation_foreground(binary_image, bckg_image, show_image)
        edges = self.color_edge_detection(segmented_image, show_image)
        contours = self.obtain_contours(edges, show_image)
        center_points = self.obtain_center_points(contours)
        
        return center_points

    def process_images(self, show_image=True):
        processed_images = []
        for idx, name in enumerate(self.annotations_df['image_name'].unique()[1:], start=1):
            truncated_image = self.preprocess_image(name, show_image)
            truncated_background = self.preprocess_image(self.annotations_df.iloc[0]['image_name'], show_image)
            image_center_points = self.process_image(name, truncated_image, truncated_background, show_image)
            
            if image_center_points and len(image_center_points) > 0:
                for point in image_center_points:
                    processed_images.append({
                        'point_x': int(point[0]), 
                        'point_y': int(point[1]), 
                        'image_id': f'image_{idx}',
                        'image_name': name
                    })
                
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
        
    def person_level_validation(self, show_images=False):
        
        results = []
        for image_name in self.predictions_df['image_name'].unique()[1:]:
            detected_points = self.predictions_df[self.predictions_df['image_name'] == image_name][['point_x', 'point_y']].values.tolist()
            manual_points = self.annotations_df[self.annotations_df['image_name'] == image_name][['point_x', 'point_y']].values.tolist()
            
            if not detected_points or not manual_points:
                continue
            
            tp, fp, fn, precision, recall, f1score = self.calculate_F1score_recall_precision(manual_points, detected_points)
            
            accuracy = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
            
            results.append({
                'image_name': image_name,
                'tp': tp,
                'fp': fp,
                'fn': fn,
                'precision': precision,
                'recall': recall,
                'f1score': f1score,
                'accuracy': accuracy
            })
            
            original_image = self.load_image(image_name, show_image=False)
            if show_images:
                for point in detected_points:
                    cv2.circle(original_image, (point[0], point[1]), 5, (0, 0, 255), -1)
                plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
                plt.title(f'Detected Points on {image_name}')
                plt.show()

        results_df = pd.DataFrame(results)
        return results_df
        
    def person_level_validation2(self, radius=30, show_images=False):
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
        