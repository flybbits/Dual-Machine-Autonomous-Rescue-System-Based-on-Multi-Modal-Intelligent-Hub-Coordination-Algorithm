#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import cv2
import rospy
from ultralytics import YOLO

from detector import Detector


class YOLODetector(Detector):
    """YOLO检测器类"""
    def __init__(self, model_path=os.path.join(os.path.dirname(__file__), 'best.pt')):
        super().__init__()
        
        self.yolo_detector = None
        self.model_path = model_path
        
        self.confidence_threshold = 0.83
        self.iou_threshold = 0.98
        
        self.detected_points = {}
        self.last_detection_results = []
        
        self.init_yolo_detector()    
        
    def init_yolo_detector(self):
        """加载YOLO模型"""
        try:
            if not os.path.exists(self.model_path):
                rospy.logwarn(f"YOLO模型文件不存在：{self.model_path}")
                exit(1)
            else:
                self.yolo_detector = YOLO(self.model_path)
        except Exception as e:
            rospy.logwarn(f"YOLO检测器初始化失败：{e}")
            exit(1)
    
    def analyze_image(self, image):
        """使用YOLO模型分析图像中的人员位置"""
        if not self.detector_enabled:
            return {}
            
        try:
            results = self.yolo_detector(
                image,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                verbose=False
            )
            
            self.detected_points = {}
            self.last_detection_results = []
            
            for result in results:
                if result.boxes is not None:
                    boxes = result.boxes
                    
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        
                        if not (x1 >= self.roi_x_min and x2 <= self.roi_x_max and y1 >= self.roi_y_min and y2 <= self.roi_y_max):
                            continue
                        
                        confidence = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        if class_id < len(self.class_names):
                            class_name = self.class_names[class_id]
                                
                            center_x = int((x1 + x2) / 2)
                            center_y = int((y1 + y2) / 2)
                            
                            self.detected_points[class_name] = (center_x, center_y)
                          
                            detection_data = {
                                'class_name': class_name,
                                'center': (center_x, center_y),
                                'bbox': (int(x1), int(y1), int(x2), int(y2)),
                                'confidence': float(confidence)
                            }
                            self.last_detection_results.append(detection_data)
                            
                            if not self.targets_found[class_name]:
                                self.targets_found[class_name] = True
                                rospy.loginfo(f"First detection of {class_name} target")
            
            return self.detected_points
            
        except Exception as e:
            rospy.logwarn(f"YOLO image analysis error: {e}")
            self.detected_points = {}
            self.last_detection_results = []
            return {}
            
    def draw_detection_box(self, image, x1, y1, x2, y2, class_name, confidence):
        """绘制检测框"""
        try:
            class_colors = {'critical': (0, 0, 255), 'injured': (0, 255, 255), 'healthy': (255, 255, 255)}
            color = class_colors.get(class_name, (255, 255, 255))
            
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(image, (int(x1), int(y1) - label_size[1] - 10), 
                         (int(x1) + label_size[0], int(y1)), color, -1)
            cv2.putText(image, label, (int(x1), int(y1) - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            cv2.circle(image, (center_x, center_y), 5, color, -1)
            
        except Exception as e:
            rospy.logwarn(f"Error drawing detection box: {e}")
            
    def set_confidence_threshold(self, threshold):
        """设置检测置信度阈值"""
        self.confidence_threshold = max(0.1, min(1.0, threshold))
        rospy.loginfo(f"Confidence threshold set to: {self.confidence_threshold}")
        
    def set_iou_threshold(self, threshold):
        """设置检测IOU阈值"""
        self.iou_threshold = max(0.1, min(1.0, threshold))
        rospy.loginfo(f"IOU threshold set to: {self.iou_threshold}")
        
    def set_model_path(self, model_path):
        """设置检测模型文件"""
        self.model_path = str(model_path)
        self.init_yolo_detector(self.model_path)
        rospy.loginfo(f"YOLO model path set to: {self.model_path}")
        