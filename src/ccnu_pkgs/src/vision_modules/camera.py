#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
from typing import Tuple
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import cv2
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose
import numpy as np
import rospy
from sensor_msgs.msg import Image

from llm_detector import LLMDetector
from utils import pixel_to_world_with_pose
from yolo_detector import YOLODetector


class Camera:
    """无人机相机类（内置检测器）"""
    def __init__(self, detector='yolo', enable_display=True):        
        self.camera_info_received = False
        self.camera_matrix = None
        self.shape = None
        
        self.vtol_x = 0.0
        self.vtol_y = 0.0
        self.vtol_z = 0.0
        self.vtol_yaw = 0.0
        
        self.enable_display = enable_display
        self.window_name = 'Camera'

        self.windows_created = {}
 
        self.current_detections = {}
        self.detection_frames = {}
        self.last_detection_bboxes = {}
        
        self.current_bboxes = {}
        self.current_confidences = {}
        
        # 初始化检测器
        detector_type = detector.lower()
        if detector_type not in ['yolo', 'llm']:
            rospy.logwarn(f"不支持的识别模式: {detector}，默认使用yolo")
            detector_type = 'yolo'
        if detector_type == 'llm':
            self.detector = LLMDetector()
        else:
            self.detector = YOLODetector()  
    
    def image_callback(self, msg: Image, name: str):
        """图像回调函数"""
        if not self.camera_info_received:
            return
            
        try:
            detected_points = {}
            cv_image_bgr = None            
            cv_image = CvBridge().imgmsg_to_cv2(msg, "rgb8")
            cv_image_bgr = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
                
            # 检测主逻辑
            detected_points = self.detector.analyze_image(cv_image_bgr)
            
            # 每次都更新检测结果，无论是否有检测到目标
            self.update_realtime_detections(detected_points)
            
            # 显示图像
            if self.enable_display:
                
                if detected_points:
                    self.draw_detection_results(cv_image_bgr, detected_points)  
                                     
                self.draw_status_info(cv_image_bgr)
                
                self.display_image(cv_image_bgr, name)
            
        except Exception as e:
            rospy.logwarn(f"Image processing error: {e}")
            pass

    def set_drone_state(self, vtol_x: float, vtol_y: float, vtol_z: float, vtol_yaw: float):
        """设置无人机状态"""
        self.vtol_x = vtol_x
        self.vtol_y = vtol_y
        self.vtol_z = vtol_z
        self.vtol_yaw = vtol_yaw
            
    def display_image(self, image: np.ndarray, name: str):
        """显示图像"""
        try:
            if not self.windows_created.get(name, False):
                cv2.namedWindow(name+'_'+self.window_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(name+'_'+self.window_name, 800, 600)
                self.windows_created[name] = True
            
            cv2.imshow(name+'_'+self.window_name, image)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                rospy.signal_shutdown("用户退出")
            elif key == ord('c'):
                self.clear_detections()
                rospy.loginfo("All detection results cleared")
            elif key == ord('d'):
                rospy.loginfo(f"Current detections: {self.current_detections}")
                rospy.loginfo(f"Detection frames: {self.detection_frames}")
                rospy.loginfo(f"Current bboxes: {self.current_bboxes}")
                rospy.loginfo(f"Current confidences: {self.current_confidences}")
                rospy.loginfo(f"Targets found: {self.detector.targets_found}")
                
        except Exception as e:
            rospy.logerr(f"Error displaying image: {e}")
    
    def draw_detection_results(self, image: np.ndarray, detected_points: dict):
        """在图像上绘制检测结果"""
        try:
            h, w = image.shape[:2]
            center_x, center_y = w // 2, h // 2

            cv2.line(image, (center_x - 20, center_y), (center_x + 20, center_y), (0, 255, 0), 2)
            cv2.line(image, (center_x, center_y - 20), (center_x, center_y + 20), (0, 255, 0), 2)
            cv2.circle(image, (center_x, center_y), 30, (0, 255, 0), 2)
            
            for class_name, center in detected_points.items():
        
                bbox = self.current_bboxes.get(class_name, None)
        
                confidence = self.current_confidences.get(class_name, 0.5)
                
                frame_count = self.detection_frames.get(class_name, 0)
                
                class_colors = {
                    'critical': (0, 0, 255),
                    'injured': (0, 255, 255),
                    'healthy': (125, 125, 125)
                }
                color = class_colors.get(class_name, (255, 255, 255))
                
                if frame_count < 5:
                    color = tuple(min(255, c + 50) for c in color)
                    thickness = 3
                else:
                    thickness = 2
                
                if bbox:
                    x1, y1, x2, y2 = bbox
                    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
                
                cv2.circle(image, center, 8, color, -1)
                cv2.circle(image, center, 12, color, thickness)
                
                status = "NEW" if frame_count < 5 else "TRACKING"
                label = f"{class_name}: {confidence:.3f} ({status})"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                
                label_x = center[0] - label_size[0] // 2
                label_y = center[1] - 30
                
                cv2.rectangle(image, 
                             (label_x - 5, label_y - label_size[1] - 5),
                             (label_x + label_size[0] + 5, label_y + 5),
                             color, -1)
                
                cv2.putText(image, label, (label_x, label_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                image_center = (image.shape[1] // 2, image.shape[0] // 2)
                cv2.line(image, center, image_center, color, thickness)
                
                cv2.line(image, (image_center[0] - 20, image_center[1]), 
                        (image_center[0] + 20, image_center[1]), (255, 255, 255), 2)
                cv2.line(image, (image_center[0], image_center[1] - 20), 
                        (image_center[0], image_center[1] + 20), (255, 255, 255), 2)
            
        except Exception as e:
            rospy.logerr(f"Error drawing detection results: {e}")
    
    def draw_status_info(self, image: np.ndarray):
        """在图像上绘制状态信息"""
        try:
            h, w = image.shape[:2]
            
            detection_enabled = self.detector.detector_enabled
            detection_count = len(self.current_detections)
            cv2.rectangle(image, (self.detector.roi_x_min, self.detector.roi_y_min), (self.detector.roi_x_max, self.detector.roi_y_max), (0, 255, 0), 2)
            
            status_texts = [
                f"Detection: {'ON' if detection_enabled else 'OFF'}",
                f"Targets: {detection_count}",
                f"Resolution: {w}x{h}"
            ]
            
            for i, text in enumerate(status_texts):
                y_pos = 20 + i * 15
                cv2.putText(image, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
        except Exception as e:
            rospy.logerr(f"Error drawing status info: {e}")
    
    def update_realtime_detections(self, detected_points: dict):
        """实时更新检测结果，直接使用YOLO的检测结果"""
        try:
            self.current_detections.clear()
            self.current_bboxes.clear()
            self.current_confidences.clear()
            
            if not detected_points:
                return
            
            if hasattr(self.detector, 'last_detection_results') and self.detector.last_detection_results:
                for detection in self.detector.last_detection_results:
                    class_name = detection['class_name']
                    if class_name in detected_points:
                        center = detected_points[class_name]
                        self.current_detections[class_name] = center
                        
                        if 'bbox' in detection:
                            self.current_bboxes[class_name] = detection['bbox']
                        else:
                            x, y = center
                            bbox_size = 50
                            bbox = (x - bbox_size//2, y - bbox_size//2, 
                                   x + bbox_size//2, y + bbox_size//2)
                            self.current_bboxes[class_name] = bbox
                        
                        if 'confidence' in detection:
                            self.current_confidences[class_name] = detection['confidence']
                        else:
                            self.current_confidences[class_name] = 0.5
                        
                        self.detection_frames[class_name] = 0
            else:
                for target_type, (px, py) in detected_points.items():
                    self.current_detections[target_type] = (px, py)
                    
                    bbox_size = 50
                    bbox = (px - bbox_size//2, py - bbox_size//2, 
                           px + bbox_size//2, py + bbox_size//2)
                    self.current_bboxes[target_type] = bbox
                    
                    self.current_confidences[target_type] = 0.5
                    
                    self.detection_frames[target_type] = 0
                
        except Exception as e:
            rospy.logerr(f"Error updating realtime detections: {e}")
    
    def clear_detections(self):
        """清除所有检测结果"""
        self.current_detections.clear()
        self.detection_frames.clear()
        self.current_bboxes.clear()
        self.current_confidences.clear()
        rospy.loginfo("All detection results cleared")

    def set_display_enabled(self, enabled):
        """设置是否启用显示功能"""
        self.enable_display = bool(enabled)
        rospy.loginfo(f"Display function {'enabled' if self.enable_display else 'disabled'}")
        
    def enable_detection(self):
        self.detector.detector_enabled = True
        rospy.loginfo("图像识别功能已启用")

    def disable_detection(self):
        self.detector.detector_enabled = False
        rospy.loginfo("图像识别功能已禁用")
        
    def destroy_window(self, window_name=None):
        """销毁OpenCV窗口"""
        try:
            if window_name is None:
                # 销毁所有窗口
                for name in self.windows_created:
                    if self.windows_created[name]:
                        cv2.destroyWindow(name+'_'+self.window_name)
                        self.windows_created[name] = False
                        rospy.loginfo(f"已销毁相机窗口: {name}")
                self.set_display_enabled(False)
            else:
                # 销毁指定窗口
                if self.windows_created.get(window_name, False):
                    cv2.destroyWindow(window_name+'_'+self.window_name)
                    self.windows_created[window_name] = False
                    rospy.loginfo(f"已销毁相机窗口: {window_name}")
        except Exception as e:
            rospy.logwarn(f"销毁窗口时出错: {e}")
    
    def convert_yolo_detection_to_world_coordinates(self, detected_points: dict, camera_pose: Pose, camera_height: float) -> dict:
        """
        将YOLO检测到的像素坐标转换为世界坐标
        """
        if not self.camera_info_received or self.camera_matrix is None:
            rospy.logwarn("相机内参未初始化，无法进行坐标转换")
            return {}
        
        world_coordinates = {}
        
        for class_name, (pixel_u, pixel_v) in detected_points.items():
            try:
                world_x, world_y = pixel_to_world_with_pose(
                    pixel_u, pixel_v, 
                    self.camera_matrix, 
                    camera_pose, 
                    camera_height
                )
                if world_x != float('inf') and world_y != float('inf'):
                    world_coordinates[class_name] = (world_x, world_y, 0.0)
                    rospy.loginfo_once(f"{class_name} 坐标转换: 像素({pixel_u}, {pixel_v}) -> 世界({world_x:.2f}, {world_y:.2f})")
                else:
                    rospy.logwarn(f"{class_name} 坐标转换失败: 像素({pixel_u}, {pixel_v})")
                    
            except Exception as e:
                rospy.logwarn(f"{class_name} 坐标转换错误: {e}")
        
        return world_coordinates

    def get_pixel_to_meter_ratio(self, camera_height: float, pixel_u: int = None, pixel_v: int = None) -> Tuple[float, float]:
        """
        计算指定像素位置处的像素到实际距离的比例
        """
        if not self.camera_info_received or self.camera_matrix is None:
            rospy.logwarn("相机内参未初始化，无法计算像素比例")
            return 0.0, 0.0
        
        # 如果没有指定像素位置，使用图像中心
        if pixel_u is None or pixel_v is None:
            pixel_u = self.shape[1] // 2
            pixel_v = self.shape[0] // 2
        
        # 获取相机焦距内参
        fx = self.camera_matrix[0, 0]  
        fy = self.camera_matrix[1, 1]
        
        # 计算像素到实际距离的比例
        ratio_x = camera_height / fx  
        ratio_y = camera_height / fy 
        
        return ratio_x, ratio_y

    def print_camera_info(self, camera_height: float = 10.0):
        """打印相机信息，包括像素比例等"""
        if not self.camera_info_received:
            rospy.logwarn("相机内参未初始化")
            return
        
        rospy.loginfo("=== 相机信息 ===")
        rospy.loginfo(f"图像尺寸: {self.shape[1]} x {self.shape[0]} 像素")
        rospy.loginfo(f"相机内参矩阵:\n{self.camera_matrix}")
        
        # 计算不同高度的像素比例
        heights = [5.0, 10.0, 20.0, 50.0]
        for h in heights:
            ratio_x, ratio_y = self.get_pixel_to_meter_ratio(h)
            rospy.loginfo(f"高度 {h}m: 1像素 = {ratio_x*1000:.1f}mm (x), {ratio_y*1000:.1f}mm (y)")
        
        # 计算图像中心处的像素比例
        center_ratio_x, center_ratio_y = self.get_pixel_to_meter_ratio(camera_height)
        rospy.loginfo(f"当前高度 {camera_height}m 下，图像中心处:")
        rospy.loginfo(f"  1像素 = {center_ratio_x*1000:.1f}mm (x方向)")
        rospy.loginfo(f"  1像素 = {center_ratio_y*1000:.1f}mm (y方向)")
        
        # 计算图像覆盖的实际区域
        image_width_m = self.shape[1] * center_ratio_x
        image_height_m = self.shape[0] * center_ratio_y
        rospy.loginfo(f"图像覆盖区域: {image_width_m:.1f}m x {image_height_m:.1f}m")
        rospy.loginfo("==================")