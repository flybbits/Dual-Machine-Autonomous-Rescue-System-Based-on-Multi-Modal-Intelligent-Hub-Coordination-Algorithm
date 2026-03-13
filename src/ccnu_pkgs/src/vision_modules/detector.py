#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from collections import deque

class Detector:
    """无人机检测器基类"""
    def __init__(self):
        self.detector_enabled = False
        self.targets_found = {'critical': False, 'injured': False, 'healthy': False}
        self.class_names = ['critical', 'injured', 'healthy']
        self.track_history = {name: deque(maxlen=30) for name in self.class_names}
        
        self.roi_x_min = 200
        self.roi_x_max = 1080
        self.roi_y_min = 160
        self.roi_y_max = 520
        
    def analyze_image(self, image):
        raise NotImplementedError
    
    def set_roi(self, x_min, x_max, y_min, y_max):
        """设置检测RoI参数避免相机边缘畸变"""
        self.roi_x_min = x_min
        self.roi_x_max = x_max
        self.roi_y_min = y_min
        self.roi_y_max = y_max
    
    def reset_targets(self):
        """重置目标检测状态"""
        self.targets_found = {'critical': False, 'injured': False, 'healthy': False}
        
        for name in self.class_names:
            self.track_history[name].clear()
            
    def get_targets_found(self):
        """获取目标检测状态"""
        return self.targets_found.copy()
    
    def get_targets_history(self):
        """获取目标检测历史"""
        return self.track_history.copy()
