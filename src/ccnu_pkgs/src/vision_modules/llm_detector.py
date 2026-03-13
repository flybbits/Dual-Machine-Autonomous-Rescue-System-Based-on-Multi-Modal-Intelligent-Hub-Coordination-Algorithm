#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import base64
import re

import cv2
import rospy

from detector import Detector

class LLMDetector(Detector):
    """LLM检测器类"""
    def __init__(self):
        super().__init__()
        self.last_llm_request_time = 0.0
        self.llm_cooldown_duration = 15.0
        # 延迟导入image_llm函数，避免循环导入
        self.image_llm = None
                
    def analyze_image(self, image):
        """使用LLM分析图像中的人员位置"""
        if not self.detector_enabled:
            return {}
        
        # 延迟导入image_llm函数
        if self.image_llm is None:
            try:
                from models.llm import image_llm
                self.image_llm = image_llm
            except ImportError:
                rospy.logwarn("LLM功能不可用，请检查models.llm模块")
                return {}
            
        if self.image_llm is None:
            rospy.logwarn("LLM功能不可用，请检查models.llm模块")
            return {}
            
        try:
            current_time = rospy.Time.now().to_sec()
            if current_time - self.last_llm_request_time < self.llm_cooldown_duration:
                return {}
            
            self.last_llm_request_time = current_time
            
            success, encoded_image = cv2.imencode('.jpg', image)
            if not success:
                rospy.logwarn("图像编码失败")
                return {}
            
            base64_str = base64.b64encode(encoded_image.tobytes()).decode('utf-8')
            
            rospy.loginfo("开始LLM图像分析...")
            llm_response = self.image_llm(base64_str)
            rospy.loginfo(f"LLM响应: {llm_response}")
            
            detected_points = self.parse_llm_response(llm_response)
            
            rospy.loginfo(f"LLM分析完成，下次请求将在{self.llm_cooldown_duration}秒后可用")
            
            return detected_points
            
        except Exception as e:
            rospy.logwarn(f"LLM图像分析错误: {e}")
            return {}
    
    def parse_llm_response(self, response):
        """解析LLM返回的坐标信息"""
        detected_points = {}
        
        try:
            patterns = {
                'critical': r'red\((\d+),(\d+)\)',
                'injured': r'yellow\((\d+),(\d+)\)', 
                'healthy': r'white\((\d+),(\d+)\)'
            }
            
            for target_type, pattern in patterns.items():
                match = re.search(pattern, response, re.IGNORECASE)
                if match:
                    x, y = int(match.group(1)), int(match.group(2))
                    detected_points[target_type] = (x, y)
                    rospy.loginfo(f"LLM检测到{target_type}目标: 像素坐标({x}, {y})")
                    
        except Exception as e:
            rospy.logwarn(f"解析LLM响应错误: {e}")
        return detected_points
        
    def get_cooldown_remaining(self):
        """获取LLM冷却剩余时间"""
        if not self.detector_enabled:
            return 0.0    
        current_time = rospy.Time.now().to_sec()
        elapsed = current_time - self.last_llm_request_time
        remaining = max(0.0, self.llm_cooldown_duration - elapsed)
        return remaining
        
    def set_cooldown_duration(self, duration):
        """设置LLM冷却时间"""
        self.llm_cooldown_duration = max(1.0, float(duration))
        rospy.loginfo(f"LLM冷却时间已设置为: {self.llm_cooldown_duration}秒") 