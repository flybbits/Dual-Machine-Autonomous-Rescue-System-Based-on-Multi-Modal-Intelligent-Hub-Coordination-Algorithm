#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
路径规划模块

该模块提供多种路径规划算法，包括：
1. 切线算法：用于避开圆形禁飞区
2. 大模型路径规划：基于LLM的智能路径规划
3. GPS引导点规划：基础的任务路径规划
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from geometry_msgs.msg import Pose
import rospy

from models import text_llm
from utils import calculate_distance, get_intercept_point


class PathPlanner:
    """路径规划模块类，包含切线算法、禁飞区避让和大模型路径规划"""
    def __init__(self):
        # GPS引导点坐标
        self.gps_guide_x = 0.0
        self.gps_guide_y = 0.0
        rospy.Subscriber('/zhihang/first_point', Pose, self.gps_guide_point_callback)
        
        # 禁飞区参数
        self.no_fly_zone_radius = 200.0  # 禁飞区半径
        self.safety_margin = 50.0        # 安全边距
        self.downtown_centers = []       # 居民区中心点列表
        rospy.Subscriber('/zhihang/downtown', Pose, self.downtown_callback)
    
    def gps_guide_point_callback(self, msg: Pose):
        """
        GPS引导点回调函数
        
        接收GPS引导点的位置信息
        
        Args:
            msg: 包含GPS引导点坐标的Pose消息
        """
        self.gps_guide_x = msg.position.x
        self.gps_guide_y = msg.position.y
    
    def plan_gps_mission(self, uav_x: float, uav_y: float):
        """
        规划GPS任务路径
        
        使用切线算法规划从无人机当前位置到GPS引导点的路径，
        避开禁飞区
        
        Args:
            uav_x: 无人机当前X坐标
            uav_y: 无人机当前Y坐标
        """
        
        initial_pos = [uav_x, uav_y]
        gps_pos = [self.gps_guide_x, self.gps_guide_y]
        
        rospy.loginfo("路径规划开始")
        
        # 使用切线算法规划路径
        self.waypoints = self.plan_path_with_tangent(initial_pos, gps_pos)
        
        if self.waypoints is None:
            rospy.logerr("路径规划失败")
            return
            
        rospy.loginfo(f"路径规划完成，共 {len(self.waypoints)} 个航点")
    
    def downtown_callback(self, msg: Pose):
        """
        居民区信息回调函数
        
        接收居民区（禁飞区）的位置信息
        
        Args:
            msg: 包含居民区坐标的Pose消息
        """
        downtown1 = [msg.position.x, msg.position.y]
        downtown2 = [msg.orientation.x, msg.orientation.y]
        self.downtown_centers = [downtown1, downtown2]
        
    def find_nearest_no_fly_zone(self, position):
        """
        找到最近的禁飞区
        
        计算给定位置到所有禁飞区的距离，返回最近的禁飞区中心
        
        Args:
            position: 待检查的位置坐标 [x, y]
            
        Returns:
            tuple: (最近的禁飞区中心, 最小距离)
        """
        if not self.downtown_centers:
            return None, float('inf')
            
        min_distance = float('inf')
        nearest_center = None
        
        for center in self.downtown_centers:
            distance = calculate_distance(position, center)
            if distance < min_distance:
                min_distance = distance
                nearest_center = center
                
        return nearest_center, min_distance
        
    def plan_path_with_tangent(self, start_pos, end_pos):
        """
        使用切线算法规划路径
        
        通过计算禁飞区圆的切线点，规划避开禁飞区的路径
        
        Args:
            start_pos: 起始位置 [x, y]
            end_pos: 目标位置 [x, y]
            
        Returns:
            list: 航点列表，包含切点和目标点
        """
        if not self.downtown_centers:
            rospy.logwarn("居民区坐标未收到，使用直线路径")
            return [end_pos]
            
        # 找到最近的禁飞区
        nearest_center, distance_to_nearest = self.find_nearest_no_fly_zone(start_pos)
    
        # 计算禁飞区半径（包含安全边距）
        r = self.no_fly_zone_radius + self.safety_margin
        x0, y0 = nearest_center  # 禁飞区中心
        x1, y1 = start_pos       # 起始点
        x2, y2 = end_pos         # 目标点
        
        try:
            # 计算切点
            thun_safe_x, thun_safe_y = get_intercept_point(r, x0, y0, x1, y1, x2, y2)
            tangent_point = [thun_safe_x, thun_safe_y]
            
            # 检查切点距离是否合理
            distance_to_tangent = calculate_distance(start_pos, tangent_point)
            
            if distance_to_tangent > 2000:
                rospy.logwarn("距离太远，使用直线路径")
                return [end_pos]
                
            return [tangent_point, end_pos]
            
        except Exception as e:
            rospy.logwarn(f"计算失败: {e}，使用直线路径")
            return [end_pos] 
    
    def plan_iris_mission(self, start_pos, healthy_point, critical_point):
        """
        使用大模型路径规划IRIS任务
        
        基于LLM的智能路径规划，考虑健康人员和重伤人员的位置
        
        Args:
            start_pos: 起始位置
            healthy_point: 健康人员位置
            critical_point: 重伤人员位置
        """
        initial_pos = start_pos
        gps_pos = [self.gps_guide_x, self.gps_guide_y]
        
        try:    
            self.waypoints = self.plan_path_with_tangent(initial_pos, gps_pos)            
        except Exception as e:
            self.waypoints = self.plan_path_with_llm(initial_pos, gps_pos, healthy_point, critical_point)
            self.waypoints.append(gps_pos)    
            
        if self.waypoints is None:
            rospy.logerr("路径规划失败")
            return

        rospy.loginfo(f"路径规划完成，共 {len(self.waypoints)} 个航点")
    
    def plan_path_with_llm(self, start_pos, end_pos, healthy_point, critical_point):
        """
        使用大模型进行路径规划
        
        调用LLM接口，根据任务需求智能规划路径
        
        Args:
            start_pos: 起始位置
            end_pos: 目标位置
            healthy_point: 健康人员位置
            critical_point: 重伤人员位置
            
        Returns:
            list: 大模型规划的航点列表
        """
        return text_llm(
            takeoff_point=start_pos,
            downtown_point=self.downtown_centers, 
            gps_point=end_pos,
            healthy_point=healthy_point,
            critical_point=critical_point
        )