#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
无人机基类模块

该模块定义了无人机的基础类UAV，提供了：
1. 基础状态管理（位置、姿态、速度等）
2. 飞行控制命令（起飞、降落、悬停等）
3. 相机信息处理
4. 航点到达检测
"""

import os
import sys
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from geometry_msgs.msg import Pose, PoseStamped, Twist
import rospy
from mavros_msgs.msg import State
from sensor_msgs.msg import CameraInfo
from mavros_msgs.msg import Altitude
from std_msgs.msg import String

from utils import calculate_distance, EulerAndQuaternionTransform
from vision_modules.camera import Camera

# 控制参数常量
MAX_LINEAR = 1000        # 最大线速度
MAX_ANG_VEL = 0.5        # 最大角速度
LINEAR_STEP_SIZE = 0.1   # 线速度步长
ANG_VEL_STEP_SIZE = 0.01 # 角速度步长

class UAV:
    """
    无人机基类
    
    该类提供无人机的基础功能，包括状态管理、飞行控制、
    相机信息处理等，是其他具体无人机类型的父类
    """
    
    def __init__(self, uav_name='uav'):
        self.mode = String()
        self.pose = Pose()
        self.twist = Twist()
        self.target = PoseStamped()
        
        # 位置和姿态状态
        self.x, self.y, self.z, self.w = 0.0, 0.0, 0.0, 0.0
        self.forward, self.leftward, self.upward, self.yaw = 0.0, 0.0, 0.0, 0.0
        self.state = State()
        
        # 飞行参数
        self.takeoff_altitude = 20.0    # 起飞高度
        self.waypoint_tolerance = 5.0   # 航点到达容差
         
        # 为每个UAV实例创建独立的Camera实例
        self.camera = Camera(detector='yolo', enable_display=True)
        self.uav_name = uav_name
        
        # 发布器（子类需要实现）
        self.cmd_mode_pub: rospy.Publisher = None
        self.cmd_pose_enu_pub: rospy.Publisher = None
    
    def camera_info_callback(self, msg: CameraInfo):
        """
        相机内参回调函数
        
        接收相机内参信息，用于坐标转换
        
        Args:
            msg: 相机信息消息
        """
        if not self.camera.camera_info_received:
            self.camera.shape = (msg.height, msg.width)
            self.camera.camera_matrix = np.array(msg.K).reshape(3, 3)
            self.camera.camera_info_received = True
    
    def alt_callback(self, msg: Altitude):
        """
        高度信息回调函数
        
        Args:
            msg: 高度消息
        """
        self.alt = msg
    
    def state_callback(self, msg: State):
        """
        飞行状态回调函数
        
        Args:
            msg: 状态消息
        """
        self.state = msg
    
    def pose_callback(self, msg: PoseStamped):
        """
        位姿信息回调函数
        
        更新无人机的位置和姿态信息
        
        Args:
            msg: 位姿消息
        """
        self.pose = msg.pose
        self.x, self.y, self.z = msg.pose.position.x, msg.pose.position.y, msg.pose.position.z
        w = msg.pose.orientation.w
        x = msg.pose.orientation.x
        y = msg.pose.orientation.y
        z = msg.pose.orientation.z
        # 将四元数转换为欧拉角，获取偏航角
        self.yaw = EulerAndQuaternionTransform((x, y, z, w))[2]
        
    def send_rotation_command(self, yaw: float):
        """
        发送旋转命令
        
        Args:
            yaw: 目标偏航角
        """
        pose = Pose()
        orien = EulerAndQuaternionTransform((0, 0, yaw))
        pose.orientation.x = orien[0]
        pose.orientation.y = orien[1]
        pose.orientation.z = orien[2]
        pose.orientation.w = orien[3]
        self.cmd_pose_enu_pub.publish(pose)
    
    def pub_move(self):
        """
        发布运动命令
        
        抽象方法，子类需要实现具体的运动命令发布逻辑
        """
        raise NotImplementedError
    
    def send_target_command(self, x: float, y: float, z: float, yaw: float):
        """
        发送目标位置命令
        
        抽象方法，子类需要实现具体的目标位置命令发送逻辑
        
        Args:
            x: 目标X坐标
            y: 目标Y坐标
            z: 目标Z坐标
            yaw: 目标偏航角
        """
        raise NotImplementedError

    def is_waypoint_reached(self, target):
        """
        检查是否到达航点
        
        Args:
            target: 目标航点坐标 (x, y)
            
        Returns:
            bool: 是否到达航点
        """
        current_pos = (self.x, self.y)
        return calculate_distance(current_pos, target) <= self.waypoint_tolerance

    def pub_cmd(self):
        """
        发布模式命令
        """
        self.cmd_mode_pub.publish(self.mode)

    def arm(self):
        self.mode.data = 'ARM'
        self.pub_cmd()
    
    def disarm(self):
        self.mode.data = 'DISARM'
        self.pub_cmd()

    def takeoff(self):
        self.mode.data = 'AUTO.TAKEOFF'
        self.pub_cmd()
        
    def land(self):
        self.mode.data = 'AUTO.LAND'
        self.pub_cmd()  

    def offboard(self):
        self.mode.data = 'OFFBOARD'
        self.pub_cmd()   
    
    def guided(self):
        self.mode.data = 'AUTO.GUIDED'
        self.pub_cmd()
    
    def rtl(self):
        self.mode.data = 'AUTO.RTL'
        self.pub_cmd()
        
    def hover(self):
        self.forward = 0.0
        self.leftward = 0.0
        self.upward = 0.0
        self.angular = 0.0
        self.pub_move()
        self.mode.data = 'HOVER'
        self.pub_cmd()
                 
    def move_forward(self):
        self.forward += LINEAR_STEP_SIZE
        self.pub_move()

    def move_backward(self):
        self.forward -= LINEAR_STEP_SIZE
        self.pub_move()

    def move_leftward(self):
        self.leftward += LINEAR_STEP_SIZE
        self.pub_move()

    def move_rightward(self):
        self.leftward -= LINEAR_STEP_SIZE
        self.pub_move() 
        
    def move_upward(self):
        self.upward += LINEAR_STEP_SIZE
        self.pub_move() 
        
    def move_downward(self):
        self.upward -= LINEAR_STEP_SIZE
        self.pub_move() 

    def yaw_increase(self):
        self.angular += ANG_VEL_STEP_SIZE
        self.pub_move()
    
    def yaw_decrease(self):
        self.angular -= ANG_VEL_STEP_SIZE
        self.pub_move()
       