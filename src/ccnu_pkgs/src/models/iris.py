#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IRIS多旋翼无人机模型

该模块定义了IRIS多旋翼无人机的具体实现，包括：
1. 多旋翼飞行控制
2. 目标检测结果发布
3. 位置和速度命令发送
4. 相机和传感器数据处理
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from geometry_msgs.msg import Pose, PoseStamped, Twist
from mavros_msgs.msg import State, Altitude
import rospy
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String

from utils import EulerAndQuaternionTransform

from .uav import UAV


class Iris(UAV):
    def __init__(self):
        super(Iris, self).__init__(uav_name='iris')
        
        # 飞行控制发布器
        self.cmd_vel_flu_pub = rospy.Publisher('/xtdrone/iris_0/cmd_vel_flu', Twist, queue_size=1)
        self.cmd_mode_pub = rospy.Publisher('/xtdrone/iris_0/cmd', String, queue_size=3) 
        self.cmd_pose_enu_pub = rospy.Publisher('/xtdrone/iris_0/cmd_pose_enu', Pose, queue_size=3)
        
        # 目标检测结果发布器
        self.critical_pub = rospy.Publisher('/zhihang2025/iris_bad_man/pose', Pose, queue_size=10)
        self.healthy_pub = rospy.Publisher('/zhihang2025/iris_healthy_man/pose', Pose, queue_size=10)
        
        # 目标位置订阅器
        rospy.Subscriber('/zhihang2025/first_man/pose', Pose, self.critical_callback)
        rospy.Subscriber('/zhihang2025/third_man/pose', Pose, self.healthy_callback)
        
        # 状态和传感器订阅器
        rospy.Subscriber('/iris_0/mavros/state', State, self.state_callback)
        rospy.Subscriber("/iris_0/camera/camera_info", CameraInfo, self.camera_info_callback)
        rospy.Subscriber("/iris_0/camera/image_raw", Image, self.camera.image_callback, 'iris')
        rospy.Subscriber("/iris_0/mavros/vision_pose/pose", PoseStamped, self.pose_callback)
        rospy.Subscriber('/iris_0/mavros/altitude', Altitude, self.alt_callback)
    
    def critical_callback(self, msg: Pose):
        """
        重伤人员位置回调函数
        
        Args:
            msg: 包含重伤人员位置的Pose消息
        """
        self.critical_point = (msg.position.x, msg.position.y)
        
    def healthy_callback(self, msg: Pose):
        """
        健康人员位置回调函数
        
        Args:
            msg: 包含健康人员位置的Pose消息
        """
        self.healthy_point = (msg.position.x, msg.position.y)
        
    def pub_move(self):
        self.send_vel_command(self.forward, self.leftward, self.upward)
    
    def send_vel_command(self, vx: float, vy: float, vz: float):
        self.twist.linear.x = vx
        self.twist.linear.y = vy
        self.twist.linear.z = vz
        self.twist.angular.x = 0.0
        self.twist.angular.y = 0.0
        self.twist.angular.z = self.yaw
        self.cmd_vel_flu_pub.publish(self.twist)
    
    def send_rotation_command(self, yaw: float):
        pose = Pose()
        orien = EulerAndQuaternionTransform((0, 0, yaw))
        pose.orientation.x = orien[0]
        pose.orientation.y = orien[1]
        pose.orientation.z = orien[2]
        pose.orientation.w = orien[3]
        self.cmd_pose_enu_pub.publish(pose)
    
    def send_target_command(self, x: float, y: float, z: float):
        pose = Pose()
        pose.position.x = x
        pose.position.y = y
        pose.position.z = z
        pose.orientation.x = 0.0
        pose.orientation.y = 0.0
        pose.orientation.z = 0.0
        pose.orientation.w = 1.0
        self.cmd_pose_enu_pub.publish(pose)
    
    def send_move_command(self, forward: float, leftward: float, upward: float, yaw: float=None):
        self.forward = forward
        self.leftward = leftward
        self.upward = upward
        if yaw is not None:
            self.yaw = yaw
        self.pub_move()
