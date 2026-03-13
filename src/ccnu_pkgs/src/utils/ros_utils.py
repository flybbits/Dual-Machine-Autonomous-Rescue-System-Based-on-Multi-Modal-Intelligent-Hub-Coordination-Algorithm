#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROS工具模块
提供ROS消息创建、发布等独立功能函数
"""

import rospy
from geometry_msgs.msg import Twist, Point, Pose, Quaternion
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from typing import Tuple, Optional
from cv_bridge import CvBridge
import numpy as np


def create_twist_message(linear_x: float = 0.0, 
                        linear_y: float = 0.0, 
                        linear_z: float = 0.0,
                        angular_x: float = 0.0, 
                        angular_y: float = 0.0, 
                        angular_z: float = 0.0) -> Twist:
    """
    创建Twist消息
    
    Args:
        linear_x, linear_y, linear_z: 线速度分量
        angular_x, angular_y, angular_z: 角速度分量
        
    Returns:
        Twist: 创建的消息
    """
    twist = Twist()
    twist.linear.x = linear_x
    twist.linear.y = linear_y
    twist.linear.z = linear_z
    twist.angular.x = angular_x
    twist.angular.y = angular_y
    twist.angular.z = angular_z
    return twist


def create_point_message(x: float, y: float, z: float = 0.0) -> Point:
    """
    创建Point消息
    
    Args:
        x, y, z: 坐标值
        
    Returns:
        Point: 创建的消息
    """
    point = Point()
    point.x = x
    point.y = y
    point.z = z
    return point


def create_pose_message(position: Tuple[float, float, float],
                       orientation: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)) -> Pose:
    """
    创建Pose消息
    
    Args:
        position: 位置 (x, y, z)
        orientation: 方向四元数 (x, y, z, w)
        
    Returns:
        Pose: 创建的消息
    """
    pose = Pose()
    pose.position = create_point_message(position[0], position[1], position[2])
    pose.orientation = Quaternion(orientation[0], orientation[1], orientation[2], orientation[3])
    return pose


def create_header(frame_id: str = "base_link", 
                 stamp: Optional[rospy.Time] = None) -> Header:
    """
    创建Header消息
    
    Args:
        frame_id: 坐标系ID
        stamp: 时间戳
        
    Returns:
        Header: 创建的消息
    """
    header = Header()
    header.frame_id = frame_id
    header.stamp = stamp if stamp is not None else rospy.Time.now()
    return header


def cv2ros(cv_image: np.ndarray, 
                         encoding: str = "bgr8",
                         frame_id: str = "camera_frame") -> Image:
    """
    将OpenCV图像转换为ROS Image消息
    
    Args:
        cv_image: OpenCV图像
        encoding: 图像编码格式
        frame_id: 坐标系ID
        
    Returns:
        Image: ROS Image消息
    """
    bridge = CvBridge()
    try:
        ros_image = bridge.cv2_to_imgmsg(cv_image, encoding)
        ros_image.header = create_header(frame_id)
        return ros_image
    except Exception as e:
        rospy.logerr(f"转换图像失败: {e}")
        return Image()


def ros2cv(ros_image: Image, 
                         encoding: str = "bgr8") -> Optional[np.ndarray]:
    """
    将ROS Image消息转换为OpenCV图像
    
    Args:
        ros_image: ROS Image消息
        encoding: 目标编码格式
        
    Returns:
        Optional[np.ndarray]: OpenCV图像或None
    """
    bridge = CvBridge()
    try:
        cv_image = bridge.imgmsg_to_cv2(ros_image, encoding)
        return cv_image
    except Exception as e:
        rospy.logerr(f"转换图像失败: {e}")
        return None
