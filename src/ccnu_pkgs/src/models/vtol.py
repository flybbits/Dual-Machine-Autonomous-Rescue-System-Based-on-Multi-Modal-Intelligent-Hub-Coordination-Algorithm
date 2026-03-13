#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VTOL垂直起降无人机模型

该模块定义了VTOL垂直起降无人机的具体实现，包括：
1. 多旋翼和固定翼模式切换
2. 垂直起飞和水平飞行控制
3. 目标检测结果发布
4. 相机和传感器数据处理
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from geometry_msgs.msg import Pose, PoseStamped, Twist
from mavros_msgs.msg import Altitude, State
import rospy
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String

from utils import EulerAndQuaternionTransform

from .uav import UAV


class Vtol(UAV):
    def __init__(self):
        super(Vtol, self).__init__(uav_name='vtol')
        self.platform = "multirotor"  # 初始为多旋翼模式
        
        # 飞行控制发布器
        self.cmd_pose_enu_pub = rospy.Publisher('/xtdrone/standard_vtol_0/cmd_pose_enu', Pose, queue_size=1)
        self.cmd_vel_flu_pub = rospy.Publisher('/xtdrone/standard_vtol_0/cmd_vel_flu', Twist, queue_size=1)
        self.cmd_mode_pub = rospy.Publisher('/xtdrone/standard_vtol_0/cmd', String, queue_size=3) 
        
        # 目标检测结果发布器
        self.critical_pub = rospy.Publisher('/zhihang2025/first_man/pose', Pose, queue_size=10)
        self.injured_pub = rospy.Publisher('/zhihang2025/second_man/pose', Pose, queue_size=10)
        self.healthy_pub = rospy.Publisher('/zhihang2025/third_man/pose', Pose, queue_size=10)
    
        # 状态和传感器订阅器
        rospy.Subscriber('/standard_vtol_0/mavros/state', State, self.state_callback)
        rospy.Subscriber('/standard_vtol_0/mavros/vision_pose/pose', PoseStamped, self.pose_callback)
        rospy.Subscriber('/standard_vtol_0/camera/image_raw', Image, self.camera.image_callback, 'vtol')
        rospy.Subscriber('/standard_vtol_0/camera/camera_info', CameraInfo, self.camera_info_callback)
        rospy.Subscriber('/standard_vtol_0/mavros/altitude', Altitude, self.alt_callback)
    
    def transition(self):
        """
        模式转换
        
        在多旋翼模式和固定翼模式之间切换
        """
        if self.platform == 'multirotor':
            self.platform = 'plane'
        else:
            self.platform = 'multirotor'
        self.mode.data = self.platform
        self.pub_cmd()   
    
    def send_target_command(self, x: float, y: float, z: float, yaw: float = 0.0):
            pose = Pose()   
            pose.position.x = x
            pose.position.y = y
            pose.position.z = z
            orien = EulerAndQuaternionTransform((0.0, 0.0, yaw))
            pose.orientation.x = orien[0]
            pose.orientation.y = orien[1]
            pose.orientation.z = orien[2]
            pose.orientation.w = orien[3]
            self.cmd_pose_enu_pub.publish(pose)
        
    def pub_move(self):
        if self.platform == "multirotor":
            # 多旋翼模式：发布速度命令
            self.twist.linear.x = self.forward
            self.twist.linear.y = self.leftward
            self.twist.linear.z = self.upward
            self.twist.angular.x = 0.0
            self.twist.angular.y = 0.0
            self.twist.angular.z = self.yaw
            self.cmd_vel_flu_pub.publish(self.twist)
        elif self.platform == "plane":
            # 固定翼模式：发布位置命令
            self.pose.position.x = self.forward
            self.pose.position.y = self.leftward
            self.pose.position.z = self.upward
            self.pose.orientation.x = 0.0
            self.pose.orientation.y = 0.0
            self.pose.orientation.z = self.yaw
            self.cmd_pose_enu_pub.publish(self.pose)
        self.pub_cmd()
    
    # def publish_detected_targets(self, world_points: dict):
    #     """
    #     发布检测到的目标位置
        
    #     将检测到的目标位置发布到相应的话题
        
    #     Args:
    #         world_points: 包含目标世界坐标的字典
    #     """
    #     mode_str = self.camera.detector.__class__.__name__
        
    #     if world_points is not None:
    #         for target_type, (x, y, z) in world_points.items():
    #             # 检查是否已经发布过该类型目标
    #             if self.camera.detector.targets_found[target_type]:
    #                 continue

    #             # 创建位置消息
    #             pose_msg = Pose()
    #             pose_msg.position.x = x
    #             pose_msg.position.y = y
    #             pose_msg.position.z = z
    #             pose_msg.orientation.w = 1.0
                
    #             # 根据目标类型发布到相应话题
    #             if target_type == 'critical':
    #                 self.critical_pub.publish(pose_msg)
    #                 rospy.loginfo(f"{mode_str}发布重伤人员位置: ({x:.1f}, {y:.1f}, {z:.1f})")
    #             elif target_type == 'injured':
    #                 self.injured_pub.publish(pose_msg)
    #                 rospy.loginfo(f"{mode_str}发布轻伤人员位置: ({x:.1f}, {y:.1f}, {z:.1f})")
    #             elif target_type == 'healthy':
    #                 self.healthy_pub.publish(pose_msg)
    #                 rospy.loginfo(f"{mode_str}发布健康人员位置: ({x:.1f}, {y:.1f}, {z:.1f})")
                
    #             # 标记该类型目标已发布
    #             self.camera.detector.targets_found[target_type] = True
    #             self.camera.detector.track_history[target_type].append((x, y, z))
                
    #             rospy.loginfo(f"{target_type}目标已发布，后续不再关注该类别")
            
            
            