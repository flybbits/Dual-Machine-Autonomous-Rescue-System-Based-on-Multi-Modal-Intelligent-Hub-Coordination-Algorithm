#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
动态目标跟踪节点

该节点负责多旋翼无人机的目标跟踪控制，包括：
1. 健康目标（白色形状）的检测和跟踪
2. 重伤目标（红色区域）的检测和跟踪
3. 基于PID控制器的精确位置控制
4. 动态高度调整和降落控制

"""

import os
import sys
from threading import Thread
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
from geometry_msgs.msg import Pose, PoseStamped, Twist
import numpy as np
import rospy
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String

from src.mission_modules import DeltaPID
from src.utils import (
    create_pose_message,
    create_twist_message,
    detect_red_contours,
    detect_squares,
    is_target_aligned,
    pixel_to_world_with_pose,
    ros2cv,
    calculate_distance,
)

# 全局变量：控制任务启动状态
start = False 

def start_callback(msg):
    """
    启动回调函数
    
    Args:
        msg: 包含启动指令的字符串消息
    """
    global start
    if msg.data == "start":
        start = True

def wait_for_iris():
    """
    订阅IRIS任务状态，当收到"start"指令时开始动态跟踪任务
    """
    global start
    rospy.Subscriber("/iris_mission/status", String, start_callback)
    while not rospy.is_shutdown():
        if start:
            rospy.loginfo("Dynamic Tracker启动")
            break

class MRTracker:
    """多旋翼无人机目标跟踪PID控制器类"""
    
    def __init__(self, uav_vel_topic: str, uav_pose_topic: str):
        """
        初始化跟踪器
        
        Args:
            uav_vel_topic: 无人机速度控制话题
            uav_pose_topic: 无人机位姿控制话题
        """
        # 相机参数初始化
        self.camera_width = None
        self.camera_height = None
        self.camera_center_x = None
        self.camera_center_y = None
        self.camera_matrix = None
        self.dist_coeffs = None
        self.camera_focal_length = None
        
        # 目标检测相关参数
        self.target_detected = False
        self.target_x, self.target_y = 0, 0
        self.vx, self.vy = 0, 0
        self.prev_vx, self.prev_vy = 0, 0
        self.world_x, self.world_y = 0, 0
        self.error_x, self.error_y = 0, 0
        self.detected_point = None
        self.position_threshold = 50  # 像素，当目标与中心距离小于此值时认为已对齐
        self.target = "shape"  # 目标类型："shape"为形状检测，"color"为颜色检测
        self.enable = False  # 跟踪器使能状态
        
        # ROS话题订阅和发布
        self.cmd_vel_pub = rospy.Publisher(uav_vel_topic, Twist, queue_size=1)
        self.target_pub = rospy.Publisher('/tracker/target', Pose, queue_size=1)  
        self.cmd_pose_pub = rospy.Publisher('/xtdrone/iris_0/cmd_pose_enu', Pose, queue_size=1)     
        self.pose_sub = rospy.Subscriber(uav_pose_topic, PoseStamped, self.pose_callback)
        self.camera_info_sub = rospy.Subscriber('/iris_0/camera/camera_info', CameraInfo, self.camera_info_callback)
        self.image_sub = rospy.Subscriber('/iris_0/camera/image_raw', Image, self.image_callback)
        self.status_sub = rospy.Subscriber('/iris_mission/status', String, self.status_callback)
        
        # 控制频率设置
        self.control_rate = rospy.Rate(10)  # 10Hz控制频率
        
        # PID控制器参数配置
        # 前后方向PID (Y轴) - 静态模式
        self.static_pid_forward_backward = DeltaPID(
            dt=0.1, # 控制周期
            p=0.2,  # 进一步增大比例系数，提高响应速度
            i=0.025,# 适当增加积分系数，消除稳态误差
            d=0.04,  # 增加微分系数，提高稳定性
            mode="static"
        )
        # 左右方向PID (X轴) - 静态模式
        self.static_pid_left_right = DeltaPID(
            dt=0.1,  
            p=0.2, 
            i=0.025,
            d=0.04,
            mode="static"
        )
        
        # 前后方向PID (Y轴) - 动态模式（备选）
        self.dynamic_pid_forward_backward = DeltaPID(
            dt=0.1, 
            p=0.2,  
            i=0.04,
            d=0.04,
            mode="dynamic"
        )
        
        # 左右方向PID (X轴) - 动态模式（备选）
        self.dynamic_pid_left_right = DeltaPID(
            dt=0.1,  
            p=0.3,  
            i=0.08,
            d=0.04,
            mode="dynamic"
        )
    
    def set_target(self, target: str):
        """
        设置目标类型
        
        Args:
            target: 目标类型，"shape"或"color"
        """
        self.target = target
    
    def pose_callback(self, msg: PoseStamped):
        """
        无人机位姿回调函数
        
        Args:
            msg: 位姿消息
        """
        self.pose = msg.pose
    
    def camera_info_callback(self, msg):
        """
        相机信息回调函数
        
        获取相机内参和图像尺寸信息，用于坐标转换
        
        Args:
            msg: 相机信息消息
        """
        self.camera_width = msg.width
        self.camera_height = msg.height
        self.camera_center_x = self.camera_width // 2
        self.camera_center_y = self.camera_height // 2
        self.camera_focal_length = msg.K[0]
        self.camera_matrix = np.array(msg.K).reshape(3, 3)
        self.dist_coeffs = np.array(msg.D)
        # 初始化PID控制器的当前值
        self.static_pid_left_right.cur_val = self.camera_center_x
        self.static_pid_forward_backward.cur_val = self.camera_center_y
        self.dynamic_pid_left_right.cur_val = self.camera_center_x
        self.dynamic_pid_forward_backward.cur_val = self.camera_center_y
        
    def status_callback(self, msg):
        """
        任务状态回调函数
        
        根据任务状态切换目标类型和运行模式
        
        Args:
            msg: 状态消息
        """
        if msg.data == "healthy":
            self.enable = True
            self.target = "shape"
            self.healthy_run()
        elif msg.data == "critical":
            self.enable = True
            self.target = "color"
            self.critical_run()
        elif msg.data == "completed":
            self.enable = False
            rospy.signal_shutdown("Dynamic Tracker关闭")
        
    def image_callback(self, msg):
        """
        图像回调函数
        
        处理相机图像，根据目标类型调用相应的检测算法，
        并显示处理结果和状态信息
        
        Args:
            msg: 图像消息
        """
        try:    
            
            cv_image = ros2cv(msg, 'bgr8')
            # 根据目标类型处理图像
            if self.target == "shape":
                display_image = self.white_image_callback(msg)
            elif self.target == "color":
                display_image = self.red_image_callback(msg)
            else:
                # 默认情况下使用原始图像
                display_image = cv_image.copy()
            
            
            if display_image is None or display_image.size == 0:
                rospy.logwarn("无效的图像数据")
                return
            # 状态信息文本
            status_texts = [
                f"Mode: {self.target}",
                f"Tracking: {'ON' if self.enable else 'OFF'}"
            ]
            # 绘制文本
            for i, text in enumerate(status_texts):
                y_pos = 20 + i * 15
                cv2.putText(display_image, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            # 绘制相机中心
            if self.camera_center_x is not None and self.camera_center_y is not None:
                cv2.circle(display_image, (self.camera_center_x, self.camera_center_y), 5, (255, 0, 0), 2)
            
            # 显示处理结果
            cv2.namedWindow("Tracker", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Tracker", 800, 600)
            cv2.imshow('Tracker', display_image)
            
            # 处理键盘输入
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                rospy.loginfo("用户退出")
                rospy.signal_shutdown("用户退出")
        except Exception as e:
            rospy.logerr(f"图像处理错误: {e}")
    
    def white_image_callback(self, msg):
        """
        白色目标图像处理回调函数
        
        检测白色形状目标（健康人员），包括正方形和白色区域检测
        
        Args:
            msg: 图像消息
            
        Returns:
            display_image: 处理后的显示图像
        """
        try:
            cv_image = ros2cv(msg, 'bgr8')
            if not self.enable:
                return cv_image
            
            # 检测正方形目标
            squares = detect_squares(cv_image, 
                                   min_area=200,
                                   max_area=900000,
                                   aspect_ratio_tolerance=0.2,
                                   edge_thickness=3)
            display_image = cv_image.copy()
            if squares:
                self.target_detected = True
                largest_square = max(squares, key=lambda s: s['area'])
                self.target_x, self.target_y = largest_square['center']
                
                cx, cy = largest_square['center']
                cv2.drawContours(display_image, [largest_square['contour']], -1, (0, 255, 0), 2)
                cv2.polylines(display_image, [largest_square['approx']], True, (255, 0, 0), 2)
                cv2.circle(display_image, (cx, cy), 5, (0, 0, 255), -1)
                cv2.line(display_image, (cx, cy), (self.camera_center_x, self.camera_center_y), (0, 0, 255), 2)
                 
            elif self.pose.position.z <= 3:
                # 低空时若检测不到正方形轮廓则检测白色区域
                hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
                lower_white = (0, 0, 200)
                upper_white = (180, 40, 255)
                mask = cv2.inRange(hsv, lower_white, upper_white)

                points = cv2.findNonZero(mask)
                if points is not None and len(points) > 50:
                    avg = points.mean(axis=0)[0]
                    cx, cy = int(avg[0]), int(avg[1])

                    self.target_detected = True
                    self.target_x, self.target_y = cx, cy

                    cv2.circle(display_image, (cx, cy), 5, (255, 0, 0), -1)
                    
                    cv2.line(display_image, (cx, cy), (self.camera_center_x, self.camera_center_y), (0, 0, 255), 2)
                    
                else:
                    self.target_detected = False
                    self.target_x, self.target_y = 0, 0
                    self.error_x, self.error_y = 0, 0
                    print("未观测到目标")                    
            
            return display_image 

        except Exception as e:
            rospy.logerr(f"图像处理错误: {e}")
            self.target_detected = False
            self.target_x, self.target_y = 0, 0
            self.vx = 0.0
            self.vy = 0.0
            try:
                return ros2cv(msg, 'bgr8')
            except:
                return None
    
    def red_image_callback(self, msg):
        """
        红色目标图像处理回调函数
        
        检测红色区域目标（重伤人员）
        
        Args:
            msg: 图像消息
            
        Returns:
            display_image: 处理后的显示图像
        """
        try:
            image = ros2cv(msg, 'bgr8')
            display_image = image.copy()
            
            if not self.enable:
                return display_image
                
            contours = detect_red_contours(image)
            
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_contour)
                if area > 100:
                    M = cv2.moments(largest_contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        cv2.circle(display_image, (cx, cy), 10, (0, 255, 0), 2)
                        if self.camera_center_x is not None and self.camera_center_y is not None:
                            cv2.line(display_image, (cx, cy), (self.camera_center_x, self.camera_center_y), (0, 0, 255), 2)
                        
                        self.target_detected = True
                        self.target_x, self.target_y = cx, cy
            else:
                self.target_detected = False
                self.target_x, self.target_y = 0, 0
            
            return display_image
            
        except Exception as e:
            rospy.logerr(f"图像处理错误: {e}")
            try:
                return ros2cv(msg, 'bgr8')
            except:
                return None
    
    def calculate_control_output(self, mode="static"):
        """
        计算控制输出
        
        基于PID控制器计算速度命令，支持静态和动态两种模式
        
        Args:
            mode: 控制模式，"static"或"dynamic"
            
        Returns:
            tuple: (vx, vy) 速度命令
        """
        if not self.target_detected:
            return 0, 0
        
        # 根据模式选择PID控制器控制模型
        if mode.lower() == "static":
            pid_forward_backward = self.static_pid_forward_backward
            pid_left_right = self.static_pid_left_right
        elif mode.lower() == "dynamic":
            pid_forward_backward = self.dynamic_pid_forward_backward
            pid_left_right = self.dynamic_pid_left_right
            
        # 动态限速（低空慢速）
        self.max_linear_x = 2 if self.pose.position.z > 5 else 0.7 if self.pose.position.z > 2 else 0.3
        self.max_linear_y = 2 if self.pose.position.z > 5 else 0.7 if self.pose.position.z > 2 else 0.3

        # PID输出（单位：速度方向的增量，单位米）
        vx_output = pid_forward_backward.calculate(self.pose.position.z, self.camera_focal_length, self.target_y)
        vy_output = pid_left_right.calculate(self.pose.position.z, self.camera_focal_length, self.target_x)
        
        if vx_output is None or vy_output is None:
            print("left_right_output or forward_backward_output is None")
            return 0, 0

        # 低通滤波（抗目标抖动）
        alpha = 0.8
        if not hasattr(self, 'last_vx'):
            self.last_vx, self.last_vy = 0, 0

        vx_output = alpha * self.last_vx + (1 - alpha) * vx_output
        vy_output = alpha * self.last_vy + (1 - alpha) * vy_output
        
        self.last_vx, self.last_vy = vx_output, vy_output

        # 限制最大速度
        self.vx = np.clip(vx_output, -self.max_linear_x, self.max_linear_x)
        self.vy = np.clip(vy_output, -self.max_linear_y, self.max_linear_y)
    
    
    def get_smooth_descent_speed(self, cur_height, min_h=0.5, max_h=15, max_vz=-1.0, min_vz=-0.4):
        """
        根据当前高度，平滑计算下降速度
        
        Args:
            cur_height: 当前高度
            min_h: 最小高度阈值
            max_h: 最大高度阈值
            max_vz: 最大下降速度
            min_vz: 最小下降速度
            
        Returns:
            float: 计算得到的下降速度
        """
        ratio = (cur_height - min_h) / (max_h - min_h)
        ratio = max(0.0, min(1.0, ratio))  # 限定在 [0,1]
        return ratio * max_vz + (1 - ratio) * min_vz
    
    def healthy_run(self):
        """
        健康目标跟踪运行模式
        
        专门用于跟踪健康人员（白色形状目标）的控制逻辑
        
        Args:
            mode: 控制模式
        """
        
        rospy.loginfo("开始跟踪健康人员")
        
        self.cmd_vel_pub.publish(create_twist_message())
        while not rospy.is_shutdown():
            tag = False
            landing_pixel = (0, 0)
            time.sleep(10)
            since = time.time()
            if self.pose.position.z > 3.5:
                print("识别轨迹中...")
                self.enable = True
                while (time.time() - since <= 15):
                    if self.target_y > landing_pixel[1]:
                        print(f"更新最低点为({self.target_x}, {self.target_y})")
                        landing_pixel = (self.target_x, self.target_y)
                        tag = True
                self.enable = False
                
                if tag:
                    # 动态scale计算（线性映射，高空小，低空大)
                    S_min = 0.7  # 高空最小系数
                    S_max = 1.2  # 低空最大系数
                    z_min = 2  # 低空阈值
                    z_max = 15.0   # 高空阈值
                    zc = max(min(self.pose.position.z, z_max), z_min)
                    t = (z_max - zc) / (z_max - z_min)
                    scale = S_min + (S_max - S_min) * t
                    pixel_error_x = self.camera_center_y - landing_pixel[1]
                    pixel_error_y = self.camera_center_x - landing_pixel[0]
                    
                    self.error_x = (self.pose.position.z / self.camera_focal_length) * pixel_error_x * (scale + 0.6)
                    self.error_y = (self.pose.position.z / self.camera_focal_length) * pixel_error_y * (scale + 0.3)
                    
                    print(f"更新步长:({self.error_x:.2f}m, {self.error_y:.2f}m)")
                    self.cmd_pose_pub.publish(create_pose_message((self.pose.position.x + self.error_x + 2, self.pose.position.y + self.error_y - 2, max(0.5, self.pose.position.z - 2))))
            
            else:
                self.enable = False    
                self.target_detected = False    
                while self.pose.position.z > 0.7:
                    self.cmd_vel_pub.publish(create_twist_message(0, 0, -0.5))
                    time.sleep(0.1)
                for _ in range(50):
                    self.cmd_vel_pub.publish(create_twist_message())
                while not self.target_detected:
                    print("悬停识别中...")
                    self.enable = True
                    self.cmd_vel_pub.publish(create_twist_message())
                prev_dist = (self.camera_width ** 2 + self.camera_height ** 2) ** 0.5
                cur_dist = prev_dist
                while cur_dist <= prev_dist:
                    prev_dist = cur_dist
                    cur_dist = calculate_distance((self.target_x, self.target_y), (self.camera_center_x, self.camera_center_y))
                self.world_x, self.world_y = self.pose.position.x, self.pose.position.y
                self.target_pub.publish(create_pose_message((self.world_x, self.world_y, 0)))
                self.cmd_pose_pub.publish(create_pose_message((self.world_x, self.world_y, 3)))
                self.enable = False
                break
            self.target_x, self.target_y = 0, 0
        
    def critical_run(self):
        """
        主控制循环
        
        执行目标跟踪的主要控制逻辑，包括目标检测、PID控制和降落
        """
        rospy.loginfo("开始跟踪危重人员")
        
        while self.enable and not rospy.is_shutdown():
            self.calculate_control_output("static")
            # 检查目标坐标是否有效
            if self.target_detected and self.target_x is not None and self.target_y is not None:
                # 检查是否对齐
                self.world_x, self.world_y = pixel_to_world_with_pose(self.target_x, self.target_y, self.camera_matrix, self.pose, self.pose.position.z)
                is_aligned = is_target_aligned((self.target_x, self.target_y), (self.camera_center_x, self.camera_center_y), self.position_threshold)
                if self.pose.position.z < 0.4 and is_aligned:
                    self.target_pub.publish(create_pose_message((self.world_x, self.world_y, 0)))
                    self.cmd_pose_pub.publish(create_pose_message((self.world_x, self.world_y, 3)))
                    time.sleep(2)
                    self.enable = False
                    break
                cmd_vel = create_twist_message(self.vx, self.vy, self.get_smooth_descent_speed(self.pose.position.z))
                print(f"当前高度：{self.pose.position.z:.2f}m")
            else:
                cmd_vel = create_twist_message()
            self.cmd_vel_pub.publish(cmd_vel)
            self.control_rate.sleep()
            
def main():
    try:      
        rospy.init_node("dynamic_tracker")
        
        # 等待IRIS任务就绪
        tracker_start_thread = Thread(target=wait_for_iris)
        tracker_start_thread.start()
        tracker_start_thread.join()
        
        # 创建跟踪器实例
        tracker = MRTracker(
            uav_vel_topic="/xtdrone/iris_0/cmd_vel_flu",
            uav_pose_topic="/iris_0/mavros/vision_pose/pose",
        )
        
        rospy.spin()

    except rospy.ROSInterruptException:
        rospy.loginfo("程序被中断")
    except Exception as e:
        rospy.logerr(f"程序错误: {e}")

if __name__ == '__main__':
    
    main() 
