#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
几何计算工具模块
提供距离计算、对齐检查等几何相关功能函数
""" 

import math
import numpy as np
from typing import Tuple, List
from geometry_msgs.msg import Pose
from scipy.spatial.transform import Rotation as R

    
def calculate_distance(point1: Tuple[float, float], 
                                point2: Tuple[float, float]) -> float:
    """
    计算两点间的欧几里得距离
    
    Args:
        point1: 第一个点 (x, y)
        point2: 第二个点 (x, y)
        
    Returns:
        float: 两点间的距离
    """
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


def is_target_aligned(target_point: Tuple[float, float], 
                     reference_point: Tuple[float, float], 
                     threshold: float) -> bool:
    """
    检查目标是否已对齐
    
    Args:
        target_point: 目标点坐标 (x, y)
        reference_point: 参考点坐标 (x, y)
        threshold: 对齐阈值
        
    Returns:
        bool: 是否已对齐
    """
    distance = calculate_distance(target_point, reference_point)
    return distance < threshold


def get_intercept_point(r, x0, y0, x1, y1, x2, y2) -> Tuple[float, float]:
        """
        计算圆外两过定点切线的交点坐标
        Args:
            r: 半径长度(m)
            x0: 圆心x坐标
            y0: 圆心y坐标
            x1: 圆外定点一x坐标
            y1: 圆外定点一y坐标
            x2: 圆外定点二x坐标
            y2: 圆外定点二y坐标
        Returns:
            (x, y)：圆外两条过定点切线的交点坐标
        """

        def compute1(x0, y0, x1, y1):
            q = x0 - x1
            p = y1 - y0
            a = q**2 - r**2
            b = 2 * p * q
            c = p ** 2 - r**2
            m = (-b - np.sqrt(b**2 - 4 * a * c)) / (2 * a)
            return m

        def compute2(x0, y0, x1, y1):
            q = x0 - x1
            p = y1 - y0
            a = q**2 - r**2
            b = 2 * p * q
            c = p ** 2 - r**2
            m = (-b + np.sqrt(b**2 - 4 * a * c)) / (2 * a)
            return m

        m1 = compute1(x0, y0, x1, y1)
        m2 = compute2(x0, y0, x2, y2)

        def find_tangent_intersection(m1, m2, x1, y1, x2, y2):
            A1, B1 = m1, -1
            A2, B2 = m2, -1
            C1 = -m1*x1 + y1
            C2 = -m2*x2 + y2
            
            A = np.array([[A1, B1], [A2, B2]])
            B = np.array([-C1, -C2])
            intersection_point = np.linalg.solve(A, B)
            return intersection_point

        (x, y) = find_tangent_intersection(m1, m2, x1, y1, x2, y2)
        return (x, y)
    
def pixel_to_world_with_pose(
    u: float,
    v: float,
    K: np.ndarray,
    camera_pose: Pose,
    camera_alt: float
) -> Tuple[float, float]:
    """
    将图像像素坐标 (u, v) 转换为地面上的世界坐标 (X_w, Y_w)，
    支持任意相机姿态（四元数），适用于无人机/机器人等实际场景。

    Args：
        u, v: 图像像素坐标（float/int）
        K: 3x3 相机内参矩阵（numpy.ndarray）
        camera_pose: geometry_msgs/Pose，包含相机的世界坐标和四元数姿态
        camera_alt: float，相机离地面高度（米），建议用准确高度

    Returns：
        (X, Y): 地面世界坐标（米）
    """
    Z = camera_alt
    f = K[0, 0]
    cx = K[0, 2]
    cy = K[1, 2]
    X = (u - cx) * Z / f
    Y = (v - cy) * Z / f
    cos_theta = math.cos(camera_pose.orientation.z)
    sin_theta = math.sin(camera_pose.orientation.z)
    X_w = cos_theta * X - sin_theta * Y + camera_pose.position.x
    Y_w = sin_theta * X + cos_theta * Y + camera_pose.position.y
    
    return X_w, Y_w

def pixel_to_world_with_pred(
    u: float,
    v: float,
    K: np.ndarray,
    camera_pose: Pose,
    camera_alt: float
) -> Tuple[float, float]:
    pixel = np.array([u, v, 1.0])
    K_inv = np.linalg.inv(K)
    norm_coords     = K_inv @ pixel
    camera_pos = np.array([
        camera_pose.position.x,
        camera_pose.position.y,
        camera_alt
    ])
    q = camera_pose.orientation
    rotation_matrix = R.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
    world_ray = rotation_matrix @ norm_coords
    if abs(world_ray[2]) > 1e-6:
        t = -camera_pos[2] / world_ray[2]
        world_point = camera_pos + t * world_ray
        return float(world_point[0]), float(world_point[1])
    else:
        return float('inf'), float('inf')
    

def EulerAndQuaternionTransform(input_data: Tuple) -> List:
    """
    四元数与欧拉角互换
    Args:
        input_data: 四元数或欧拉角元组
    Returns:
        转换后的欧拉角或四元数列表
    """
    data_len = len(input_data)
    angle_is_not_rad = False
 
    if data_len == 3:
        r = 0
        p = 0
        y = 0
        if angle_is_not_rad: # 180 ->pi
            r = math.radians(input_data[0]) 
            p = math.radians(input_data[1])
            y = math.radians(input_data[2])
        else:
            r = input_data[0] 
            p = input_data[1]
            y = input_data[2]
 
        sinp = math.sin(p/2)
        siny = math.sin(y/2)
        sinr = math.sin(r/2)
 
        cosp = math.cos(p/2)
        cosy = math.cos(y/2)
        cosr = math.cos(r/2)
 
        w = cosr*cosp*cosy + sinr*sinp*siny
        x = sinr*cosp*cosy - cosr*sinp*siny
        y = cosr*sinp*cosy + sinr*cosp*siny
        z = cosr*cosp*siny - sinr*sinp*cosy
        return [x, y, z, w]
 
    elif data_len == 4:
 
        w = input_data[3] 
        x = input_data[0]
        y = input_data[1]
        z = input_data[2]
 
        r = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        p = math.asin(2 * (w * y - z * x))
        y = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
 
        if angle_is_not_rad : # pi -> 180
            r = math.degrees(r)
            p = math.degrees(p)
            y = math.degrees(y)
        return [r,p,y]
    
