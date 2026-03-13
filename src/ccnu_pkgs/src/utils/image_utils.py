#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图像处理工具模块
提供图像预处理、颜色检测、形状检测等独立功能函数
"""

import cv2
from cv2.typing import MatLike
import numpy as np
from typing import Sequence, Tuple, List, Dict, Optional


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    图像预处理
    
    Args:
        image: 输入图像
        
    Returns:
        np.ndarray: 预处理后的图像
    """
    kernel = np.ones((8, 8), np.uint8)
    
    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 高斯模糊去噪
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # 二值化
    _, binary = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY)
    
    binary = cv2.dilate(binary, kernel)
    
    # 形态学操作
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    
    return closed


def create_color_mask(hsv_image: np.ndarray, color: str) -> np.ndarray:
    """
    创建颜色掩码
    
    Args:
        hsv_image: HSV格式的图像
        color: 目标颜色 ('red', 'blue', 'green')
        
    Returns:
        np.ndarray: 颜色掩码
    """
    if color == 'red':
        # 红色范围（HSV中红色跨越0度）
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
        mask = mask1 + mask2
        
    elif color == 'blue':
        lower_blue = np.array([100, 100, 100])
        upper_blue = np.array([130, 255, 255])
        mask = cv2.inRange(hsv_image, lower_blue, upper_blue)
        
    elif color == 'green':
        lower_green = np.array([40, 100, 100])
        upper_green = np.array([80, 255, 255])
        mask = cv2.inRange(hsv_image, lower_green, upper_green)
        
    elif color == 'gray':
        lower_gray_white = np.array([0, 0, 150])
        upper_gray_white = np.array([180, 60, 255])
        mask = cv2.inRange(hsv_image, lower_gray_white, upper_gray_white)
        
    else:
        # 默认返回全零掩码
        mask = np.zeros(hsv_image.shape[:2], dtype=np.uint8)
    
    return mask


def calculate_contour_center(contour: np.ndarray) -> Optional[Tuple[int, int]]:
    """
    计算轮廓中心点
    
    Args:
        contour: 轮廓
        
    Returns:
        Optional[Tuple[int, int]]: 中心点坐标 (x, y) 或 None
    """
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        return cx, cy
    return None


def detect_squares(image: np.ndarray, 
                  min_area: float = 100, 
                  max_area: float = 900000,
                  aspect_ratio_tolerance: float = 0.15,
                  edge_thickness: int = 2) -> List[Dict]:
    """
    检测图像中的正方形轮廓
    
    Args:
        image: 原始图像
        min_area: 最小面积
        max_area: 最大面积
        aspect_ratio_tolerance: 长宽比容差
        edge_thickness: 边缘厚度（像素）
        
    Returns:
        List[Dict]: 检测到的正方形轮廓列表
    """
    
    # 灰色掩码
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = create_color_mask(hsv, 'gray')

    # 形态学去噪
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    # 应用掩膜
    filtered_image = cv2.bitwise_and(image, image, mask=mask)
    gray = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2GRAY)

    # 使用Canny边缘检测
    edges = cv2.Canny(gray, 50, 150)
    
    # 形态学操作来连接边缘
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (edge_thickness, edge_thickness))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    # 查找轮廓
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    squares = []
    
    for contour in contours:
        # 计算轮廓面积
        area = cv2.contourArea(contour)
        
        # 面积过滤
        if area < min_area or area > max_area:
            continue
        
        # 获取边界矩形
        x, y, w, h = cv2.boundingRect(contour)
        
        # 检查是否为正方形（长宽比接近1:1）
        aspect_ratio = w / h if h > 0 else 0
        if abs(aspect_ratio - 1.0) > aspect_ratio_tolerance:
            continue
        
        # 计算轮廓中心
        center = calculate_contour_center(contour)
        if center is None:
            continue
        
        # 计算轮廓的近似多边形
        perimeter = cv2.arcLength(contour, True)
        epsilon = 0.02 * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # 检查是否为四边形（4个顶点）
        if len(approx) == 4:
            # 检查是否为凸四边形
            if cv2.isContourConvex(approx):
                # 计算边长
                sides = []
                for i in range(4):
                    pt1 = approx[i][0]
                    pt2 = approx[(i + 1) % 4][0]
                    side_length = np.sqrt((pt2[0] - pt1[0])**2 + (pt2[1] - pt1[1])**2)
                    sides.append(side_length)
                
                # 检查边长是否相近（正方形特性）
                avg_side = np.mean(sides)
                side_variance = np.var(sides)
                if side_variance < avg_side * 0.1:  # 边长变化不超过10%
                    squares.append({
                        'contour': contour,
                        'center': center,
                        'area': area,
                        'bbox': (x, y, w, h),
                        'approx': approx,
                        'sides': sides,
                        'perimeter': perimeter
                    })
    
    return squares


def draw_detection_results(image: np.ndarray, 
                          target_center: Tuple[int, int], 
                          camera_center: Tuple[int, int],
                          target_type: str = "target") -> np.ndarray:
    """
    在图像上绘制检测结果
    
    Args:
        image: 输入图像
        target_center: 目标中心点
        camera_center: 相机中心点
        target_type: 目标类型标签
        
    Returns:
        np.ndarray: 绘制了检测结果的图像
    """
    # 绘制目标中心
    cv2.circle(image, target_center, 10, (0, 255, 0), 2)
    
    # 绘制相机中心
    cv2.circle(image, camera_center, 5, (255, 0, 0), 2)
    
    # 绘制连接线
    cv2.line(image, target_center, camera_center, (0, 0, 255), 2)
    
    # 添加标签
    cv2.putText(image, target_type, (target_center[0] + 15, target_center[1]), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    return image


def detect_red_contours(image: np.ndarray) -> Sequence[MatLike]:
    """
    检测图像中的红色轮廓
    
    Args:
        image: 原始图像
        
    Returns:
        Sequence[MatLike]: 检测到的轮廓列表
    """
    
    # 转换为HSV颜色空间
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    mask = create_color_mask(hsv, 'red')
    
    # 形态学操作
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # 查找轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    return contours