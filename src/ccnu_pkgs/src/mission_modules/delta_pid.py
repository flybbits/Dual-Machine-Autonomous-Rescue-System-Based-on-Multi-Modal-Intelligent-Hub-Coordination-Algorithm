#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增量式PID控制器模块

该模块实现了增量式PID控制算法，支持静态和动态两种模式：
1. 静态模式：适用于固定目标跟踪
2. 动态模式：适用于移动目标跟踪
3. 自适应参数：根据高度动态调整PID参数
"""

import numpy as np

class DeltaPID(object):
    """
    增量式PID控制器类（静态动态双模式）
    """

    def __init__(self, dt, p, i, d, mode="static")-> None:
        """
        初始化PID控制器
        
        Args:
            dt: 控制周期（秒）
            p: 比例系数
            i: 积分系数
            d: 微分系数
            mode: 控制模式，"static"或"dynamic"
        """
        self.dt = dt
        self.k_p = p
        self.k_i = i
        self.k_d = d
        self.mode = mode
        
        # 状态变量
        self.cur_val = 0      # 当前值
        self.last_target = 0  # 上一时刻目标值
        self.target = 0       # 当前目标值
        
        # 动态模式相关变量
        self.v_target = 0.0   # 目标速度
        self.last_output = 0  # 上一时刻输出
        
        # 误差历史
        self._pre_error = 0      # 上一时刻误差
        self._pre_pre_error = 0  # 上上时刻误差

    def calculate(self, height, focal_length, new_target):
        """
        计算PID控制输出
        
        根据当前高度自适应调整PID参数，计算控制增量
        
        Args:
            height: 当前高度
            focal_length: 相机焦距
            new_target: 新的目标值
            
        Returns:
            float: 控制增量输出
        """
        # 限制高度范围在0.5-10米之间
        h = np.clip(height, 0.5, 10)

        # 根据高度自适应调整PID参数
        # 低空时使用较小的参数，高空时使用较大的参数
        k_p = np.interp(h, [0.5, 10], [0.03, 0.1])
        k_i = np.interp(h, [0.5, 10], [0.004, 0.02])
        k_d = np.interp(h, [0.5, 10], [0.005, 0.02])
        
        if self.mode == "static":
            # 静态模式：适用于固定目标跟踪
            self.target = new_target
            error = self.cur_val - self.target

            # 死区处理：误差小于4时忽略
            if abs(error) < 4:
                error = 0.0
                
            # 计算PID增量
            p_change = k_p * (error - self._pre_error)  # 比例项增量
            i_change = k_i * error                      # 积分项增量
            d_change = k_d * (error - 2 * self._pre_error + self._pre_pre_error)  # 微分项增量
            delta_output = p_change + i_change + d_change

            # 更新误差历史
            self._pre_pre_error = self._pre_error
            self._pre_error = error

            return delta_output 

        if self.mode == "dynamic":
            # 动态模式：适用于移动目标跟踪
            self.target = new_target
            # 计算目标速度
            self.v_target = (self.target - self.last_target) / self.dt
            
            # 预测误差：考虑目标运动
            error = self.cur_val - (self.target + self.v_target * self.dt)
            if abs(error) < 4:
                error = 0.0

            # 计算PID增量
            p_change = k_p * (error - self._pre_error)
            i_change = k_i * error
            d_change = k_d * (error - 2 * self._pre_error + self._pre_pre_error)
            delta_output = p_change + i_change + d_change
            
            # 更新状态
            self.last_output = delta_output
            self.last_target = self.target
            self._pre_pre_error = self._pre_error
            self._pre_error = error
            
            return delta_output

