#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IRIS多旋翼无人机任务节点

该节点负责IRIS无人机的完整任务执行，包括：
1. 起飞和导航到目标区域
2. 目标检测和跟踪协调
3. 与动态跟踪器的任务状态同步
4. 返航和降落
"""

import time
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rospy
from threading import Thread
from geometry_msgs.msg import Pose
from gazebo_msgs.msg import ModelStates
from std_msgs.msg import String

from src.models.iris import Iris
from src.mission_modules.path_planner import PathPlanner
from src.utils import calculate_distance
from src.utils.ros_utils import create_pose_message

    
def tracker_target_callback(msg, args):
    """
    处理来自tracker节点的目标检测结果
    
    接收动态跟踪器检测到的目标位置，并发布到相应的任务话题
    
    Args:
        msg: 目标位置消息
        args: 回调参数，包含IRIS实例和发布器
    """
    global part, stage
    iris, pub = args
    people = ["健康", "危重"]
    world_pose = create_pose_message((msg.position.x, msg.position.y, 0))
    rospy.loginfo(f"发布{people[part]}人员坐标: ({msg.position.x:.2f}, {msg.position.y:.2f}, {0:.2f})")
    if part == 0:
        iris.healthy_pub.publish(world_pose)
        part = 1
    else:
        iris.critical_pub.publish(world_pose)
        stage = 4
        rospy.loginfo("识别完成")

def vtol_status_callback(msg):
    """
    VTOL任务状态回调函数
    
    监听VTOL任务完成状态，当VTOL任务完成后启动IRIS任务
    
    Args:
        msg: 状态消息
    """
    global start
    if msg.data == "completed":
        start = True

def wait_for_vtol():
    """
    等待VTOL任务完成
    
    订阅VTOL任务状态，当收到"completed"指令时开始执行IRIS任务
    """
    global start
    rospy.Subscriber("/vtol_mission/status", String, vtol_status_callback)
    while not rospy.is_shutdown():
        if start:
            rospy.loginfo("VTOL任务完成，IRIS任务启动")
            break
        time.sleep(0.1)
        
# 全局变量定义
start = False    # 任务启动标志
part = 0         # 当前检测的目标部分（0:健康人员, 1:重伤人员）
stage = 0        # 当前任务阶段

def execute_iris_mission(iris: Iris, path_planner: PathPlanner): 
    """
    执行IRIS无人机任务
    
    控制IRIS无人机完成完整的救援任务流程
    
    Args:
        iris: IRIS无人机实例
        path_planner: 路径规划器实例
    """
    
    global part, stage
    takeoff_x, takeoff_y = iris.x, iris.y
    stage2_initialized = False
    stage3_initialized = False
    stage4_initialized = False
    stages = ["初始化", "起飞", "导航", "识别", "返航"]
    rate = rospy.Rate(20)
    
    # 任务状态发布器
    status_pub = rospy.Publisher('/iris_mission/status', String, queue_size=10)
    
    iris.camera.detector.set_roi(0, iris.camera.shape[1], 0, iris.camera.shape[0])
    rospy.sleep(1)  
    rospy.loginfo("IRIS任务开始")
    while not rospy.is_shutdown():
        print("**********IRIS Mission**********")
        print(f"Stage {stage}: {stages[stage]}")
        print(f"Status:")
        # 阶段0：系统初始化和任务规划
        if stage == 0:
            if (path_planner.gps_guide_x != 0 or path_planner.gps_guide_y != 0) and len(path_planner.downtown_centers) >= 2 and (iris.x != 0 or iris.y != 0):
                
                path_planner.plan_iris_mission((iris.x, iris.y), iris.healthy_point, iris.critical_point)
                
                # 发送起飞命令
                for _ in range(100):
                    iris.send_target_command(0, 0, iris.takeoff_altitude)
                    time.sleep(0.1)
                
                time.sleep(3)
                iris.offboard()
                time.sleep(3)
            
                iris.arm()
                time.sleep(1)
                
                stage = 1
            
        # 阶段1：起飞   
        elif stage == 1:
            iris.send_target_command(0, 0, iris.takeoff_altitude)
            print(1, f"当前高度: {iris.z:.1f}m")
            if abs(iris.z - iris.takeoff_altitude) < 1:
                rospy.loginfo("起飞完成")
                iris.hover()
                time.sleep(2)
                iris.offboard()
                stage = 2
        
        # 阶段2：导航 
        elif stage == 2:
            if not stage2_initialized:
                stage2_initialized = True
                stage2_waypoint_index = 0
            if stage2_waypoint_index < len(path_planner.waypoints):
                target = path_planner.waypoints[stage2_waypoint_index]
                waypoint_name = "切点" if stage2_waypoint_index == 0 and len(path_planner.waypoints) > 1 else "GPS引导点"
                iris.send_target_command(target[0], target[1], iris.takeoff_altitude)
                if iris.is_waypoint_reached(target):
                    print(f"到达{waypoint_name}: ({target[0]:.1f}, {target[1]:.1f})")
                    stage2_waypoint_index += 1
                print(f"飞往{waypoint_name}: ({target[0]:.1f}, {target[1]:.1f}), 当前位置：({iris.x:.1f}, {iris.y:.1f})")
            else:
                rospy.loginfo("导航完成")
                stage = 3
          
        # 阶段3：目标跟踪阶段      
        elif stage == 3:
            if not stage3_initialized:
                stage3_initialized = True
                iris.camera.enable_detection()
                target_points = [("healthy", (iris.healthy_point[0], iris.healthy_point[1] + 5)), ("critical",iris.critical_point)]
                target_names = ["健康", "危重"]
                part = 0
                
                # 启动动态跟踪器节点
                status_pub.publish(String("start"))
                rospy.Subscriber("/tracker/target", Pose, tracker_target_callback, callback_args=(iris, status_pub))
                
                rospy.sleep(2)
            
            # 当前目标点
            current_target = target_points[part][1]
            current_target_name = target_points[part][0]
            print(f"前往{target_names[part]}人员位置：({current_target[0]:.1f}, {current_target[1]:.1f})")
            
            # 发送目标位置命令
            iris.send_target_command(*current_target, 15)
            
            # 检查是否到达目标位置
            if iris.is_waypoint_reached(current_target):
                print(f"到达{target_names[part]}人员位置，开始检测")
                # 发布任务状态给tracker节点
                status_pub.publish(String(current_target_name))
                last_part = part
                last_stage = stage
                while not rospy.is_shutdown():
                    if last_part != part or last_stage != stage:
                        break
                    rate.sleep()
            rate.sleep()

        # 阶段4：返航阶段
        elif stage == 4:
            if not stage4_initialized:
                status_pub.publish(String("completed"))
                stage4_initialized = True
                return_stage = 0
                landing_height = 5
                target_points = [path_planner.waypoints[1], path_planner.waypoints[0], (takeoff_x, takeoff_y)]
            current_target_point = target_points[return_stage]
            iris.send_target_command(*current_target_point, landing_height)
            if return_stage != 2:
                if iris.is_waypoint_reached(current_target_point):
                    return_stage += 1
            else:
                # 到达起飞点附近，开始降落
                if calculate_distance(current_target_point, (iris.x, iris.y)) < 50:
                    for _ in range(50):
                        iris.send_vel_command(0, 0, 0)
                        time.sleep(0.1)
                    time.sleep(5)
                    iris.hover()
                    time.sleep(2)
                    while iris.z > 0.1:
                        iris.land()
                        print(f"距离地面还有{iris.z:.1f}m")
                        rate.sleep()
                    time.sleep(2)
                    while iris.state.armed:
                        iris.disarm()
                        rate.sleep()
                    rospy.loginfo("IRIS任务完成")
                    status_pub.publish(String("completed"))
                    break
    rate.sleep()      

if __name__ == "__main__":
    rospy.init_node("iris_mission")
    start_thread = Thread(target=wait_for_vtol)
    start_thread.start()
    # 创建IRIS实例和路径规划器
    iris = Iris()
    time.sleep(1)
    path_planner = PathPlanner()
    # 等待VTOL任务完成
    start_thread.join()
    execute_iris_mission(iris, path_planner)
    rospy.signal_shutdown("任务完成")

