#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VTOL垂直起降无人机任务节点

该节点负责VTOL无人机的完整任务执行，包括：
1. 起飞和模式转换（垂直起飞到固定翼飞行）
2. 导航到目标区域
3. 沿Y轴扫描检测目标（重伤、轻伤、健康人员）
4. 返航和降落
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rospy
import time
from std_msgs.msg import String
from src.models.vtol import Vtol
from src.mission_modules.path_planner import PathPlanner
from src.utils import calculate_distance, create_pose_message, pixel_to_world_with_pose



def execute_vtol_mission(vtol: Vtol, path_planner: PathPlanner):
    """
    执行VTOL无人机任务
    
    控制VTOL无人机完成完整的救援任务流程
    
    Args:
        vtol: VTOL无人机实例
        path_planner: 路径规划器实例
    """
    stage = 0  # 当前任务阶段
    takeoff_x, takeoff_y = vtol.x, vtol.y
    stage2_initialized = False
    stage3_initialized = False
    stage4_initialized = False
    rate = rospy.Rate(20)
    stages = ["初始化", "起飞", "导航", "识别", "返航"]
    # 任务状态发布器
    status_pub = rospy.Publisher('/vtol_mission/status', String, queue_size=10)
    rospy.sleep(1)  # 等待发布器初始化  
    rospy.loginfo("VTOL任务开始")
    while not rospy.is_shutdown():
        print("**********VTOL Mission**********")
        print(f"Stage {stage}: {stages[stage]}")
        print(f"Status:")
        # 阶段0：系统初始化和任务规划
        if stage == 0:
            if (path_planner.gps_guide_x != 0 or path_planner.gps_guide_y != 0) and len(path_planner.downtown_centers) >= 2 and (vtol.x != 0 or vtol.y != 0):
                
                path_planner.plan_gps_mission(vtol.x, vtol.y)
                
                # 发送起飞命令
                for _ in range(50):
                    vtol.send_target_command(0, 0, vtol.takeoff_altitude)
                    time.sleep(0.1)
                    
                vtol.offboard()
                time.sleep(1)
                
                vtol.arm()
                time.sleep(1)
                
                stage = 1
        
        # 阶段1：起飞 
        elif stage == 1:
            vtol.send_target_command(0, 0, vtol.takeoff_altitude)
            print(f"当前高度: {vtol.z:.1f}m")
            if abs(vtol.z - vtol.takeoff_altitude) < 1:
                rospy.loginfo("起飞完成")
                vtol.transition()
                time.sleep(5)
                stage = 2
        
        # 阶段2：导航到目标区域        
        elif stage == 2:
            if not stage2_initialized:
                stage2_initialized = True
                current_waypoint_index = 0
                waypoints = path_planner.waypoints
                
            if current_waypoint_index < len(waypoints):
                target = waypoints[current_waypoint_index]
                waypoint_name = "切点" if current_waypoint_index == 0 and len(waypoints) > 1 else "GPS引导点"
                
                vtol.send_target_command(target[0], target[1], 20)
                
                if vtol.is_waypoint_reached(target):
                    print(f"到达{waypoint_name}: ({target[0]:.1f}, {target[1]:.1f})")
                    current_waypoint_index += 1
                print(f"飞往{waypoint_name}: ({target[0]:.1f}, {target[1]:.1f}), 当前位置：({vtol.x:.1f}, {vtol.y:.1f})")
                
            else:
                rospy.loginfo("导航完成")
                stage = 3  

        # 阶段3：定速巡航扫描
        elif stage == 3:
            if not stage3_initialized:
                stage3_initialized = True   
                # 定义扫描的关键点：三个航点用于规划扫描路径
                stage3_key_points = [
                    (path_planner.gps_guide_x - 45, path_planner.gps_guide_y - 170, 10), 
                    (path_planner.gps_guide_x, path_planner.gps_guide_y - 80, 8), 
                    (path_planner.gps_guide_x - 10, path_planner.gps_guide_y + 170, 8)]           
                stage3_substage = -1
                stage3_current_key_point = (path_planner.gps_guide_x + 20, path_planner.gps_guide_y + 110, 10)
                stage3_targets_published = {'critical': False, 'injured': False, 'healthy': False}
                detection_results = {'critical': None, 'injured': None, 'healthy': None}
                last_detection_point = {}
                last_detection_distance = {}
                
            # 检查是否完成所有目标检测
            if None not in detection_results.values():
                rospy.loginfo("已检测所有目标")
                rospy.loginfo(f"危重人员：{detection_results['critical']}")
                rospy.loginfo(f"轻伤人员：{detection_results['injured']}")
                rospy.loginfo(f"健康人员：{detection_results['healthy']}")
                stage = 4
                status_pub.publish(String("return"))
                continue
            
            # 检查是否到达扫描边界，切换扫描点
            if vtol.is_waypoint_reached(stage3_current_key_point):
                stage3_substage = (stage3_substage + 1) % 3
                stage3_current_key_point = stage3_key_points[stage3_substage]
                if stage3_substage != 0:
                    vtol.camera.enable_detection()
                    print("扫描检测开启")
                else:
                    vtol.camera.disable_detection()
                    print("扫描检测关闭")
                
            vtol.send_target_command(*stage3_current_key_point)
            
            print(f"正在飞向扫描航点：({stage3_current_key_point[0]:.1f}, {stage3_current_key_point[1]:.1f})，当前位置：({vtol.x:.1f}, {vtol.y:.1f})")
            # 在 VTOL 阶段三中
            if vtol.camera.current_detections:
                
                for class_name, center in vtol.camera.current_detections.items():
                    cx, cy = center
                    distance = ((cx - vtol.camera.shape[1] // 2) ** 2 + 
                                (cy - vtol.camera.shape[0] // 2) ** 2) ** 0.5
                    wx, wy = pixel_to_world_with_pose(cx, cy, vtol.camera.camera_matrix, vtol.pose, vtol.pose.position.z)
                                        
                    if class_name not in last_detection_point:
                        last_detection_point[class_name] = (wx, wy)
                        last_detection_distance[class_name] = distance
                        continue

                    prev_distance = last_detection_distance[class_name]
                    if distance < prev_distance:
                        last_detection_point[class_name] = (wx, wy)
                        last_detection_distance[class_name] = distance
                    else:
                        if wx != float('inf') and wy != float('inf') and detection_results[class_name] is None:
                            detection_results[class_name] = (last_detection_point[class_name][0], last_detection_point[class_name][1], 0.0)

                            pose_msg = create_pose_message((last_detection_point[class_name][0], last_detection_point[class_name][1], 0))
                            if class_name == 'critical':
                                vtol.critical_pub.publish(pose_msg)
                            elif class_name == 'injured':
                                vtol.injured_pub.publish(pose_msg)
                            elif class_name == 'healthy':
                                vtol.healthy_pub.publish(pose_msg)

                        last_detection_point[class_name] = (wx, wy)
                        last_detection_distance[class_name] = distance


        # 阶段4：返航阶段        
        elif stage == 4:
            if not stage4_initialized:
                stage4_initialized = True
                return_stage = 0
                landing_height = 10
                target_points = [path_planner.waypoints[1], path_planner.waypoints[0], (takeoff_x, takeoff_y)]
            current_target_point = target_points[return_stage]
            vtol.send_target_command(*current_target_point, landing_height)
            if return_stage != 2:
                if vtol.is_waypoint_reached(current_target_point):
                    return_stage += 1
            else:
                # 到达起飞点附近，开始降落
                if calculate_distance(current_target_point, (vtol.x, vtol.y)) < 80:
                    vtol.hover()
                    time.sleep(2)
                    while vtol.z > 0.1:
                        vtol.land()
                        print(f"距离地面还有{vtol.z:.1f}m")
                        rate.sleep()
                    time.sleep(2)
                    while vtol.state.armed:
                        vtol.disarm()
                        rate.sleep()
                    rospy.loginfo("VTOL任务完成")
                    status_pub.publish(String("completed"))
                    break
        rate.sleep()
        

if __name__ == "__main__":
    rospy.init_node("vtol_mission")
    vtol = Vtol()
    time.sleep(1)
    path_planner = PathPlanner()
    time.sleep(1)    
    execute_vtol_mission(vtol, path_planner)

