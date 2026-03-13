#!/bin/bash
# -*- coding: utf-8 -*-
echo "Start Mission Nodes..."
echo "VTOL -> Iris -> Dynamic Tracker"
echo "==========================================="

echo "启动VTOL任务..."
python3 $PWD/vtol_mission_node.py &
VTOL_PID=$!  # 保存VTOL进程ID
sleep 2

echo "启动IRIS任务..."
python3 $PWD/iris_mission_node.py & 
IRIS_PID=$!  # 保存IRIS进程ID
sleep 2

echo "启动Dynamic Tracker任务..."
python3 $PWD/dynamic_tracker_node.py &
TRACKER_PID=$!  # 保存Tracker进程ID

# 等待所有任务完成
wait $VTOL_PID $IRIS_PID $TRACKER_PID

echo "所有任务已完成" 
