# 基于多模态智枢协同算法的双机自主救援系统 

## 作品简介

该软件包是一个基于ROS的无人机协同救援系统，包含VTOL垂直起降无人机、IRIS多旋翼无人机和动态目标跟踪器，用于协同执行复杂的地面目标检测和救援任务。

## 作品结构

```
ccnu_pkgs/
├── scripts/                           # 主要任务节点
│   ├── vtol_mission_node.py           # VTOL无人机任务节点
│   ├── iris_mission_node.py           # IRIS无人机任务节点
│   ├── dynamic_tracker_node.py        # 动态跟踪器节点
│   └── start_mission.sh               # 任务启动脚本
└── src/                               # 核心功能模块
    ├── models/                        # 无人机模型类
    │   ├── __init__.py                # 模型包初始化
    │   ├── uav.py                     # 基础无人机类
    │   ├── vtol.py                    # VTOL无人机控制
    │   ├── iris.py                    # IRIS无人机控制
    │   └── llm.py                     # 大语言模型接口
    ├── mission_modules/               # 任务执行模块
    │   ├── __init__.py                # 任务模块包初始化
    │   ├── path_planner.py            # 路径规划器
    │   └── delta_pid.py               # PID控制器
    ├── vision_modules/                # 视觉处理模块
    │   ├── __init__.py                # 视觉模块包初始化
    │   ├── camera.py                  # 相机控制
    │   ├── detector.py                # 基础检测器
    │   ├── yolo_detector.py           # YOLO目标检测
    │   ├── llm_detector.py            # LLM目标检测
    │   └── best.pt                    # YOLO模型权重文件
    └── utils/                         # 工具函数模块
        ├── __init__.py                # 工具包初始化
        ├── geometry_utils.py          # 几何计算工具
        ├── image_utils.py             # 图像处理工具
        └── ros_utils.py               # ROS通信工具
```

## 使用说明

### 方法一：启动任务脚本
```bash
cd src/ccnu_pkgs/scripts
chmod +x start_mission.sh
./start_mission.sh
```

### 方法二：依次启动节点
```bash
# 启动VTOL任务
python3 vtol_mission_node.py

# 启动IRIS任务
python3 iris_mission_node.py

# 启动动态跟踪器
python3 dynamic_tracker_node.py
```

## 注意事项

- 本作品垂起无人机的巡航速度参数FW_AIRSPD_TRIM需在QGC中设置为12m/s
- 本作品使用世界环境文件为删减多余树木和房屋后的zhihang2025.world
