#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
大语言模型接口模块

该模块提供了与大语言模型的接口，包括：
1. 图像理解：基于视觉的LLM调用
2. 文本理解：基于文本的LLM调用
3. 路径规划：使用LLM进行智能路径规划
"""

import json
import re

import requests

def image_llm(data_base64):
    """
    图像理解LLM接口
    
    使用大语言模型分析图像内容，返回图像描述
    
    Args:
        data_base64: Base64编码的图像数据
        
    Returns:
        str: 图像描述文本，出错时返回错误信息
    """
    headers = {
        "Authorization": "Bearer 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",
        "Content-Type": "application/json",
    }

    payload = {
        "model": "meta-llama/llama-3.2-11b-vision-instruct:free",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Please describe what you see in the given picture."},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{data_base64}"
                        }
                    }
                ]
            }
        ]
    }

    try:
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            data=json.dumps(payload)
        )
        response.raise_for_status()
        result = response.json()
        
        content = result["choices"][0]["message"]["content"]
        return content

    except Exception as e:
        return f"调用大模型出错: {str(e)}"

def text_llm(**kwargs):
    """
    文本理解LLM接口
    
    使用大语言模型进行路径规划，根据任务需求生成最优路径
    
    Args:
        **kwargs: 关键字参数，包含：
            - takeoff_point: 起飞点坐标
            - downtown_point: 居民区（禁飞区）坐标
            - gps_point: GPS引导点坐标
            - healthy_point: 健康人员位置
            - critical_point: 重伤人员位置
            
    Returns:
        str: 路径规划结果，格式为坐标列表字符串
    """
    takeoff_point = kwargs.get("takeoff_point")
    downtown_point = kwargs.get("downtown_point")
    gps_point = kwargs.get("gps_point")
    healthy_point = kwargs.get("healthy_point")
    critical_point = kwargs.get("critical_point")
    
    # 构建路径规划提示词
    prompt = f"""
    背景：在一个二维坐标系上有圆心{downtown_point}、半径250的圆形禁飞区，一架多旋翼无人机需从{takeoff_point}出发避开圆形禁飞区前往{gps_point}，最终到达{healthy_point}。
    任务：请给出一个最优方案，在出发点和GPS引导点之间的飞行阶段设置1个航点使无人机飞行路径最短。
    返回格式：[(x1, y1)]。
    注意：不要返回任何其他推理过程。
    """
     
    # 调用LLM接口
    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": "Bearer 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",
            "Content-Type": "application/json",
        },
        data=json.dumps({
            "model": "deepseek/deepseek-r1-0528-qwen3-8b:free",
            "messages": [
            {
                "role": "user",
                "content": prompt
            }
            ],
        })
    ).json()
    content = response["choices"][0]["message"]["content"]
    
    # 使用正则表达式提取坐标
    pattern = re.compile(r"\[(.*?)\]")
    match = pattern.search(content)
    if match:
        path = match.group(1)
        return path
    else:
        return None
    
if __name__ == "__main__":
    # 测试函数
    print(text_llm(takeoff_point=(2.5, 2.7), downtown_point=(1200, 0), gps_point=(1500, 0)))