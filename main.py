#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
水质预测系统主程序
协调数据处理、模型训练和评估、界面展示
"""

import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import subprocess

# 导入模块
import data_preprocessing
import exploratory_analysis
import model_training
from ui import WaterQualityPredictionUI
from PyQt5.QtWidgets import QApplication

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("log/main.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def create_dirs():
    """创建必要的目录结构"""
    dirs = [
        'data/raw',
        'data/processed',
        'results/figures',
        'results/metrics',
        'models/saved',
        'log'
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    logger.info("目录结构创建完成")


def run_data_pipeline():
    """运行数据处理流程"""
    logger.info("开始数据处理流程")

    # 数据预处理
    logger.info("1. 执行数据预处理")
    clean_data, split_data = data_preprocessing.main()

    # 探索性数据分析
    logger.info("2. 执行探索性数据分析")
    analysis_results = exploratory_analysis.main()

    logger.info("数据处理流程完成")
    return clean_data, split_data, analysis_results


def run_model_pipeline(clean_data, split_data, model_args=None):
    """
    运行模型训练和评估流程

    Args:
        clean_data: 清洗后的数据
        split_data: 分割后的数据
        model_args: 模型训练参数字典
    """
    logger.info("开始模型训练和评估流程")

    # 创建命令行参数字符串
    cmd_args = []
    if model_args:
        if 'models' in model_args and model_args['models']:
            cmd_args.extend(['--models'] + model_args['models'])
        if 'top_n_stations' in model_args:
            cmd_args.extend(['--top_n_stations', str(model_args['top_n_stations'])])
        if 'features' in model_args and model_args['features']:
            cmd_args.extend(['--features'] + model_args['features'])
        # 添加预测步长参数
        if 'forecast_horizon' in model_args:
            cmd_args.extend(['--forecast_horizon', str(model_args['forecast_horizon'])])

    # 模型训练和评估
    logger.info(f"执行模型训练，参数: {cmd_args}")

    if cmd_args:
        # 使用外部进程运行model_training.py以传递命令行参数
        cmd = [sys.executable, 'model_training.py'] + cmd_args
        logger.info(f"执行命令: {' '.join(cmd)}")
        try:
            # 修改这里：去掉stdout和stderr的重定向，让输出直接显示在终端
            process = subprocess.Popen(cmd)

            # 等待进程完成
            process.wait()

            if process.returncode != 0:
                logger.error(f"模型训练失败，返回码: {process.returncode}")
        except Exception as e:
            logger.error(f"执行模型训练失败: {str(e)}")
    else:
        # 直接调用模型训练函数
        model_results = model_training.main()

    logger.info("模型训练和评估流程完成")


def run_ui():
    """运行用户界面"""
    logger.info("启动用户界面")

    app = QApplication(sys.argv)

    # 创建并显示主窗口
    window = WaterQualityPredictionUI()
    window.show()

    sys.exit(app.exec_())


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="水质预测系统")
    parser.add_argument('--mode', type=str, default='all',
                      choices=['data', 'model', 'ui', 'all'],
                      help="运行模式: data(数据处理), model(模型训练), ui(用户界面), all(全部)")
    parser.add_argument('--no-process', action='store_true',
                      help="跳过数据处理步骤")
    parser.add_argument('--no-training', action='store_true',
                      help="跳过模型训练步骤")

    # 添加模型训练相关参数
    parser.add_argument('--models', nargs='+',
                      choices=['arima', 'sarima', 'arima_ann', 'cnn_lstm', 'cnn_gru_attention', 'xgboost', 'all'],
                      help="指定要训练的模型，可多选")
    parser.add_argument('--top_n_stations', type=int,
                      help="选择数据量最多的前N个站点进行训练")
    parser.add_argument('--features', nargs='+',
                      help="指定要分析的特征，默认使用所有特征")
    # 添加多步预测支持
    parser.add_argument('--forecast_horizon', type=int, default=5,
                      help="预测步长，预测未来多少个时间点，默认为5")

    args = parser.parse_args()

    # 创建目录结构
    create_dirs()

    clean_data = None
    split_data = None
    analysis_results = None

    # 根据运行模式执行不同流程
    if args.mode in ['data', 'all'] and not args.no_process:
        clean_data, split_data, analysis_results = run_data_pipeline()

    if args.mode in ['model', 'all'] and not args.no_training:
        # 收集模型训练参数
        model_args = {}
        if args.models:
            model_args['models'] = args.models
        if args.top_n_stations:
            model_args['top_n_stations'] = args.top_n_stations
        if args.features:
            model_args['features'] = args.features
        # 添加预测步长参数
        model_args['forecast_horizon'] = args.forecast_horizon

        if clean_data is None or split_data is None:
            # 如果没有执行数据处理，尝试加载已处理的数据
            try:
                logger.info("尝试加载已处理的数据")
                # 这里可以实现加载逻辑，或者直接调用model_training
                run_model_pipeline(None, None, model_args)
            except Exception as e:
                logger.error(f"加载已处理数据失败: {str(e)}")
                if not model_args:
                    logger.error("无法完成模型训练，请先执行数据处理流程")
                    return
        else:
            run_model_pipeline(clean_data, split_data, model_args)

    if args.mode in ['ui', 'all']:
        run_ui()

    logger.info("程序执行完成")


if __name__ == "__main__":
    main()