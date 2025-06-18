#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模型训练主程序
用于训练和评估各种时间序列预测模型，包括：
1. ARIMA/SARIMA
2. ARIMA-ANN
3. CNN-LSTM
4. CNN-GRU-Attention
5. XGBoost
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns
from tqdm import tqdm
import warnings
import logging
import glob
import argparse
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from config import get_english_feature_name, get_english_station_name
import joblib

# 导入模型模块
from models.arima_model import ARIMAModel
from models.arima_ann_model import ARIMANNModel
from models.cnn_lstm_model import CNNLSTMModel
from models.cnn_gru_attention_model import CNNGRUAttentionModel
from models.xgboost_model import XGBoostModel

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("log/model_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 忽略警告
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
# plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']
# plt.rcParams['axes.unicode_minus'] = False

# 定义常量
PROCESSED_DATA_DIR = 'data/processed/'
FIGURES_DIR = 'results/figures/'
METRICS_DIR = 'results/metrics/'
MODELS_SAVE_DIR = 'models/saved/'
FEATURES_TO_ANALYZE = [
    '水温(℃)', 'pH(无量纲)', '溶解氧(mg/L)', '电导率(μS/cm)',
    '浊度(NTU)', '高锰酸盐指数(mg/L)', '氨氮(mg/L)',
    '总磷(mg/L)', '总氮(mg/L)'
]

# 定义可用模型列表
AVAILABLE_MODELS = [
    'arima',
    'sarima',
    'arima_ann',
    'cnn_lstm',
    'cnn_gru_attention',
    'xgboost',
    'all'
]


def create_dirs():
    """创建必要的目录结构"""
    dirs = [
        PROCESSED_DATA_DIR,
        FIGURES_DIR,
        METRICS_DIR,
        MODELS_SAVE_DIR,
        'log'
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    logger.info("目录结构创建完成")


def load_split_data():
    """
    加载已拆分的训练、验证和测试数据

    Returns:
        dict: 包含训练集、验证集和测试集的字典
    """
    logger.info("加载已拆分的数据")

    split_data = {
        'train': {},
        'val': {},
        'test': {}
    }

    # 加载各集合数据
    for split_type in split_data.keys():
        split_dir = os.path.join(PROCESSED_DATA_DIR, split_type)

        # 检查目录是否存在
        if not os.path.exists(split_dir):
            logger.error(f"目录不存在: {split_dir}")
            continue

        # 获取目录中的所有CSV文件
        csv_files = glob.glob(os.path.join(split_dir, "*.csv"))
        logger.info(f"在{split_type}集目录中找到 {len(csv_files)} 个站点数据文件")

        # 加载每个站点的数据
        for csv_file in tqdm(csv_files, desc=f"加载{split_type}集"):
            # 从文件名获取站点名称
            station_filename = os.path.basename(csv_file)
            station_name = os.path.splitext(station_filename)[0].replace('_', '/')

            # 加载数据
            try:
                df = pd.read_csv(csv_file, encoding='utf-8')
                # 如果有监测时间列，将其设为索引
                if '监测时间' in df.columns:
                    df['监测时间'] = pd.to_datetime(df['监测时间'])
                    df.set_index('监测时间', inplace=True)

                # 存储数据
                split_data[split_type][station_name] = df
            except Exception as e:
                logger.error(f"加载文件失败 {csv_file}: {str(e)}")

    # 统计数据量
    for split_type, stations_data in split_data.items():
        logger.info(f"{split_type}集包含 {len(stations_data)} 个站点")

    return split_data


def train_arima_models(split_data, features=None, top_n_stations=5, use_sarima=False):
    """
    训练ARIMA或SARIMA模型

    Args:
        split_data: 包含训练、验证和测试数据的字典
        features: 要分析的特征列表，默认为None时使用FEATURES_TO_ANALYZE
        top_n_stations: 选择数据量最多的前N个站点进行训练，默认为5
        use_sarima: 是否使用季节性SARIMA模型，默认为False

    Returns:
        dict: 训练好的模型结果
    """
    if features is None:
        features = FEATURES_TO_ANALYZE

    logger.info(f"开始训练{'SARIMA' if use_sarima else 'ARIMA'}模型")

    # 选择数据量最多的前N个站点
    station_data_counts = {station: len(data) for station, data in split_data['train'].items()}
    sorted_stations = sorted(station_data_counts.items(), key=lambda x: x[1], reverse=True)
    top_stations = [station for station, _ in sorted_stations[:top_n_stations]]

    logger.info(f"选择了数据量最多的 {len(top_stations)} 个站点: {top_stations}")

    # 创建结果存储字典
    model_results = {}
    all_metrics = []

    # 对每个站点和特征训练模型
    for station in tqdm(top_stations, desc="站点进度"):
        station_results = {}

        # 获取该站点的训练、验证和测试数据
        train_data = split_data['train'].get(station)
        val_data = split_data['val'].get(station)
        test_data = split_data['test'].get(station)

        if train_data is None or val_data is None or test_data is None:
            logger.warning(f"站点 {station} 缺少训练、验证或测试数据，跳过")
            continue

        # 对每个特征训练模型
        for feature in tqdm(features, desc=f"站点 {station} 特征进度", leave=False):
            if feature not in train_data.columns:
                logger.warning(f"站点 {station} 缺少特征 {feature}，跳过")
                continue

            try:
                # 提取特征数据
                train_feature = train_data[feature].dropna()
                val_feature = val_data[feature].dropna()
                test_feature = test_data[feature].dropna()

                # 如果数据点太少，跳过
                if len(train_feature) < 10 or len(val_feature) < 5 or len(test_feature) < 5:
                    logger.warning(f"站点 {station} 特征 {feature} 数据点不足，跳过")
                    continue

                # 创建模型实例
                model = ARIMAModel(seasonal=use_sarima)

                # 查找最佳参数 (使用训练集和验证集数据)
                combined_train_val = pd.concat([train_feature, val_feature])
                best_params = model.auto_arima(
                    train_feature,
                    val_feature,
                    feature_name=feature,
                    station_name=station
                )

                # 使用最佳参数和所有训练数据重新训练模型
                model.fit(combined_train_val, feature_name=feature, station_name=station, params=best_params)

                # 进行预测
                forecast_mean, forecast_ci = model.predict(test_feature)

                # 绘制结果
                safe_station = station.replace('/', '_').replace('\\', '_').replace(':', '_')
                safe_feature = feature.replace('/', '_').replace('\\', '_').replace(':', '_')
                model_type = 'sarima' if use_sarima else 'arima'
                plot_path = os.path.join(FIGURES_DIR, f'{model_type}_{safe_station}_{safe_feature}_forecast.png')

                model.plot_results(combined_train_val, test_feature, forecast_mean, forecast_ci, save_path=plot_path)

                # 保存模型
                model.save_model(save_dir=MODELS_SAVE_DIR)

                # 存储结果
                metrics = model.metrics.copy()
                metrics.update({
                    'station': station,
                    'feature': feature,
                    'model_type': 'SARIMA' if use_sarima else 'ARIMA',
                    'params': str(best_params)
                })
                all_metrics.append(metrics)

                station_results[feature] = {
                    'model': model,
                    'forecast_mean': forecast_mean,
                    'forecast_ci': forecast_ci,
                    'metrics': metrics
                }

                logger.info(f"站点 {station} 特征 {feature} {'SARIMA' if use_sarima else 'ARIMA'}模型训练完成")

            except Exception as e:
                logger.error(f"站点 {station} 特征 {feature} 模型训练失败: {str(e)}")

        # 存储该站点的所有特征模型结果
        model_results[station] = station_results

    # 保存所有指标
    metrics_df = pd.DataFrame(all_metrics)
    metrics_file = os.path.join(METRICS_DIR, f"{'sarima' if use_sarima else 'arima'}_metrics.csv")
    metrics_df.to_csv(metrics_file, index=False, encoding='utf-8')
    logger.info(f"模型评估指标已保存至: {metrics_file}")

    return model_results


def train_arima_ann_models(split_data, features=None, top_n_stations=5,
                          hidden_layers=[128, 64, 32], dropout_rate=0.3,
                          learning_rate=0.001, batch_size=16, epochs=200, patience=30):
    """
    训练ARIMA-ANN混合模型

    Args:
        split_data: 包含训练、验证和测试数据的字典
        features: 要分析的特征列表，默认为None时使用FEATURES_TO_ANALYZE
        top_n_stations: 选择数据量最多的前N个站点进行训练，默认为5
        hidden_layers: 神经网络隐藏层单元列表
        dropout_rate: Dropout层的丢弃率
        learning_rate: 学习率
        batch_size: 批处理大小
        epochs: 训练轮数
        patience: 早停策略的耐心值

    Returns:
        dict: 训练好的模型结果
    """
    if features is None:
        features = FEATURES_TO_ANALYZE

    logger.info("开始训练ARIMA-ANN混合模型")

    # 选择数据量最多的前N个站点
    station_data_counts = {station: len(data) for station, data in split_data['train'].items()}
    sorted_stations = sorted(station_data_counts.items(), key=lambda x: x[1], reverse=True)
    top_stations = [station for station, _ in sorted_stations[:top_n_stations]]

    logger.info(f"选择了数据量最多的 {len(top_stations)} 个站点: {top_stations}")

    # 创建结果存储字典
    model_results = {}
    all_metrics = []

    # 对每个站点和特征训练模型
    for station in tqdm(top_stations, desc="站点进度"):
        station_results = {}

        # 获取该站点的训练、验证和测试数据
        train_data = split_data['train'].get(station)
        val_data = split_data['val'].get(station)
        test_data = split_data['test'].get(station)

        if train_data is None or val_data is None or test_data is None:
            logger.warning(f"站点 {station} 缺少训练、验证或测试数据，跳过")
            continue

        # 对每个特征训练模型
        for feature in tqdm(features, desc=f"站点 {station} 特征进度", leave=False):
            if feature not in train_data.columns:
                logger.warning(f"站点 {station} 缺少特征 {feature}，跳过")
                continue

            try:
                # 提取特征数据
                train_feature = train_data[feature].dropna()
                val_feature = val_data[feature].dropna()
                test_feature = test_data[feature].dropna()

                # 如果数据点太少，跳过
                if len(train_feature) < 20 or len(val_feature) < 10 or len(test_feature) < 10:
                    logger.warning(f"站点 {station} 特征 {feature} 数据点不足，跳过")
                    continue

                # 创建模型实例，使用自定义参数
                model = ARIMANNModel(
                    arima_order=(2, 1, 2),  # 默认值，会被auto_arima覆盖
                    hidden_layers=hidden_layers,
                    dropout_rate=dropout_rate,
                    learning_rate=learning_rate,
                    batch_size=batch_size,
                    epochs=epochs,
                    patience=patience
                )

                # 训练模型
                model.fit(
                    train_feature,
                    val_feature,
                    feature_name=feature,
                    station_name=station,
                    auto_arima=True
                )

                # 进行预测
                predictions, arima_pred, nn_pred = model.predict(test_feature)

                # 绘制结果
                safe_station = station.replace('/', '_').replace('\\', '_').replace(':', '_')
                safe_feature = feature.replace('/', '_').replace('\\', '_').replace(':', '_')
                plot_path = os.path.join(FIGURES_DIR, f'arima_ann_{safe_station}_{safe_feature}_forecast.png')

                # 绘制结果（使用concatenated训练和验证数据）
                combined_train_val = pd.concat([train_feature, val_feature])
                model.plot_results(combined_train_val, test_feature, predictions, arima_pred, nn_pred,
                                   save_path=plot_path)

                # 绘制训练历史
                history_plot_path = os.path.join(FIGURES_DIR, f'arima_ann_{safe_station}_{safe_feature}_history.png')
                model.plot_training_history(save_path=history_plot_path)

                # 保存模型
                model.save_model(save_dir=MODELS_SAVE_DIR)

                # 存储结果
                metrics = model.metrics.copy()
                metrics.update({
                    'station': station,
                    'feature': feature,
                    'model_type': 'ARIMA-ANN'
                })
                all_metrics.append(metrics)

                station_results[feature] = {
                    'model': model,
                    'predictions': predictions,
                    'arima_pred': arima_pred,
                    'nn_pred': nn_pred,
                    'metrics': metrics
                }

                logger.info(f"站点 {station} 特征 {feature} ARIMA-ANN模型训练完成")

            except Exception as e:
                logger.error(f"站点 {station} 特征 {feature} 模型训练失败: {str(e)}")

        # 存储该站点的所有特征模型结果
        model_results[station] = station_results

    # 保存所有指标
    metrics_df = pd.DataFrame(all_metrics)
    metrics_file = os.path.join(METRICS_DIR, "arima_ann_metrics.csv")
    metrics_df.to_csv(metrics_file, index=False, encoding='utf-8')
    logger.info(f"模型评估指标已保存至: {metrics_file}")

    return model_results


def train_cnn_lstm_models(split_data, features=None, top_n_stations=5):
    """
    训练CNN-LSTM模型

    Args:
        split_data: 包含训练、验证和测试数据的字典
        features: 要分析的特征列表，默认为None时使用FEATURES_TO_ANALYZE
        top_n_stations: 选择数据量最多的前N个站点进行训练，默认为5

    Returns:
        dict: 训练好的模型结果
    """
    if features is None:
        features = FEATURES_TO_ANALYZE

    logger.info("开始训练CNN-LSTM模型")

    # 选择数据量最多的前N个站点
    station_data_counts = {station: len(data) for station, data in split_data['train'].items()}
    sorted_stations = sorted(station_data_counts.items(), key=lambda x: x[1], reverse=True)
    top_stations = [station for station, _ in sorted_stations[:top_n_stations]]

    logger.info(f"选择了数据量最多的 {len(top_stations)} 个站点: {top_stations}")

    # 创建结果存储字典
    model_results = {}
    all_metrics = []

    # 对每个站点和特征训练模型
    for station in tqdm(top_stations, desc="站点进度"):
        station_results = {}

        # 获取该站点的训练、验证和测试数据
        train_data = split_data['train'].get(station)
        val_data = split_data['val'].get(station)
        test_data = split_data['test'].get(station)

        if train_data is None or val_data is None or test_data is None:
            logger.warning(f"站点 {station} 缺少训练、验证或测试数据，跳过")
            continue

        # 对每个特征训练模型
        for feature in tqdm(features, desc=f"站点 {station} 特征进度", leave=False):
            if feature not in train_data.columns:
                logger.warning(f"站点 {station} 缺少特征 {feature}，跳过")
                continue

            try:
                # 提取特征数据
                train_feature = train_data[feature].dropna()
                val_feature = val_data[feature].dropna()
                test_feature = test_data[feature].dropna()

                # 如果数据点太少，跳过
                if len(train_feature) < 30 or len(val_feature) < 10 or len(test_feature) < 10:
                    logger.warning(f"站点 {station} 特征 {feature} 数据点不足，跳过")
                    continue

                # 创建模型实例
                model = CNNLSTMModel(
                    seq_length=12,  # 使用过去12个时间点进行预测
                    lstm_units=50,
                    dropout_rate=0.2,
                    learning_rate=0.001
                )

                # 训练模型
                model.fit(
                    train_feature,
                    val_feature,
                    feature_name=feature,
                    station_name=station
                )

                # 进行预测
                predictions, actual = model.predict(test_feature)

                # 绘制结果
                safe_station = station.replace('/', '_').replace('\\', '_').replace(':', '_')
                safe_feature = feature.replace('/', '_').replace('\\', '_').replace(':', '_')
                plot_path = os.path.join(FIGURES_DIR, f'cnn_lstm_{safe_station}_{safe_feature}_forecast.png')

                model.plot_results(train_feature, test_feature, predictions, save_path=plot_path)

                # 绘制训练历史
                history_plot_path = os.path.join(FIGURES_DIR, f'cnn_lstm_{safe_station}_{safe_feature}_history.png')
                model.plot_training_history(save_path=history_plot_path)

                # 保存模型
                model.save_model(save_dir=MODELS_SAVE_DIR)

                # 存储结果
                metrics = model.metrics.copy()
                metrics.update({
                    'station': station,
                    'feature': feature,
                    'model_type': 'CNN-LSTM'
                })
                all_metrics.append(metrics)

                station_results[feature] = {
                    'model': model,
                    'predictions': predictions,
                    'actual': actual,
                    'metrics': metrics
                }

                logger.info(f"站点 {station} 特征 {feature} CNN-LSTM模型训练完成")

            except Exception as e:
                logger.error(f"站点 {station} 特征 {feature} 模型训练失败: {str(e)}")

        # 存储该站点的所有特征模型结果
        model_results[station] = station_results

    # 保存所有指标
    metrics_df = pd.DataFrame(all_metrics)
    metrics_file = os.path.join(METRICS_DIR, "cnn_lstm_metrics.csv")
    metrics_df.to_csv(metrics_file, index=False, encoding='utf-8')
    logger.info(f"模型评估指标已保存至: {metrics_file}")

    return model_results


def train_cnn_gru_attention_models(split_data, features=None, top_n_stations=5, forecast_horizon=5):
    """
    训练改进的CNN-GRU-Attention模型（支持多步预测）

    Args:
        split_data: 包含训练、验证和测试数据的字典
        features: 要分析的特征列表，默认为None时使用FEATURES_TO_ANALYZE
        top_n_stations: 选择数据量最多的前N个站点进行训练，默认为5
        forecast_horizon: 预测步长，预测未来多少个时间点

    Returns:
        dict: 训练好的模型结果
    """
    if features is None:
        features = FEATURES_TO_ANALYZE

    logger.info(f"开始训练改进的CNN-GRU-Attention模型，预测步长: {forecast_horizon}")

    # 选择数据量最多的前N个站点
    station_data_counts = {station: len(data) for station, data in split_data['train'].items()}
    sorted_stations = sorted(station_data_counts.items(), key=lambda x: x[1], reverse=True)
    top_stations = [station for station, _ in sorted_stations[:top_n_stations]]

    logger.info(f"选择了数据量最多的 {len(top_stations)} 个站点: {top_stations}")

    # 创建结果存储字典
    model_results = {}
    all_metrics = []

    # 对每个站点和特征训练模型
    for station in tqdm(top_stations, desc="站点进度"):
        station_results = {}

        # 获取该站点的训练、验证和测试数据
        train_data = split_data['train'].get(station)
        val_data = split_data['val'].get(station)
        test_data = split_data['test'].get(station)

        if train_data is None or val_data is None or test_data is None:
            logger.warning(f"站点 {station} 缺少训练、验证或测试数据，跳过")
            continue

        # 对每个特征训练模型
        for feature in tqdm(features, desc=f"站点 {station} 特征进度", leave=False):
            if feature not in train_data.columns:
                logger.warning(f"站点 {station} 缺少特征 {feature}，跳过")
                continue

            try:
                # 提取特征数据
                train_feature = train_data[feature].dropna()
                val_feature = val_data[feature].dropna()
                test_feature = test_data[feature].dropna()

                # 如果数据点太少，跳过
                # 注意：现在需要更多数据点来支持多步预测
                min_required_points = 12 + forecast_horizon  # 序列长度 + 预测步长
                if len(train_feature) < min_required_points * 2 or len(val_feature) < min_required_points or len(
                        test_feature) < min_required_points:
                    logger.warning(f"站点 {station} 特征 {feature} 数据点不足，跳过")
                    continue

                # 创建模型实例
                model = CNNGRUAttentionModel(
                    seq_length=12,  # 使用过去12个时间点进行预测
                    forecast_horizon=forecast_horizon,  # 预测未来forecast_horizon个时间点
                    gru_units=50,
                    attention_heads=4,
                    dropout_rate=0.2,
                    learning_rate=0.001
                )

                # 训练模型
                model.fit(
                    train_feature,
                    val_feature,
                    feature_name=feature,
                    station_name=station
                )

                # 进行预测
                predictions, actual = model.predict(test_feature)

                # 绘制结果
                safe_station = station.replace('/', '_').replace('\\', '_').replace(':', '_')
                safe_feature = feature.replace('/', '_').replace('\\', '_').replace(':', '_')
                plot_path = os.path.join(FIGURES_DIR, f'cnn_gru_att_{safe_station}_{safe_feature}_forecast.png')

                model.plot_results(train_feature, test_feature, predictions, save_path=plot_path)

                # 绘制训练历史
                history_plot_path = os.path.join(FIGURES_DIR, f'cnn_gru_att_{safe_station}_{safe_feature}_history.png')
                model.plot_training_history(save_path=history_plot_path)

                # 保存模型
                model.save_model(save_dir=MODELS_SAVE_DIR)

                # 存储结果和评估指标
                if model.metrics:
                    # 准备所有步长的评估指标
                    for step_metric in model.metrics.get('step_metrics', []):
                        step_metrics = step_metric.copy()
                        step_metrics.update({
                            'station': station,
                            'feature': feature,
                            'model_type': f'CNN-GRU-Attention-Step{step_metric["step"]}'
                        })
                        all_metrics.append(step_metrics)

                    # 添加平均指标
                    avg_metrics = {
                        'station': station,
                        'feature': feature,
                        'model_type': 'CNN-GRU-Attention-Avg',
                        'RMSE': model.metrics['avg_RMSE'],
                        'MAE': model.metrics['avg_MAE'],
                        'MAPE': model.metrics['avg_MAPE'],
                        'R2': model.metrics['avg_R2']
                    }
                    all_metrics.append(avg_metrics)

                station_results[feature] = {
                    'model': model,
                    'predictions': predictions,
                    'actual': actual,
                    'metrics': model.metrics
                }

                logger.info(f"站点 {station} 特征 {feature} CNN-GRU-Attention模型训练完成")

            except Exception as e:
                logger.error(f"站点 {station} 特征 {feature} 模型训练失败: {str(e)}")
                logger.exception(e)

        # 存储该站点的所有特征模型结果
        model_results[station] = station_results

    # 保存所有指标
    metrics_df = pd.DataFrame(all_metrics)
    metrics_file = os.path.join(METRICS_DIR, "cnn_gru_att_metrics.csv")
    metrics_df.to_csv(metrics_file, index=False, encoding='utf-8')
    logger.info(f"模型评估指标已保存至: {metrics_file}")

    return model_results


def train_xgboost_models(split_data, features=None, top_n_stations=5):
    """
    训练XGBoost模型

    Args:
        split_data: 包含训练、验证和测试数据的字典
        features: 要分析的特征列表，默认为None时使用FEATURES_TO_ANALYZE
        top_n_stations: 选择数据量最多的前N个站点进行训练，默认为5

    Returns:
        dict: 训练好的模型结果
    """
    if features is None:
        features = FEATURES_TO_ANALYZE

    logger.info("开始训练XGBoost模型")

    # 选择数据量最多的前N个站点
    station_data_counts = {station: len(data) for station, data in split_data['train'].items()}
    sorted_stations = sorted(station_data_counts.items(), key=lambda x: x[1], reverse=True)
    top_stations = [station for station, _ in sorted_stations[:top_n_stations]]

    logger.info(f"选择了数据量最多的 {len(top_stations)} 个站点: {top_stations}")

    # 创建结果存储字典
    model_results = {}
    all_metrics = []

    # 对每个站点和特征训练模型
    for station in tqdm(top_stations, desc="站点进度"):
        station_results = {}

        # 获取该站点的训练、验证和测试数据
        train_data = split_data['train'].get(station)
        val_data = split_data['val'].get(station)
        test_data = split_data['test'].get(station)

        if train_data is None or val_data is None or test_data is None:
            logger.warning(f"站点 {station} 缺少训练、验证或测试数据，跳过")
            continue

        # 对每个特征训练模型
        for feature in tqdm(features, desc=f"站点 {station} 特征进度", leave=False):
            if feature not in train_data.columns:
                logger.warning(f"站点 {station} 缺少特征 {feature}，跳过")
                continue

            try:
                # 提取特征数据
                train_feature = train_data[feature].dropna()
                val_feature = val_data[feature].dropna()
                test_feature = test_data[feature].dropna()

                # 如果数据点太少，跳过
                if len(train_feature) < 20 or len(val_feature) < 5 or len(test_feature) < 5:
                    logger.warning(f"站点 {station} 特征 {feature} 数据点不足，跳过")
                    continue

                # 创建模型实例
                model = XGBoostModel(
                    max_depth=5,
                    n_estimators=100,
                    learning_rate=0.1,
                    auto_tune=True
                )

                # 训练模型
                model.fit(
                    train_feature,
                    val_feature,
                    feature_name=feature,
                    station_name=station,
                    lookback=12
                )

                # 进行预测
                predictions = model.predict(test_feature, lookback=12)

                # 绘制结果
                safe_station = station.replace('/', '_').replace('\\', '_').replace(':', '_')
                safe_feature = feature.replace('/', '_').replace('\\', '_').replace(':', '_')
                plot_path = os.path.join(FIGURES_DIR, f'xgboost_{safe_station}_{safe_feature}_forecast.png')

                model.plot_results(train_feature, test_feature, predictions, save_path=plot_path)

                # 绘制特征重要性
                feature_imp_path = os.path.join(FIGURES_DIR,
                                                f'xgboost_{safe_station}_{safe_feature}_feature_importance.png')
                model.plot_feature_importance(save_path=feature_imp_path)

                # 保存模型
                model.save_model(save_dir=MODELS_SAVE_DIR)

                # 存储结果
                metrics = model.metrics.copy()
                metrics.update({
                    'station': station,
                    'feature': feature,
                    'model_type': 'XGBoost'
                })
                all_metrics.append(metrics)

                station_results[feature] = {
                    'model': model,
                    'predictions': predictions,
                    'metrics': metrics
                }

                logger.info(f"站点 {station} 特征 {feature} XGBoost模型训练完成")

            except Exception as e:
                logger.error(f"站点 {station} 特征 {feature} 模型训练失败: {str(e)}")

        # 存储该站点的所有特征模型结果
        model_results[station] = station_results

    # 保存所有指标
    metrics_df = pd.DataFrame(all_metrics)
    metrics_file = os.path.join(METRICS_DIR, "xgboost_metrics.csv")
    metrics_df.to_csv(metrics_file, index=False, encoding='utf-8')
    logger.info(f"模型评估指标已保存至: {metrics_file}")

    return model_results


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="时间序列预测模型训练程序")

    # 添加模型选择参数
    parser.add_argument('--models', nargs='+', default=['all'], choices=AVAILABLE_MODELS,
                        help="指定要训练的模型，可选值: " + ", ".join(AVAILABLE_MODELS))

    # 添加站点数量参数
    parser.add_argument('--top_n_stations', type=int, default=3,
                        help="选择数据量最多的前N个站点进行训练，默认为3")

    # 添加特征选择参数
    parser.add_argument('--features', nargs='+', default=None,
                        help="指定要分析的特征，默认使用所有特征")

    # 添加预测步长参数
    parser.add_argument('--forecast_horizon', type=int, default=5,
                        help="预测步长，预测未来多少个时间点，默认为5")

    return parser.parse_args()


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()

    logger.info("开始模型训练")
    logger.info(f"选择的模型: {args.models}")

    # 创建目录结构
    create_dirs()

    # 加载拆分数据
    split_data = load_split_data()

    # 确定要分析的特征
    features = args.features or FEATURES_TO_ANALYZE
    logger.info(f"分析的特征: {features}")

    # 获取预测步长
    forecast_horizon = args.forecast_horizon or 5
    logger.info(f"预测步长: {forecast_horizon}")

    # 训练选定的模型
    results = {}

    # 如果选择了'all'，则训练所有模型
    if 'all' in args.models:
        models_to_train = [m for m in AVAILABLE_MODELS if m != 'all']
    else:
        models_to_train = args.models

    # 训练模型
    for model in models_to_train:
        try:
            if model == 'arima':
                # 训练ARIMA模型
                logger.info("开始训练ARIMA模型")
                results['arima'] = train_arima_models(
                    split_data,
                    features=features,
                    top_n_stations=args.top_n_stations,
                    use_sarima=False
                )
            elif model == 'sarima':
                # 训练SARIMA模型
                logger.info("开始训练SARIMA模型")
                results['sarima'] = train_arima_models(
                    split_data,
                    features=features,
                    top_n_stations=args.top_n_stations,
                    use_sarima=True
                )
            elif model == 'arima_ann':
                # 训练ARIMA-ANN混合模型 - 使用自定义参数
                logger.info("开始训练ARIMA-ANN混合模型")
                results['arima_ann'] = train_arima_ann_models(
                    split_data,
                    features=features,
                    top_n_stations=args.top_n_stations,
                    hidden_layers=[128, 64, 32],  # 更深的网络
                    dropout_rate=0.3,  # 增加dropout以防止过拟合
                    batch_size=16,  # 更小的批量
                    epochs=200,  # 更多的训练轮数
                    patience=30  # 更长的耐心值
                )
            elif model == 'cnn_lstm':
                # 训练CNN-LSTM模型
                logger.info("开始训练CNN-LSTM模型")
                results['cnn_lstm'] = train_cnn_lstm_models(
                    split_data,
                    features=features,
                    top_n_stations=args.top_n_stations
                )
            elif model == 'cnn_gru_attention':
                # 训练改进的CNN-GRU-Attention模型
                logger.info("开始训练改进的CNN-GRU-Attention模型")
                results['cnn_gru_attention'] = train_cnn_gru_attention_models(
                    split_data,
                    features=features,
                    top_n_stations=args.top_n_stations,
                    forecast_horizon=forecast_horizon  # 传递预测步长参数
                )
            elif model == 'xgboost':
                # 训练XGBoost模型
                logger.info("开始训练XGBoost模型")
                results['xgboost'] = train_xgboost_models(
                    split_data,
                    features=features,
                    top_n_stations=args.top_n_stations
                )
        except Exception as e:
            logger.error(f"{model}模型训练失败: {str(e)}")

    logger.info("模型训练完成")
    return results


if __name__ == "__main__":
    main()
