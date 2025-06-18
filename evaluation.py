#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模型评估模块
用于评估和比较不同模型的预测性能
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from tqdm import tqdm
import warnings
import logging
import glob
import joblib
from config import get_english_feature_name, get_english_station_name

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("log/evaluation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 忽略警告
warnings.filterwarnings('ignore')

# 设置matplotlib参数为英文
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']

# 定义常量
METRICS_DIR = 'results/metrics/'
FIGURES_DIR = 'results/figures/'
MODELS_SAVE_DIR = 'models/saved/'


def load_metrics():
    """
    加载所有模型的评估指标

    Returns:
        dict: 包含各个模型评估指标的字典
    """
    logger.info("加载模型评估指标")

    metrics_dict = {}

    # 查找所有评估指标文件
    metrics_files = glob.glob(os.path.join(METRICS_DIR, '*.csv'))

    for file_path in metrics_files:
        model_name = os.path.basename(file_path).split('_')[0]

        try:
            metrics_df = pd.read_csv(file_path, encoding='utf-8')

            # 转换指标中的站点名和特征名为英文
            if 'station' in metrics_df.columns:
                metrics_df['station_english'] = metrics_df['station'].apply(get_english_station_name)
                metrics_df['original_station'] = metrics_df['station']
                metrics_df['station'] = metrics_df['station_english']

            if 'feature' in metrics_df.columns:
                metrics_df['feature_english'] = metrics_df['feature'].apply(get_english_feature_name)
                metrics_df['original_feature'] = metrics_df['feature']
                metrics_df['feature'] = metrics_df['feature_english']

            metrics_dict[model_name] = metrics_df
            logger.info(f"已加载 {model_name} 模型指标，共 {len(metrics_df)} 条记录")
        except Exception as e:
            logger.error(f"加载 {file_path} 失败: {str(e)}")

    return metrics_dict


def compare_models(metrics_dict, output_path=None):
    """
    比较不同模型的性能

    Args:
        metrics_dict: 包含各个模型评估指标的字典
        output_path: 比较结果保存路径

    Returns:
        pandas.DataFrame: 模型比较结果
    """
    logger.info("比较不同模型性能")

    all_models_metrics = []

    # 整合所有模型的评估指标
    for model_name, metrics_df in metrics_dict.items():
        # 添加模型名称列
        df = metrics_df.copy()
        df['model'] = model_name
        all_models_metrics.append(df)

    if not all_models_metrics:
        logger.warning("没有找到模型指标")
        return None

    # 合并所有评估指标
    combined_df = pd.concat(all_models_metrics, ignore_index=True)

    # 计算每个模型在各个评估指标上的平均值
    avg_metrics = combined_df.groupby('model').agg({
        'RMSE': 'mean',
        'MAE': 'mean',
        'MAPE': 'mean',
        'R2': 'mean'
    }).reset_index()

    # 根据RMSE升序排列
    avg_metrics = avg_metrics.sort_values('RMSE')

    # 保存比较结果
    if output_path:
        avg_metrics.to_csv(output_path, index=False, encoding='utf-8')
        logger.info(f"模型比较结果已保存至: {output_path}")

    return avg_metrics


def visualize_model_comparison(avg_metrics, save_path=None):
    """
    可视化模型比较结果

    Args:
        avg_metrics: 包含模型平均评估指标的DataFrame
        save_path: 图表保存路径
    """
    logger.info("可视化模型比较结果")

    if avg_metrics is None or avg_metrics.empty:
        logger.warning("没有数据可视化")
        return

    # 创建子图
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    axes = axes.flatten()

    metrics = ['RMSE', 'MAE', 'MAPE', 'R2']
    colors = ['blue', 'green', 'red', 'purple']

    for i, metric in enumerate(metrics):
        ax = axes[i]

        # 排序
        sorted_df = avg_metrics.sort_values(metric, ascending=False if metric == 'R2' else True)

        # 绘制条形图
        bars = ax.bar(sorted_df['model'], sorted_df[metric], color=colors[i], alpha=0.7)

        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom')

        # 设置标题和标签
        ax.set_title(f'Average {metric} Comparison')
        ax.set_xlabel('Model')
        ax.set_ylabel(metric)

        # 旋转x轴标签
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

        # 添加网格线
        ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()

    # 保存图表
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"模型比较图表已保存至: {save_path}")

    plt.close(fig)


def evaluate_prediction_by_feature(metrics_dict, save_dir=None):
    """
    按特征评估不同模型的预测性能 - 增强版支持多步预测

    Args:
        metrics_dict: 包含各个模型评估指标的字典
        save_dir: 评估结果保存目录
    """
    logger.info("按特征评估预测性能")

    all_models_metrics = []

    # 整合所有模型的评估指标
    for model_name, metrics_df in metrics_dict.items():
        # 添加模型名称列
        if 'feature' in metrics_df.columns:
            df = metrics_df.copy()
            df['model'] = model_name
            all_models_metrics.append(df)

    if not all_models_metrics:
        logger.warning("没有找到包含特征信息的模型指标")
        return

    # 合并所有评估指标
    combined_df = pd.concat(all_models_metrics, ignore_index=True)

    # 检查是否有多步预测指标（model_type中包含Step字样）
    has_multi_step = combined_df['model_type'].str.contains(
        'Step').any() if 'model_type' in combined_df.columns else False

    # 按特征分组计算平均值
    if has_multi_step:
        # 对于多步预测模型，分别处理每个步长和平均指标
        # 只选择平均指标和单步模型指标进行整体比较
        feature_metrics = combined_df[~combined_df['model_type'].str.contains('Step', na=False)].groupby(
            ['feature', 'model']).agg({
            'RMSE': 'mean',
            'MAE': 'mean',
            'MAPE': 'mean',
            'R2': 'mean'
        }).reset_index()

        # 单独处理多步预测模型的每个步长指标
        step_metrics = combined_df[combined_df['model_type'].str.contains('Step', na=False)].copy()
    else:
        # 对于传统单步预测模型，直接分组计算
        feature_metrics = combined_df.groupby(['feature', 'model']).agg({
            'RMSE': 'mean',
            'MAE': 'mean',
            'MAPE': 'mean',
            'R2': 'mean'
        }).reset_index()
        step_metrics = None

    # 为每个特征创建一个评估结果
    features = combined_df['feature'].unique()

    for feature in tqdm(features, desc="生成基于特征的评估报告"):
        # 获取原始的中文特征名
        original_feature = None
        if 'original_feature' in combined_df.columns:
            feature_rows = combined_df[combined_df['feature'] == feature]
            if not feature_rows.empty and 'original_feature' in feature_rows.columns:
                original_feature = feature_rows.iloc[0]['original_feature']

        # 使用英文特征名
        english_feature = feature

        # 筛选当前特征的数据
        feature_data = feature_metrics[feature_metrics['feature'] == feature]

        if feature_data.empty:
            continue

        # 根据RMSE排序
        feature_data = feature_data.sort_values('RMSE')

        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        axes = axes.flatten()

        metrics = ['RMSE', 'MAE', 'MAPE', 'R2']
        colors = ['blue', 'green', 'red', 'purple']

        for i, metric in enumerate(metrics):
            ax = axes[i]

            # 排序
            sorted_df = feature_data.sort_values(metric, ascending=False if metric == 'R2' else True)

            # 绘制条形图
            bars = ax.bar(sorted_df['model'], sorted_df[metric], color=colors[i], alpha=0.7)

            # 添加数值标签
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height,
                        f'{height:.3f}',
                        ha='center', va='bottom')

            # 设置标题和标签 - 使用英文特征名
            ax.set_title(f'{english_feature} - Average {metric} Comparison')
            ax.set_xlabel('Model')
            ax.set_ylabel(metric)

            # 旋转x轴标签
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

            # 添加网格线
            ax.grid(True, linestyle='--', alpha=0.5)

        plt.tight_layout()

        # 保存图表
        if save_dir:
            # 创建安全的文件名
            safe_feature = english_feature.replace('/', '_').replace('\\', '_').replace(':', '_')
            save_path = os.path.join(save_dir, f'feature_comparison_{safe_feature}.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.close(fig)

        # 如果有多步预测数据，创建多步预测比较图
        if step_metrics is not None:
            feature_step_data = step_metrics[step_metrics['feature'] == feature]

            if not feature_step_data.empty:
                # 提取步长信息
                feature_step_data['step'] = feature_step_data['model_type'].str.extract(r'Step(\d+)').astype(int)

                # 按步长分组，创建折线图
                plt.figure(figsize=(15, 10))

                # 按模型分组
                for model, group in feature_step_data.groupby('model'):
                    # 按步长排序
                    group = group.sort_values('step')
                    # 绘制RMSE随步长的变化
                    plt.plot(group['step'], group['RMSE'], marker='o', label=f'{model}')

                # 使用英文特征名
                plt.title(f'{english_feature} - Multi-step Prediction RMSE Comparison')
                plt.xlabel('Prediction Step')
                plt.ylabel('RMSE')
                plt.legend()
                plt.grid(True, linestyle='--', alpha=0.5)
                plt.tight_layout()

                # 保存图表
                if save_dir:
                    multi_step_path = os.path.join(save_dir, f'multi_step_{safe_feature}_rmse.png')
                    plt.savefig(multi_step_path, dpi=300, bbox_inches='tight')

                plt.close()

                # 创建R²变化图
                plt.figure(figsize=(15, 10))

                # 按模型分组
                for model, group in feature_step_data.groupby('model'):
                    # 按步长排序
                    group = group.sort_values('step')
                    # 绘制R²随步长的变化
                    plt.plot(group['step'], group['R2'], marker='o', label=f'{model}')

                # 使用英文特征名
                plt.title(f'{english_feature} - Multi-step Prediction R² Comparison')
                plt.xlabel('Prediction Step')
                plt.ylabel('R²')
                plt.legend()
                plt.grid(True, linestyle='--', alpha=0.5)
                plt.tight_layout()

                # 保存图表
                if save_dir:
                    multi_step_r2_path = os.path.join(save_dir, f'multi_step_{safe_feature}_r2.png')
                    plt.savefig(multi_step_r2_path, dpi=300, bbox_inches='tight')

                plt.close()


def evaluate_prediction_by_station(metrics_dict, save_dir=None):
    """
    按站点评估不同模型的预测性能

    Args:
        metrics_dict: 包含各个模型评估指标的字典
        save_dir: 评估结果保存目录
    """
    logger.info("按站点评估预测性能")

    all_models_metrics = []

    # 整合所有模型的评估指标
    for model_name, metrics_df in metrics_dict.items():
        # 添加模型名称列
        if 'station' in metrics_df.columns:
            df = metrics_df.copy()
            df['model'] = model_name
            all_models_metrics.append(df)

    if not all_models_metrics:
        logger.warning("没有找到包含站点信息的模型指标")
        return

    # 合并所有评估指标
    combined_df = pd.concat(all_models_metrics, ignore_index=True)

    # 按站点分组计算平均值
    station_metrics = combined_df.groupby(['station', 'model']).agg({
        'RMSE': 'mean',
        'MAE': 'mean',
        'MAPE': 'mean',
        'R2': 'mean'
    }).reset_index()

    # 为每个站点创建一个评估结果
    stations = combined_df['station'].unique()

    for station in tqdm(stations, desc="生成基于站点的评估报告"):
        # 获取原始的中文站点名
        original_station = None
        if 'original_station' in combined_df.columns:
            station_rows = combined_df[combined_df['station'] == station]
            if not station_rows.empty and 'original_station' in station_rows.columns:
                original_station = station_rows.iloc[0]['original_station']

        # 使用英文站点名
        english_station = station

        # 筛选当前站点的数据
        station_data = station_metrics[station_metrics['station'] == station]

        if station_data.empty:
            continue

        # 根据RMSE排序
        station_data = station_data.sort_values('RMSE')

        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        axes = axes.flatten()

        metrics = ['RMSE', 'MAE', 'MAPE', 'R2']
        colors = ['blue', 'green', 'red', 'purple']

        for i, metric in enumerate(metrics):
            ax = axes[i]

            # 排序
            sorted_df = station_data.sort_values(metric, ascending=False if metric == 'R2' else True)

            # 绘制条形图
            bars = ax.bar(sorted_df['model'], sorted_df[metric], color=colors[i], alpha=0.7)

            # 添加数值标签
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height,
                        f'{height:.3f}',
                        ha='center', va='bottom')

            # 设置标题和标签 - 使用英文站点名
            ax.set_title(f'{english_station} - Average {metric} Comparison')
            ax.set_xlabel('Model')
            ax.set_ylabel(metric)

            # 旋转x轴标签
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

            # 添加网格线
            ax.grid(True, linestyle='--', alpha=0.5)

        plt.tight_layout()

        # 保存图表
        if save_dir:
            # 创建安全的文件名
            safe_station = english_station.replace('/', '_').replace('\\', '_').replace(':', '_')
            save_path = os.path.join(save_dir, f'station_comparison_{safe_station}.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.close(fig)

    logger.info("站点评估完成")


def main():
    """主函数"""
    logger.info("开始模型评估")

    # 确保输出目录存在
    os.makedirs(METRICS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # 加载所有模型的评估指标
    metrics_dict = load_metrics()

    if not metrics_dict:
        logger.error("没有找到模型指标")
        return

    # 比较不同模型的性能
    output_path = os.path.join(METRICS_DIR, 'model_comparison.csv')
    avg_metrics = compare_models(metrics_dict, output_path)

    # 可视化模型比较结果
    save_path = os.path.join(FIGURES_DIR, 'model_comparison.png')
    visualize_model_comparison(avg_metrics, save_path)

    # 按特征评估不同模型的预测性能
    evaluate_prediction_by_feature(metrics_dict, FIGURES_DIR)

    # 按站点评估不同模型的预测性能
    evaluate_prediction_by_station(metrics_dict, FIGURES_DIR)

    logger.info("模型评估完成")


if __name__ == "__main__":
    main()