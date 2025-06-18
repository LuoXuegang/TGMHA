#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
结果可视化模块
用于生成各种可视化图表，包括：
1. 时间序列预测结果
2. 模型比较图表
3. 特征重要性
4. 各种水质指标的分布和相关性
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
        logging.FileHandler("log/visualization.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 忽略警告
warnings.filterwarnings('ignore')

# 设置matplotlib参数为英文
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
# 注释掉中文字体设置
# plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']
# plt.rcParams['axes.unicode_minus'] = False

# 定义常量
PROCESSED_DATA_DIR = 'data/processed/'
FIGURES_DIR = 'results/figures/'
METRICS_DIR = 'results/metrics/'


def load_data():
    """
    加载已处理的数据

    Returns:
        dict: 包含清洗后的数据和分割后的数据集
    """
    logger.info("Loading data...")

    # 加载清洗后的完整数据
    clean_data_path = os.path.join(PROCESSED_DATA_DIR, 'clean_data.csv')
    df_clean = None

    try:
        df_clean = pd.read_csv(clean_data_path, encoding='utf-8')
        # 如果有监测时间列，将其转换为datetime
        if '监测时间' in df_clean.columns:
            df_clean['监测时间'] = pd.to_datetime(df_clean['监测时间'])
        logger.info(f"Loaded cleaned data, {len(df_clean)} records in total")
    except Exception as e:
        logger.error(f"Failed to load cleaned data: {str(e)}")

    # 加载分割后的数据集
    split_data = {
        'train': {},
        'val': {},
        'test': {}
    }

    for split_type in split_data.keys():
        split_dir = os.path.join(PROCESSED_DATA_DIR, split_type)

        if os.path.exists(split_dir):
            csv_files = glob.glob(os.path.join(split_dir, "*.csv"))

            for csv_file in csv_files:
                try:
                    station_filename = os.path.basename(csv_file)
                    station_name = os.path.splitext(station_filename)[0].replace('_', '/')

                    df = pd.read_csv(csv_file, encoding='utf-8')
                    # 如果有监测时间列，将其转换为datetime
                    if '监测时间' in df.columns:
                        df['监测时间'] = pd.to_datetime(df['监测时间'])
                        df.set_index('监测时间', inplace=True)

                    split_data[split_type][station_name] = df
                except Exception as e:
                    logger.error(f"Failed to load file {csv_file}: {str(e)}")

        logger.info(f"Loaded {split_type} dataset, {len(split_data[split_type])} stations in total")

    return {
        'df_clean': df_clean,
        'split_data': split_data
    }


def create_water_quality_dashboard(df_clean, save_path=None):
    """
    创建水质数据仪表盘

    Args:
        df_clean: 清洗后的数据
        save_path: 图表保存路径
    """
    logger.info("Creating water quality dashboard")

    if df_clean is None:
        logger.error("Missing cleaned data")
        return

    # 检查必要的列是否存在
    required_cols = [
        '水温(℃)', 'pH(无量纲)', '溶解氧(mg/L)', '电导率(μS/cm)',
        '浊度(NTU)', '高锰酸盐指数(mg/L)', '氨氮(mg/L)',
        '总磷(mg/L)', '总氮(mg/L)', '水质类别'
    ]

    missing_cols = [col for col in required_cols if col not in df_clean.columns]
    if missing_cols:
        logger.warning(f"Missing required columns: {missing_cols}")

    available_cols = [col for col in required_cols if col in df_clean.columns]

    # 创建仪表盘
    fig = plt.figure(figsize=(20, 15))
    gs = GridSpec(3, 3, figure=fig)

    # 1. 水质类别分布 (饼图)
    if '水质类别' in available_cols:
        ax1 = fig.add_subplot(gs[0, 0])
        water_quality_counts = df_clean['水质类别'].value_counts()

        # 去除非标准水质类别
        standard_categories = ['Ⅰ', 'Ⅱ', 'Ⅲ', 'Ⅳ', 'Ⅴ', '劣Ⅴ']
        water_quality_counts = water_quality_counts.filter(
            items=[cat for cat in standard_categories if cat in water_quality_counts.index])

        # 绘制饼图
        wedges, texts, autotexts = ax1.pie(
            water_quality_counts,
            labels=water_quality_counts.index,
            autopct='%1.1f%%',
            startangle=90,
            colors=sns.color_palette("Set3", len(water_quality_counts))
        )

        # 增强文本可读性
        for text in texts + autotexts:
            text.set_fontsize(10)

        ax1.set_title('Water Quality Category Distribution')

    # 2. 各项指标的统计分布 (箱线图)
    numeric_cols = [col for col in available_cols if col != '水质类别']
    english_cols = [get_english_feature_name(col) for col in numeric_cols]

    if numeric_cols:
        ax2 = fig.add_subplot(gs[0, 1:])

        # 创建长格式数据
        long_data = []
        for i, col in enumerate(numeric_cols):
            if pd.api.types.is_numeric_dtype(df_clean[col]):
                temp_df = pd.DataFrame({
                    'indicator': english_cols[i],
                    'value': df_clean[col]
                })
                long_data.append(temp_df)

        if long_data:
            long_df = pd.concat(long_data)

            # 绘制箱线图
            sns.boxplot(x='indicator', y='value', data=long_df, ax=ax2)
            ax2.set_title('Water Quality Indicator Distribution')
            ax2.set_xlabel('')
            plt.xticks(rotation=45, ha='right')

    # 3. 相关性热图
    if numeric_cols:
        ax3 = fig.add_subplot(gs[1, :2])

        # 计算相关系数
        corr_matrix = df_clean[numeric_cols].corr()

        # 修改列名为英文
        corr_matrix.columns = english_cols
        corr_matrix.index = english_cols

        # 绘制热图
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(
            corr_matrix,
            mask=mask,
            annot=True,
            fmt=".2f",
            cmap='coolwarm',
            square=True,
            linewidths=.5,
            cbar_kws={"shrink": .5},
            ax=ax3
        )
        ax3.set_title('Water Quality Indicator Correlation')

    # 4. 站点水质类别分布 (前20个站点)
    if '水质类别' in available_cols and '断面名称' in df_clean.columns:
        ax4 = fig.add_subplot(gs[1, 2])

        # 计算各站点的水质类别众数
        station_quality = df_clean.groupby('断面名称')['水质类别'].agg(
            lambda x: x.mode()[0] if not x.mode().empty else np.nan
        ).reset_index()

        # 将站点名称转换为英文
        station_quality['station_name'] = station_quality['断面名称'].apply(get_english_station_name)

        # 统计各水质类别的站点数量
        quality_counts = station_quality['水质类别'].value_counts().sort_index()

        # 绘制条形图
        bars = ax4.bar(
            quality_counts.index,
            quality_counts.values,
            color=sns.color_palette("Set3", len(quality_counts))
        )

        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax4.text(
                bar.get_x() + bar.get_width() / 2.,
                height,
                f'{int(height)}',
                ha='center',
                va='bottom'
            )
        ax4.set_title('Number of Stations by Water Quality Category')
        ax4.set_xlabel('Water Quality Category')
        ax4.set_ylabel('Number of Stations')

    # 5. pH值与溶解氧关系散点图 (按水质类别着色)
    if all(col in available_cols for col in ['pH(无量纲)', '溶解氧(mg/L)', '水质类别']):
        ax5 = fig.add_subplot(gs[2, 0])

        # 获取英文特征名
        ph_eng = get_english_feature_name('pH(无量纲)')
        do_eng = get_english_feature_name('溶解氧(mg/L)')

        sns.scatterplot(
            x='pH(无量纲)',
            y='溶解氧(mg/L)',
            hue='水质类别',
            data=df_clean,
            palette='Set1',
            ax=ax5
        )

        ax5.set_title('pH vs Dissolved Oxygen Relationship')
        ax5.set_xlabel(ph_eng)
        ax5.set_ylabel(do_eng)
        ax5.grid(True, linestyle='--', alpha=0.5)

    # 6. 氨氮与总磷关系散点图 (按水质类别着色)
    if all(col in available_cols for col in ['氨氮(mg/L)', '总磷(mg/L)', '水质类别']):
        ax6 = fig.add_subplot(gs[2, 1])

        # 获取英文特征名
        nh4_eng = get_english_feature_name('氨氮(mg/L)')
        tp_eng = get_english_feature_name('总磷(mg/L)')

        sns.scatterplot(
            x='氨氮(mg/L)',
            y='总磷(mg/L)',
            hue='水质类别',
            data=df_clean,
            palette='Set1',
            ax=ax6
        )

        ax6.set_title('Ammonia Nitrogen vs Total Phosphorus Relationship')
        ax6.set_xlabel(nh4_eng)
        ax6.set_ylabel(tp_eng)
        ax6.grid(True, linestyle='--', alpha=0.5)

    # 7. 水温与电导率关系散点图 (按水质类别着色)
    if all(col in available_cols for col in ['水温(℃)', '电导率(μS/cm)', '水质类别']):
        ax7 = fig.add_subplot(gs[2, 2])

        # 获取英文特征名
        temp_eng = get_english_feature_name('水温(℃)')
        cond_eng = get_english_feature_name('电导率(μS/cm)')

        sns.scatterplot(
            x='水温(℃)',
            y='电导率(μS/cm)',
            hue='水质类别',
            data=df_clean,
            palette='Set1',
            ax=ax7
        )

        ax7.set_title('Water Temperature vs Conductivity Relationship')
        ax7.set_xlabel(temp_eng)
        ax7.set_ylabel(cond_eng)
        ax7.grid(True, linestyle='--', alpha=0.5)

    # 调整布局
    plt.tight_layout()

    # 保存图表
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Water quality dashboard saved to: {save_path}")

    plt.close(fig)


def create_water_quality_distribution_maps(df_clean, save_dir=None):
    """
    创建水质指标的地理分布图

    Args:
        df_clean: 清洗后的数据
        save_dir: 图表保存目录
    """
    # 注意：由于无法获取地理坐标信息，此函数仅作为示例
    # 实际应用需要添加站点的经纬度信息

    logger.info("Creating water quality indicator distribution maps")

    if df_clean is None:
        logger.error("Missing cleaned data")
        return

    # 检查必要的列是否存在
    if '断面名称' not in df_clean.columns:
        logger.error("Station name column missing in data")
        return

    # 水质指标列表
    indicators = [
        '水温(℃)', 'pH(无量纲)', '溶解氧(mg/L)', '电导率(μS/cm)',
        '浊度(NTU)', '高锰酸盐指数(mg/L)', '氨氮(mg/L)',
        '总磷(mg/L)', '总氮(mg/L)'
    ]

    available_indicators = [ind for ind in indicators if ind in df_clean.columns]

    if not available_indicators:
        logger.error("No water quality indicator columns found in data")
        return

    # 计算各站点水质指标的平均值
    station_avg = df_clean.groupby('断面名称')[available_indicators].mean().reset_index()

    # 添加英文站点名称
    station_avg['station_eng'] = station_avg['断面名称'].apply(get_english_station_name)

    # 创建水质指标分布图
    for indicator in available_indicators:
        try:
            plt.figure(figsize=(12, 8))

            # 获取英文指标名称
            indicator_eng = get_english_feature_name(indicator)

            # 这里模拟地理分布图，实际应用需要使用地理坐标绘制真实地图
            # 使用站点作为X轴，指标值作为Y轴的条形图
            station_sorted = station_avg.sort_values(indicator)
            top_20_stations = station_sorted.tail(20)  # 取指标值最高的20个站点

            plt.bar(top_20_stations['station_eng'], top_20_stations[indicator], color='skyblue')

            plt.title(f'{indicator_eng} - Top 20 Stations by Value')
            plt.xticks(rotation=90)
            plt.xlabel('Station Name')
            plt.ylabel(indicator_eng)
            plt.grid(True, linestyle='--', alpha=0.5)

            plt.tight_layout()

            # 保存图表
            if save_dir:
                safe_indicator = indicator.replace('/', '_').replace('\\', '_').replace(':', '_')
                save_path = os.path.join(save_dir, f'distribution_{safe_indicator}.png')
                plt.savefig(save_path, dpi=300, bbox_inches='tight')

            plt.close()
        except Exception as e:
            logger.error(f"Failed to create distribution map for {indicator}: {str(e)}")


def visualize_time_series_predictions(split_data, metrics_dict=None, save_dir=None):
    """
    可视化时间序列预测结果

    Args:
        split_data: 包含训练、验证和测试数据的字典
        metrics_dict: 模型评估指标
        save_dir: 图表保存目录
    """
    logger.info("Visualizing time series prediction results")

    # 检查数据
    if not split_data or not all(key in split_data for key in ['train', 'test']):
        logger.error("Missing required datasets")
        return

    # 获取站点列表
    stations = list(split_data['train'].keys())

    if not stations:
        logger.error("No station data available")
        return

    # 获取特征列表（以第一个站点为例）
    first_station = stations[0]
    if first_station not in split_data['train']:
        logger.error(f"Station {first_station} does not exist")
        return

    features = list(split_data['train'][first_station].columns)

    # 加载模型评估指标（如果提供）
    model_metrics = {}
    if metrics_dict:
        for model_name, metrics_df in metrics_dict.items():
            if 'station' in metrics_df.columns and 'feature' in metrics_df.columns:
                for _, row in metrics_df.iterrows():
                    station = row['station']
                    feature = row['feature']

                    if station not in model_metrics:
                        model_metrics[station] = {}

                    if feature not in model_metrics[station]:
                        model_metrics[station][feature] = {}

                    model_metrics[station][feature][model_name] = {
                        'RMSE': row.get('RMSE', None),
                        'MAE': row.get('MAE', None),
                        'MAPE': row.get('MAPE', None),
                        'R2': row.get('R2', None)
                    }

    # 选择部分站点和特征进行可视化（避免生成过多图表）
    selected_stations = stations[:5]  # 取前5个站点

    for station in tqdm(selected_stations, desc="Station Processing Progress"):
        # 转换站点名称为英文
        english_station = get_english_station_name(station)

        for feature in features:
            # 转换特征名称为英文
            english_feature = get_english_feature_name(feature)

            # 检查数据是否存在
            if (station in split_data['train'] and station in split_data['test'] and
                    feature in split_data['train'][station].columns and
                    feature in split_data['test'][station].columns):

                # 获取训练和测试数据
                train_data = split_data['train'][station][feature]
                test_data = split_data['test'][station][feature]

                # 创建图表
                plt.figure(figsize=(12, 6))

                # 绘制训练数据
                plt.plot(train_data.index, train_data.values, 'b-', label='Training Data')
                plt.plot(test_data.index, test_data.values, 'g-', label='Test Data')

                # 添加模型评估信息（如果有）
                if station in model_metrics and feature in model_metrics[station]:
                    metrics_text = ""
                    for model_name, metrics in model_metrics[station][feature].items():
                        metrics_text += f"{model_name}:\n"
                        for metric_name, value in metrics.items():
                            if value is not None:
                                metrics_text += f"  {metric_name}: {value:.4f}\n"

                    plt.figtext(0.02, 0.02, metrics_text, fontsize=9,
                                bbox=dict(facecolor='white', alpha=0.8))

                # 设置标题和标签 - 使用英文站点和特征名
                plt.title(f'{english_station} - {english_feature} Time Series')
                plt.xlabel('Time')
                plt.ylabel(english_feature)

                plt.legend()
                plt.grid(True, linestyle='--', alpha=0.5)

                # 自动旋转日期标签以避免重叠
                plt.gcf().autofmt_xdate()

                plt.tight_layout()

                # 保存图表
                if save_dir:
                    safe_station = station.replace('/', '_').replace('\\', '_').replace(':', '_')
                    safe_feature = feature.replace('/', '_').replace('\\', '_').replace(':', '_')
                    save_path = os.path.join(save_dir, f'timeseries_{safe_station}_{safe_feature}.png')
                    plt.savefig(save_path, dpi=300, bbox_inches='tight')

                plt.close()


def create_model_performance_summary(metrics_dict, save_path=None):
    """
    创建模型性能总结图表

    Args:
        metrics_dict: 包含各个模型评估指标的字典
        save_path: 图表保存路径
    """
    logger.info("Creating model performance summary chart")

    if not metrics_dict:
        logger.error("Missing model evaluation metrics")
        return

    # 汇总各模型的平均指标
    summary_data = []

    for model_name, metrics_df in metrics_dict.items():
        if 'RMSE' in metrics_df.columns:
            model_summary = {
                'model': model_name,
                'RMSE': metrics_df['RMSE'].mean(),
                'MAE': metrics_df['MAE'].mean() if 'MAE' in metrics_df.columns else np.nan,
                'MAPE': metrics_df['MAPE'].mean() if 'MAPE' in metrics_df.columns else np.nan,
                'R2': metrics_df['R2'].mean() if 'R2' in metrics_df.columns else np.nan,
                'count': len(metrics_df)
            }
            summary_data.append(model_summary)

    if not summary_data:
        logger.error("Cannot calculate model performance summary")
        return

    # 创建汇总数据框
    summary_df = pd.DataFrame(summary_data)

    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()

    metrics = ['RMSE', 'MAE', 'MAPE', 'R2']
    colors = plt.cm.tab10(np.linspace(0, 1, len(summary_df)))

    for i, metric in enumerate(metrics):
        ax = axes[i]

        # 按指标排序（R2是越高越好，其他是越低越好）
        sorted_df = summary_df.sort_values(metric, ascending=False if metric == 'R2' else True)

        # 绘制条形图
        bars = ax.barh(sorted_df['model'], sorted_df[metric], color=colors)

        # 添加数值标签
        for bar in bars:
            width = bar.get_width()
            label_x_pos = width * 1.01
            ax.text(label_x_pos, bar.get_y() + bar.get_height() / 2, f'{width:.4f}',
                    va='center')

        # 设置标题和标签
        ax.set_title(f'Average {metric}')
        ax.set_xlabel(metric)
        ax.set_ylabel('Model')

        # 添加网格线
        ax.grid(True, linestyle='--', alpha=0.5)

    # 设置总标题
    plt.suptitle('Model Performance Comparison', fontsize=16)

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    # 保存图表
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Model performance summary chart saved to: {save_path}")

    plt.close(fig)


def main():
    """主函数"""
    logger.info("Starting result visualization")

    # 确保输出目录存在
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # 加载数据
    data_dict = load_data()
    df_clean = data_dict['df_clean']
    split_data = data_dict['split_data']

    # 创建水质数据仪表盘
    dashboard_path = os.path.join(FIGURES_DIR, 'water_quality_dashboard.png')
    create_water_quality_dashboard(df_clean, dashboard_path)

    # 创建水质指标的地理分布图
    create_water_quality_distribution_maps(df_clean, FIGURES_DIR)

    # 加载模型评估指标
    metrics_dict = {}
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
        except Exception as e:
            logger.error(f"Failed to load {file_path}: {str(e)}")

    # 可视化时间序列预测结果
    visualize_time_series_predictions(split_data, metrics_dict, FIGURES_DIR)

    # 创建模型性能总结图表
    summary_path = os.path.join(FIGURES_DIR, 'model_performance_summary.png')
    create_model_performance_summary(metrics_dict, summary_path)

    logger.info("Result visualization completed")


if __name__ == "__main__":
    main()