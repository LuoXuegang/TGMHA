#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据预处理模块
用于处理原始水质数据，包括：
1. 数据清洗（缺失值、异常值处理）
2. 数据格式转换
3. 时间序列数据准备
4. 数据拆分（训练集、验证集、测试集）
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

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("log/data_preprocessing.log"),
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
RAW_DATA_PATH = 'data/raw/sichuan_data.csv'
PROCESSED_DATA_DIR = 'data/processed/'
FEATURES_TO_ANALYZE = [
    '水温(℃)', 'pH(无量纲)', '溶解氧(mg/L)', '电导率(μS/cm)',
    '浊度(NTU)', '高锰酸盐指数(mg/L)', '氨氮(mg/L)',
    '总磷(mg/L)', '总氮(mg/L)'
]


def create_dirs():
    """创建必要的目录结构"""
    dirs = [
        'data/processed',
        'results/figures',
        'results/metrics',
        'models/saved',
        'log'
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    logger.info("目录结构创建完成")


def load_data(file_path=RAW_DATA_PATH):
    """
    加载原始数据

    Args:
        file_path: 数据文件路径

    Returns:
        pandas.DataFrame: 加载的数据
    """
    logger.info(f"开始加载数据: {file_path}")
    try:
        # 使用utf-8编码读取CSV文件
        df = pd.read_csv(file_path, encoding='utf-8')
        logger.info(f"数据加载成功，共 {len(df)} 条记录")
        return df
    except Exception as e:
        logger.error(f"数据加载失败: {str(e)}")
        raise


def clean_data(df):
    """
    数据清洗

    Args:
        df: 原始数据DataFrame

    Returns:
        pandas.DataFrame: 清洗后的数据
    """
    logger.info("开始数据清洗")

    # 制作数据副本
    df_clean = df.copy()

    # 记录原始数据大小
    original_size = len(df_clean)
    logger.info(f"原始数据大小: {original_size}")

    # 1. 处理日期时间格式
    logger.info("处理日期时间格式")
    if '监测时间' in df_clean.columns:
        # 检查监测时间列的格式，确保格式统一
        try:
            # 检查是否有空值
            if df_clean['监测时间'].isna().sum() > 0:
                logger.warning(f"监测时间列存在 {df_clean['监测时间'].isna().sum()} 个空值")

            # 尝试转换日期格式
            df_clean['监测时间'] = pd.to_datetime(df_clean['监测时间'], errors='coerce')

            # 检查转换后是否有NaT值
            if df_clean['监测时间'].isna().sum() > 0:
                logger.warning(f"监测时间格式转换后有 {df_clean['监测时间'].isna().sum()} 个无效值")
        except Exception as e:
            logger.error(f"日期格式处理失败: {str(e)}")
    else:
        logger.warning("数据中不存在'监测时间'列，跳过日期处理")

    # 2. 处理特征数值列的数据类型
    logger.info("处理特征数值列的数据类型")
    for col in tqdm(FEATURES_TO_ANALYZE, desc="处理特征列"):
        if col in df_clean.columns:
            # 替换特殊值
            df_clean[col] = df_clean[col].replace({'--': np.nan, '*': np.nan})

            # 转换为数值类型
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

            # 记录缺失值情况
            missing_count = df_clean[col].isna().sum()
            if missing_count > 0:
                missing_pct = missing_count / len(df_clean) * 100
                logger.info(f"列 '{col}' 有 {missing_count} 个缺失值 ({missing_pct:.2f}%)")

    # 3. 删除所有特征都为空的行
    all_features_null = df_clean[FEATURES_TO_ANALYZE].isna().all(axis=1)
    if all_features_null.sum() > 0:
        logger.info(f"删除所有特征都为空的行: {all_features_null.sum()} 行")
        df_clean = df_clean[~all_features_null]

    # 4. 异常值处理 - 使用IQR方法
    logger.info("处理异常值")
    for col in tqdm(FEATURES_TO_ANALYZE, desc="处理异常值"):
        if col in df_clean.columns:
            # 仅处理数值列
            if pd.api.types.is_numeric_dtype(df_clean[col]):
                # 计算IQR
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1

                # 定义异常值边界
                lower_bound = Q1 - 2.0 * IQR
                upper_bound = Q3 + 2.0 * IQR

                # 统计异常值数量
                outliers = ((df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)).sum()
                if outliers > 0:
                    outlier_pct = outliers / df_clean[col].count() * 100
                    logger.info(f"列 '{col}' 有 {outliers} 个异常值 ({outlier_pct:.2f}%)")

                    # 将异常值替换为NaN
                    # 注意：这里不直接删除异常值，而是将其标记为缺失值，后续可能进行插补
                    df_clean.loc[(df_clean[col] < lower_bound) | (df_clean[col] > upper_bound), col] = np.nan

    # 5. 处理缺失值 - 对于时间序列数据，可以使用前向或后向填充
    logger.info("处理缺失值")

    # 对于站点特定的数据，按站点和时间分组后填充
    if all(col in df_clean.columns for col in ['断面名称', '监测时间']):
        # 按断面名称和时间排序
        df_clean = df_clean.sort_values(['断面名称', '监测时间'])

        # 使用前向填充处理缺失值（在同一断面内）
        for station in tqdm(df_clean['断面名称'].unique(), desc="按站点处理缺失值"):
            station_mask = df_clean['断面名称'] == station
            # 对数值特征进行填充
            for col in FEATURES_TO_ANALYZE:
                if col in df_clean.columns:
                    # 使用前向和后向填充组合
                    df_clean.loc[station_mask, col] = df_clean.loc[station_mask, col].fillna(method='ffill')
                    df_clean.loc[station_mask, col] = df_clean.loc[station_mask, col].fillna(method='bfill')

    # 如果还存在缺失值，使用列的中位数填充
    for col in FEATURES_TO_ANALYZE:
        if col in df_clean.columns:
            missing_after = df_clean[col].isna().sum()
            if missing_after > 0:
                logger.info(f"列 '{col}' 在填充后仍有 {missing_after} 个缺失值，使用中位数填充")
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())

    # 记录清洗后的数据大小
    cleaned_size = len(df_clean)
    logger.info(f"清洗后数据大小: {cleaned_size}")
    logger.info(f"移除了 {original_size - cleaned_size} 条记录")

    return df_clean


def prepare_time_series(df, target_features=None):
    """
    准备时间序列数据

    Args:
        df: 清洗后的数据
        target_features: 目标特征列表，默认为None时使用所有FEATURES_TO_ANALYZE

    Returns:
        dict: 按断面名称分组的时间序列数据
    """
    if target_features is None:
        target_features = FEATURES_TO_ANALYZE

    logger.info("准备时间序列数据")

    # 检查必要的列是否存在
    required_cols = ['断面名称', '监测时间'] + target_features
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logger.error(f"缺少必要的列: {missing_cols}")
        raise ValueError(f"缺少必要的列: {missing_cols}")

    # 确保日期列是datetime类型
    if not pd.api.types.is_datetime64_dtype(df['监测时间']):
        logger.info("转换监测时间为datetime类型")
        df['监测时间'] = pd.to_datetime(df['监测时间'], errors='coerce')

    # 按断面名称分组并整理时间序列
    time_series_data = {}

    for station in tqdm(df['断面名称'].unique(), desc="准备站点时间序列"):
        # 获取单个站点的数据
        station_data = df[df['断面名称'] == station].copy()

        # 按时间排序
        station_data = station_data.sort_values('监测时间')

        # 选择目标特征
        features_data = station_data[['监测时间'] + target_features]

        # 使用日期作为索引
        features_data.set_index('监测时间', inplace=True)

        # 存储到字典中
        time_series_data[station] = features_data

    logger.info(f"共准备了 {len(time_series_data)} 个站点的时间序列数据")
    return time_series_data


def split_time_series(time_series_data, train_ratio=0.8, val_ratio=0.1):
    """
    拆分时间序列数据为训练集、验证集和测试集

    Args:
        time_series_data: 按站点分组的时间序列数据字典
        train_ratio: 训练集比例，默认0.7
        val_ratio: 验证集比例，默认0.15（测试集比例为1-train_ratio-val_ratio）

    Returns:
        dict: 包含训练集、验证集和测试集的字典
    """
    logger.info("拆分时间序列数据")

    split_data = {
        'train': {},
        'val': {},
        'test': {}
    }

    for station, data in tqdm(time_series_data.items(), desc="拆分站点数据"):
        # 获取数据点数量
        n = len(data)

        # 计算拆分点
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)

        # 拆分数据
        split_data['train'][station] = data.iloc[:train_end]
        split_data['val'][station] = data.iloc[train_end:val_end]
        split_data['test'][station] = data.iloc[val_end:]

        # 记录拆分情况
        logger.info(f"站点 '{station}' 数据拆分: 训练集 {len(split_data['train'][station])},"
                    f" 验证集 {len(split_data['val'][station])}, 测试集 {len(split_data['test'][station])}")

    return split_data


def save_processed_data(data_dict, base_dir=PROCESSED_DATA_DIR):
    """
    保存处理后的数据

    Args:
        data_dict: 数据字典，包含'df_clean'和'split_data'
        base_dir: 基础目录
    """
    logger.info(f"保存处理后的数据到 {base_dir}")

    # 保存清洗后的完整数据集
    clean_data_path = os.path.join(base_dir, 'clean_data.csv')
    data_dict['df_clean'].to_csv(clean_data_path, index=False, encoding='utf-8')
    logger.info(f"清洗后的完整数据保存至: {clean_data_path}")

    # 保存按站点拆分的数据
    for split_type, stations_data in data_dict['split_data'].items():
        # 创建拆分类型目录
        split_dir = os.path.join(base_dir, split_type)
        os.makedirs(split_dir, exist_ok=True)

        # 保存每个站点的数据
        for station, df in stations_data.items():
            # 创建合法的文件名
            station_filename = station.replace('/', '_').replace('\\', '_').replace(':', '_')
            station_path = os.path.join(split_dir, f"{station_filename}.csv")
            df.to_csv(station_path, encoding='utf-8')

        logger.info(f"{split_type}集中的 {len(stations_data)} 个站点数据已保存")


def main():
    """主函数"""
    logger.info("开始数据预处理")

    # 创建目录结构
    create_dirs()

    # 加载数据
    df = load_data()

    # 数据清洗
    df_clean = clean_data(df)

    # 准备时间序列数据
    time_series_data = prepare_time_series(df_clean)

    # 拆分数据
    split_data = split_time_series(time_series_data)

    # 保存处理后的数据
    save_processed_data({
        'df_clean': df_clean,
        'split_data': split_data
    })

    logger.info("数据预处理完成")
    return df_clean, split_data


if __name__ == "__main__":
    main()