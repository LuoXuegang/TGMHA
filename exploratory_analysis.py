#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
探索性数据分析模块
用于对水质数据进行探索性分析，包括：
1. 基本统计分析
2. 数据分布可视化
3. 时间序列趋势分析
4. 相关性分析
5. ACF和PACF分析
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
import warnings
import logging
from datetime import datetime
import matplotlib.dates as mdates  # 添加这一行
from matplotlib.gridspec import GridSpec  # 如果使用GridSpec，也添加这一行
from config import get_english_feature_name, get_english_station_name

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("log/exploratory_analysis.log"),
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
PROCESSED_DATA_PATH = 'data/processed/clean_data.csv'
FIGURES_DIR = 'results/figures/'
FEATURES_TO_ANALYZE = [
    '水温(℃)', 'pH(无量纲)', '溶解氧(mg/L)', '电导率(μS/cm)',
    '浊度(NTU)', '高锰酸盐指数(mg/L)', '氨氮(mg/L)',
    '总磷(mg/L)', '总氮(mg/L)', '叶绿素α(mg/L)', '藻密度(cells/L)'
]


def load_clean_data(file_path=PROCESSED_DATA_PATH):
    """
    加载清洗后的数据

    Args:
        file_path: 清洗后数据文件路径

    Returns:
        pandas.DataFrame: 清洗后的数据
    """
    logger.info(f"加载清洗后的数据: {file_path}")
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
        # 如果监测时间列存在，转换为datetime
        if '监测时间' in df.columns:
            df['监测时间'] = pd.to_datetime(df['监测时间'])
        logger.info(f"数据加载成功，共 {len(df)} 条记录")
        return df
    except Exception as e:
        logger.error(f"数据加载失败: {str(e)}")
        raise


def basic_statistics(df):
    """基本统计分析"""
    logger.info("进行基本统计分析")

    # 选择数值型特征
    numeric_features = [col for col in FEATURES_TO_ANALYZE if col in df.columns]

    # 创建副本以避免修改原始数据
    df_numeric = df[numeric_features].copy()

    # 转换为数值类型，将非数值内容如"*"替换为NaN
    for col in numeric_features:
        df_numeric[col] = pd.to_numeric(df_numeric[col], errors='coerce')

    # 计算基本统计量
    stats = df_numeric.describe().transpose()

    # 计算其他统计量
    stats['missing'] = df_numeric.isnull().sum()
    stats['missing_pct'] = df_numeric.isnull().sum() / len(df) * 100
    stats['skewness'] = df_numeric.skew()
    stats['kurtosis'] = df_numeric.kurtosis()

    # 保存统计结果
    stats_path = os.path.join(FIGURES_DIR, 'basic_statistics.csv')
    stats.to_csv(stats_path, encoding='utf-8')
    logger.info(f"基本统计结果已保存至: {stats_path}")

    return stats


def visualize_distributions(df):
    """
    数据分布可视化

    Args:
        df: 清洗后的数据
    """
    logger.info("进行数据分布可视化")

    # 选择数值型特征
    numeric_features = [col for col in FEATURES_TO_ANALYZE if col in df.columns]

    # 创建直方图和箱线图
    for feature in tqdm(numeric_features, desc="绘制分布图"):
        # 获取特征的英文名称
        english_feature = get_english_feature_name(feature)

        # 创建一个两列子图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # 直方图 + 密度图
        sns.histplot(df[feature].dropna(), kde=True, ax=ax1)
        ax1.set_title(f'{english_feature} Distribution Histogram')
        ax1.set_xlabel(english_feature)
        ax1.set_ylabel('Frequency')

        # 箱线图
        sns.boxplot(y=df[feature].dropna(), ax=ax2)
        ax2.set_title(f'{english_feature} Box Plot')
        ax2.set_ylabel(english_feature)

        # 调整布局
        plt.tight_layout()

        # 保存图像
        safe_feature = feature.replace("/", "_").replace("\\", "_").replace(":", "_")
        dist_path = os.path.join(FIGURES_DIR, f'distribution_{safe_feature}.png')
        plt.savefig(dist_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

    logger.info(f"分布可视化图表已保存至: {FIGURES_DIR}")


def visualize_time_series(df, top_n_stations=5):
    """
    时间序列可视化（优化版）

    Args:
        df: 清洗后的数据
        top_n_stations: 选择数据量最多的前N个站点进行可视化
    """
    logger.info("进行时间序列可视化")

    # 检查必要的列是否存在
    if not all(col in df.columns for col in ['断面名称', '监测时间']):
        logger.error("缺少必要的列：'断面名称'或'监测时间'")
        return

    # 确保日期列是datetime类型
    if not pd.api.types.is_datetime64_dtype(df['监测时间']):
        df['监测时间'] = pd.to_datetime(df['监测时间'], errors='coerce')

    # 统计各站点的数据量
    station_counts = df['断面名称'].value_counts()

    # 选择数据量最多的前N个站点
    top_stations = station_counts.head(top_n_stations).index.tolist()
    logger.info(f"选择了数据量最多的 {len(top_stations)} 个站点进行时间序列可视化")

    # 选择需要可视化的特征
    features_to_plot = [col for col in FEATURES_TO_ANALYZE if col in df.columns]

    # 按站点分别绘制时间序列图
    for station in tqdm(top_stations, desc="绘制站点时间序列"):
        # 获取站点英文名称
        english_station = get_english_station_name(station)

        # 获取站点数据
        station_data = df[df['断面名称'] == station].copy()

        # 按时间排序
        station_data = station_data.sort_values('监测时间')

        # 为每个特征创建单独的时间序列图
        for feature in features_to_plot:
            # 获取特征英文名称
            english_feature = get_english_feature_name(feature)

            # 跳过含有非数值数据的特征
            if not pd.api.types.is_numeric_dtype(station_data[feature]):
                continue

            fig, ax = plt.subplots(figsize=(10, 6))

            # 移除包含NaN的行
            valid_data = station_data[['监测时间', feature]].dropna()

            # 移除异常值以获得更好的视觉效果
            if len(valid_data) > 10:  # 确保有足够的数据点
                Q1 = valid_data[feature].quantile(0.05)
                Q3 = valid_data[feature].quantile(0.95)
                IQR = Q3 - Q1
                valid_data = valid_data[(valid_data[feature] >= Q1 - 1.5 * IQR) &
                                        (valid_data[feature] <= Q3 + 1.5 * IQR)]

            # 绘制时间序列，使用较小的标记和更透明的线条
            ax.plot(valid_data['监测时间'], valid_data[feature],
                    marker='o', markersize=4, linestyle='-', linewidth=1.5, alpha=0.7)

            # 使用更好的时间轴格式
            locator = mdates.AutoDateLocator(minticks=5, maxticks=10)
            formatter = mdates.ConciseDateFormatter(locator)
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(formatter)

            # 添加网格线提高可读性
            ax.grid(True, linestyle='--', alpha=0.3)

            # 设置标题和标签，使用更清晰的字体
            plt.title(f'{english_station} - {english_feature} Time Series', fontsize=14)
            plt.xlabel('Time', fontsize=12)
            plt.ylabel(english_feature, fontsize=12)
            plt.xticks(rotation=30, ha='right')

            # 添加均值和趋势线
            if len(valid_data) > 5:
                # 均值线
                mean_val = valid_data[feature].mean()
                plt.axhline(y=mean_val, color='r', linestyle='--', alpha=0.5)
                plt.text(valid_data['监测时间'].iloc[0], mean_val, f'Mean: {mean_val:.2f}',
                         va='bottom', ha='left', color='r', fontsize=10)

                # 趋势线
                try:
                    z = np.polyfit(np.arange(len(valid_data)), valid_data[feature], 1)
                    p = np.poly1d(z)
                    plt.plot(valid_data['监测时间'], p(np.arange(len(valid_data))),
                             "r--", alpha=0.5)
                except:
                    pass  # 跳过趋势线如果拟合失败

            plt.tight_layout()

            # 保存图像，使用更高的DPI获得更清晰的图像
            safe_station = station.replace('/', '_').replace('\\', '_').replace(':', '_')
            safe_feature = feature.replace('/', '_').replace('\\', '_').replace(':', '_')
            ts_path = os.path.join(FIGURES_DIR, f'timeseries_{safe_station}_{safe_feature}.png')
            plt.savefig(ts_path, dpi=300, bbox_inches='tight')
            plt.close(fig)

    logger.info(f"时间序列可视化图表已保存至: {FIGURES_DIR}")


def correlation_analysis(df):
    """特征相关性分析（优化版）"""
    logger.info("进行特征相关性分析")

    # 选择数值型特征
    numeric_features = [col for col in FEATURES_TO_ANALYZE if col in df.columns]
    english_features = [get_english_feature_name(col) for col in numeric_features]

    # 创建副本以避免修改原始数据
    df_numeric = df[numeric_features].copy()

    # 转换为数值类型，将非数值内容如"*"替换为NaN
    for col in numeric_features:
        df_numeric[col] = pd.to_numeric(df_numeric[col], errors='coerce')

    # 计算相关系数矩阵
    corr_matrix = df_numeric.corr()

    # 更改索引和列名为英文
    corr_matrix_eng = pd.DataFrame(corr_matrix.values,
                                   index=english_features,
                                   columns=english_features)

    # 保存相关系数矩阵
    corr_path = os.path.join(FIGURES_DIR, 'correlation_matrix.csv')
    corr_matrix_eng.to_csv(corr_path, encoding='utf-8')

    # 绘制相关性热图
    plt.figure(figsize=(12, 10))

    # 使用更好的配色方案
    mask = np.triu(np.ones_like(corr_matrix_eng, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    # 绘制热图，使用更好的注释格式
    sns.heatmap(corr_matrix_eng, mask=mask, annot=True, fmt=".2f",
                cmap=cmap, square=True, linewidths=.5,
                cbar_kws={"shrink": .5})

    # 更清晰的标题和轴标签
    plt.title('Feature Correlation Heatmap', fontsize=16, pad=20)
    plt.tight_layout()

    # 保存热图
    heatmap_path = os.path.join(FIGURES_DIR, 'correlation_heatmap.png')
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"相关性分析结果已保存至: {FIGURES_DIR}")


def time_series_acf_pacf_analysis(df, top_n_stations=3):
    """
    时间序列ACF和PACF分析（优化版）

    Args:
        df: 清洗后的数据
        top_n_stations: 选择数据量最多的前N个站点进行分析
    """
    logger.info("进行时间序列ACF和PACF分析")

    # 导入必要的库
    import matplotlib.dates as mdates
    from statsmodels.tsa.stattools import adfuller
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

    # 检查必要的列是否存在
    if not all(col in df.columns for col in ['断面名称', '监测时间']):
        logger.error("缺少必要的列：'断面名称'或'监测时间'")
        return

    # 统计各站点的数据量
    station_counts = df['断面名称'].value_counts()

    # 选择数据量最多的前N个站点
    top_stations = station_counts.head(top_n_stations).index.tolist()
    logger.info(f"选择了数据量最多的 {len(top_stations)} 个站点进行ACF和PACF分析")

    # 选择需要分析的特征，排除已知问题特征
    features_for_analysis = [col for col in FEATURES_TO_ANALYZE
                             if col in df.columns and col not in ['叶绿素α(mg/L)', '藻密度(cells/L)']]

    # 创建ACF和PACF分析结果存储
    acf_pacf_results = {}

    # 按站点进行分析
    for station in tqdm(top_stations, desc="站点ACF和PACF分析"):
        # 获取站点英文名称
        english_station = get_english_station_name(station)

        # 获取站点数据
        station_data = df[df['断面名称'] == station].copy()

        # 按时间排序
        station_data = station_data.sort_values('监测时间')

        # 对每个特征进行分析
        for feature in features_for_analysis:
            # 获取特征英文名称
            english_feature = get_english_feature_name(feature)

            # 提取特征时间序列，并确保时间序列是等距的
            ts = station_data[feature].dropna()

            # 检查数据点是否足够
            if len(ts) < 20:  # 数据点太少，跳过
                logger.warning(f"站点 {station} 的特征 {feature} 数据点不足，跳过ACF和PACF分析")
                continue

            try:
                # 创建更清晰的图像布局
                fig = plt.figure(figsize=(12, 10))
                gs = GridSpec(3, 1, height_ratios=[2, 1, 1])

                # 原始时间序列图
                ax1 = fig.add_subplot(gs[0])
                ax1.plot(station_data['监测时间'], station_data[feature],
                         marker='o', markersize=4, linestyle='-', alpha=0.7)

                # 设置更好的时间轴格式
                locator = mdates.AutoDateLocator(minticks=5, maxticks=10)
                formatter = mdates.ConciseDateFormatter(locator)
                ax1.xaxis.set_major_locator(locator)
                ax1.xaxis.set_major_formatter(formatter)

                ax1.set_title(f'{english_station} - {english_feature} Time Series', fontsize=14)
                ax1.set_xlabel('Time', fontsize=12)
                ax1.set_ylabel(english_feature, fontsize=12)
                ax1.grid(True, linestyle='--', alpha=0.3)

                # 进行 ADF 平稳性检验
                try:
                    adf_result = adfuller(ts)
                    adf_p_value = adf_result[1]
                    is_stationary = adf_p_value < 0.05
                    stationarity_text = f"ADF Test p-value: {adf_p_value:.4f}\nStationarity: {'Stationary' if is_stationary else 'Non-stationary'}"

                    # 使用更明显的文本框显示结果
                    ax1.text(0.05, 0.95, stationarity_text, transform=ax1.transAxes,
                             verticalalignment='top',
                             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                             fontsize=10)
                except Exception as e:
                    logger.warning(f"站点 {station} 的特征 {feature} ADF测试失败: {str(e)}")

                # 绘制ACF图，使用更多的色彩对比
                ax2 = fig.add_subplot(gs[1])
                try:
                    plot_acf(ts, ax=ax2, lags=min(40, len(ts) // 2), alpha=0.05, title='')
                    ax2.set_title(f'Autocorrelation Function (ACF) - {english_feature}', fontsize=12)
                    ax2.grid(True, linestyle='--', alpha=0.3)
                except Exception as e:
                    logger.warning(f"站点 {station} 的特征 {feature} ACF分析失败: {str(e)}")
                    ax2.text(0.5, 0.5, f"ACF analysis failed: {str(e)}", ha='center', va='center',
                             transform=ax2.transAxes)

                # 绘制PACF图
                ax3 = fig.add_subplot(gs[2])
                try:
                    plot_pacf(ts, ax=ax3, lags=min(40, len(ts) // 2), alpha=0.05, method='ywm', title='')
                    ax3.set_title(f'Partial Autocorrelation Function (PACF) - {english_feature}', fontsize=12)
                    ax3.grid(True, linestyle='--', alpha=0.3)
                except Exception as e:
                    logger.warning(f"站点 {station} 的特征 {feature} PACF分析失败: {str(e)}")
                    ax3.text(0.5, 0.5, f"PACF analysis failed: {str(e)}", ha='center', va='center',
                             transform=ax3.transAxes)

                # 添加建议的ARIMA参数
                if 'is_stationary' in locals() and is_stationary:
                    # 简单地基于ACF和PACF图提供ARIMA参数建议
                    ax3.text(0.05, 0.05, "Suggested ARIMA parameters:\np = 1-3, d = 0, q = 1-2",
                             transform=ax3.transAxes,
                             verticalalignment='bottom',
                             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8),
                             fontsize=10)
                elif 'is_stationary' in locals():
                    ax3.text(0.05, 0.05, "Suggested ARIMA parameters:\np = 1-2, d = 1, q = 1", transform=ax3.transAxes,
                             verticalalignment='bottom',
                             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                             fontsize=10)

                # 调整布局
                plt.tight_layout()

                # 保存图像
                safe_station = station.replace('/', '_').replace('\\', '_').replace(':', '_')
                safe_feature = feature.replace('/', '_').replace('\\', '_').replace(':', '_')
                acf_pacf_path = os.path.join(FIGURES_DIR, f'acf_pacf_{safe_station}_{safe_feature}.png')
                plt.savefig(acf_pacf_path, dpi=300, bbox_inches='tight')
                plt.close(fig)

                # 存储分析结果
                key = f"{station}_{feature}"
                acf_pacf_results[key] = {
                    'station': station,
                    'english_station': english_station,
                    'feature': feature,
                    'english_feature': english_feature,
                    'is_stationary': is_stationary if 'is_stationary' in locals() else None,
                    'adf_p_value': adf_p_value if 'adf_p_value' in locals() else None
                }

            except Exception as e:
                logger.error(f"站点 {station} 的特征 {feature} 分析失败: {str(e)}")

    # 保存ACF和PACF分析结果
    acf_pacf_df = pd.DataFrame.from_dict(acf_pacf_results, orient='index')
    acf_pacf_results_path = os.path.join(FIGURES_DIR, 'acf_pacf_analysis_results.csv')
    acf_pacf_df.to_csv(acf_pacf_results_path, encoding='utf-8')

    logger.info(f"ACF和PACF分析结果已保存至: {FIGURES_DIR}")
    return acf_pacf_results


def analyze_station_data_availability(df):
    """
    分析各站点数据可用性

    Args:
        df: 清洗后的数据
    """
    logger.info("分析各站点数据可用性")

    # 检查必要的列是否存在
    if not all(col in df.columns for col in ['断面名称', '监测时间']):
        logger.error("缺少必要的列：'断面名称'或'监测时间'")
        return

    # 计算各站点的数据记录数
    station_counts = df['断面名称'].value_counts().reset_index()
    station_counts.columns = ['断面名称', '记录数']

    # 添加英文站点名
    station_counts['英文站点名'] = station_counts['断面名称'].apply(get_english_station_name)

    # 按记录数降序排序
    station_counts = station_counts.sort_values('记录数', ascending=False)

    # 保存结果
    station_counts_path = os.path.join(FIGURES_DIR, 'station_data_counts.csv')
    station_counts.to_csv(station_counts_path, index=False, encoding='utf-8')

    # 可视化前20个站点的数据量
    plt.figure(figsize=(14, 8))
    top_20_stations = station_counts.head(20)
    # 使用英文站点名绘图
    sns.barplot(x='记录数', y='英文站点名', data=top_20_stations)
    plt.title('Top 20 Stations by Data Count')
    plt.xlabel('Record Count')
    plt.ylabel('Station')
    plt.tight_layout()

    # 保存图表
    top_stations_path = os.path.join(FIGURES_DIR, 'top_20_stations_data_count.png')
    plt.savefig(top_stations_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"站点数据可用性分析结果已保存至: {FIGURES_DIR}")

    return station_counts


def main():
    """主函数"""
    logger.info("开始探索性数据分析")

    # 确保输出目录存在
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # 加载清洗后的数据
    df = load_clean_data()

    # 基本统计分析
    stats = basic_statistics(df)

    # 数据分布可视化
    visualize_distributions(df)

    # 分析各站点数据可用性
    station_counts = analyze_station_data_availability(df)

    # 时间序列可视化（使用数据量最多的前5个站点）
    visualize_time_series(df, top_n_stations=5)

    # 相关性分析
    correlation_analysis(df)

    # ACF和PACF分析（使用数据量最多的前3个站点）
    acf_pacf_results = time_series_acf_pacf_analysis(df, top_n_stations=3)

    logger.info("探索性数据分析完成")

    return {
        'stats': stats,
        'station_counts': station_counts,
        'acf_pacf_results': acf_pacf_results
    }


if __name__ == "__main__":
    main()