#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
水质预测系统界面模块
使用PyQt5实现图形用户界面，用于:
1. 查看不同断面的水质监测数据
2. 展示不同模型的评估指标
3. 可视化预测结果
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QSizePolicy
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import seaborn as sns
from PyQt5.QtWidgets import (QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QComboBox, QPushButton, QTableWidget,
                             QTableWidgetItem, QHeaderView, QSplitter, QFileDialog,
                             QMessageBox, QGridLayout)
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QIcon, QFont
import glob
import logging
import json
import warnings

# 忽略警告
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("log/ui.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 定义常量
PROCESSED_DATA_DIR = 'data/processed/'
FIGURES_DIR = 'results/figures/'
METRICS_DIR = 'results/metrics/'
MODELS_SAVE_DIR = 'models/saved/'


class MatplotlibCanvas(FigureCanvas):
    """Matplotlib画布类，用于显示图表"""

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        """
        初始化画布

        Args:
            parent: 父级QWidget
            width: 宽度(英寸)
            height: 高度(英寸)
            dpi: 分辨率
        """
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax = self.fig.add_subplot(111)
        super(MatplotlibCanvas, self).__init__(self.fig)
        self.setParent(parent)

        # 自动调整图表大小
        FigureCanvas.setSizePolicy(self,
                                   QSizePolicy.Expanding,
                                   QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def clear_plot(self):
        """清除图表"""
        self.ax.clear()
        self.draw()

    def plot_time_series(self, df, feature_name, title="", show_legend=True):
        """
        绘制时间序列数据

        Args:
            df: 包含时间序列数据的DataFrame
            feature_name: 要绘制的特征名称
            title: 图表标题
            show_legend: 是否显示图例
        """
        self.ax.clear()
        if feature_name in df.columns:
            # 确保时间索引
            if not isinstance(df.index, pd.DatetimeIndex):
                # 如果有监测时间列，将其设为索引
                if '监测时间' in df.columns:
                    df = df.set_index('监测时间')

            # 绘制时间序列
            self.ax.plot(df.index, df[feature_name], 'o-', label=feature_name)

            # 设置标题和标签
            self.ax.set_title(title or f'{feature_name}时间序列')
            self.ax.set_xlabel('时间')
            self.ax.set_ylabel(feature_name)

            # 设置x轴日期格式
            self.fig.autofmt_xdate()

            if show_legend:
                self.ax.legend()

            self.ax.grid(True, linestyle='--', alpha=0.5)

        else:
            self.ax.text(0.5, 0.5, f"数据中不存在特征: {feature_name}",
                         ha='center', va='center', transform=self.ax.transAxes)

        self.fig.tight_layout()
        self.draw()

    def plot_prediction_comparison(self, train_data, test_data, predictions, model_name, feature_name):
        """
        绘制预测结果对比图

        Args:
            train_data: 训练数据
            test_data: 测试数据
            predictions: 预测结果
            model_name: 模型名称
            feature_name: 特征名称
        """
        self.ax.clear()

        # 绘制训练数据
        self.ax.plot(train_data.index, train_data, 'b-', label='训练数据', alpha=0.7)

        # 绘制测试数据
        self.ax.plot(test_data.index, test_data, 'g-', label='测试数据', alpha=0.7)

        # 绘制预测结果 (可能需要调整索引，取决于预测结果的格式)
        if isinstance(predictions, pd.Series):
            self.ax.plot(predictions.index, predictions, 'r--', label='预测结果')
        else:
            # 假设预测结果是与测试数据对应的数组
            self.ax.plot(test_data.index[-len(predictions):], predictions, 'r--', label='预测结果')

        # 设置标题和标签
        self.ax.set_title(f'{model_name} - {feature_name} 预测结果')
        self.ax.set_xlabel('时间')
        self.ax.set_ylabel(feature_name)

        # 自动旋转日期标签以避免重叠
        self.fig.autofmt_xdate()

        self.ax.legend()
        self.ax.grid(True, linestyle='--', alpha=0.5)

        self.fig.tight_layout()
        self.draw()

    def plot_evaluation_metrics(self, metrics_df, model_type=None):
        """
        绘制评估指标比较图

        Args:
            metrics_df: 包含评估指标的DataFrame
            model_type: 模型类型筛选，如果为None则显示所有模型
        """
        self.ax.clear()

        # 如果指定了模型类型，则进行筛选
        if model_type and 'model_type' in metrics_df.columns:
            df = metrics_df[metrics_df['model_type'] == model_type]
        else:
            df = metrics_df

        # 选择需要显示的指标
        metrics_to_plot = ['RMSE', 'MAE', 'MAPE', 'R2']
        metrics_exists = [m for m in metrics_to_plot if m in df.columns]

        if not metrics_exists:
            self.ax.text(0.5, 0.5, "没有可用的评估指标数据",
                         ha='center', va='center', transform=self.ax.transAxes)
            self.draw()
            return

        # 如果有站点和特征信息，创建标签
        if 'station' in df.columns and 'feature' in df.columns:
            labels = df.apply(lambda row: f"{row['station']}-{row['feature']}", axis=1)
        else:
            labels = df.index

        # 绘制评估指标条形图
        x = np.arange(len(labels))
        width = 0.2

        for i, metric in enumerate(metrics_exists):
            if metric in df.columns:
                offset = width * (i - len(metrics_exists) / 2 + 0.5)
                self.ax.bar(x + offset, df[metric], width, label=metric)

        # 设置x轴标签
        self.ax.set_xticks(x)
        self.ax.set_xticklabels(labels, rotation=45, ha='right')

        # 设置标题和标签
        self.ax.set_title(f"模型评估指标对比 {model_type or ''}")
        self.ax.set_ylabel('指标值')

        self.ax.legend()
        self.ax.grid(True, linestyle='--', alpha=0.5)

        self.fig.tight_layout()
        self.draw()


class WaterQualityPredictionUI(QMainWindow):
    """水质预测系统界面类"""

    def __init__(self):
        """初始化界面"""
        super().__init__()

        # 设置窗口标题和大小
        self.setWindowTitle("水质预测系统")
        self.setGeometry(100, 100, 1200, 800)

        # 加载数据和模型
        self.load_data()
        self.load_metrics()

        # 初始化界面
        self.init_ui()

    def load_data(self):
        """加载处理后的数据"""
        try:
            logger.info("加载数据...")

            # 存储各集合数据
            self.split_data = {
                'train': {},
                'val': {},
                'test': {}
            }

            # 加载各集合数据
            for split_type in self.split_data.keys():
                split_dir = os.path.join(PROCESSED_DATA_DIR, split_type)

                # 检查目录是否存在
                if os.path.exists(split_dir):
                    # 获取目录中的所有CSV文件
                    csv_files = glob.glob(os.path.join(split_dir, "*.csv"))

                    # 加载每个站点的数据
                    for csv_file in csv_files:
                        # 从文件名获取站点名称
                        station_filename = os.path.basename(csv_file)
                        station_name = os.path.splitext(station_filename)[0].replace('_', '/')

                        # 加载数据
                        try:
                            df = pd.read_csv(csv_file, encoding='utf-8')
                            # 如果有监测时间列，将其转换为datetime
                            if '监测时间' in df.columns:
                                df['监测时间'] = pd.to_datetime(df['监测时间'])
                                df.set_index('监测时间', inplace=True)

                            # 存储数据
                            self.split_data[split_type][station_name] = df
                        except Exception as e:
                            logger.error(f"加载文件失败 {csv_file}: {str(e)}")

            # 提取站点和特征列表
            self.stations = list(self.split_data['train'].keys())

            if self.stations:
                sample_df = next(iter(self.split_data['train'].values()))
                self.features = list(sample_df.columns)
            else:
                self.features = []

            logger.info(f"数据加载完成，共 {len(self.stations)} 个站点")

        except Exception as e:
            logger.error(f"数据加载失败: {str(e)}")
            QMessageBox.critical(self, "错误", f"数据加载失败: {str(e)}")

    def load_metrics(self):
        """加载模型评估指标"""
        try:
            logger.info("加载模型评估指标...")

            # 存储各模型的评估指标
            self.metrics_data = {}

            # 查找并加载评估指标文件
            metrics_files = glob.glob(os.path.join(METRICS_DIR, "*.csv"))

            for metrics_file in metrics_files:
                model_name = os.path.splitext(os.path.basename(metrics_file))[0]

                try:
                    metrics_df = pd.read_csv(metrics_file, encoding='utf-8')
                    self.metrics_data[model_name] = metrics_df
                except Exception as e:
                    logger.error(f"加载评估指标文件失败 {metrics_file}: {str(e)}")

            # 提取模型类型列表
            self.model_types = list(self.metrics_data.keys())

            logger.info(f"评估指标加载完成，共 {len(self.model_types)} 种模型")

        except Exception as e:
            logger.error(f"评估指标加载失败: {str(e)}")
            QMessageBox.critical(self, "错误", f"评估指标加载失败: {str(e)}")

    def init_ui(self):
        """初始化用户界面"""
        # 创建中央窗口部件
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        # 创建主布局
        main_layout = QVBoxLayout(central_widget)

        # 创建选择面板
        selection_panel = QWidget()
        selection_layout = QHBoxLayout(selection_panel)

        # 站点选择
        station_label = QLabel("断面名称:")
        self.station_combo = QComboBox()
        self.station_combo.addItems(self.stations)
        self.station_combo.currentIndexChanged.connect(self.update_feature_combo)

        # 特征选择
        feature_label = QLabel("水质指标:")
        self.feature_combo = QComboBox()
        if self.features:
            self.feature_combo.addItems(self.features)
        self.feature_combo.currentIndexChanged.connect(self.update_plots)

        # 模型选择
        model_label = QLabel("模型类型:")
        self.model_combo = QComboBox()
        self.model_combo.addItems(self.model_types)
        self.model_combo.currentIndexChanged.connect(self.update_plots)

        # 添加到选择布局
        selection_layout.addWidget(station_label)
        selection_layout.addWidget(self.station_combo)
        selection_layout.addWidget(feature_label)
        selection_layout.addWidget(self.feature_combo)
        selection_layout.addWidget(model_label)
        selection_layout.addWidget(self.model_combo)

        # 添加选择面板到主布局
        main_layout.addWidget(selection_panel)

        # 创建标签页
        self.tab_widget = QTabWidget()

        # 数据可视化标签页
        self.data_tab = QWidget()
        data_layout = QVBoxLayout(self.data_tab)

        # 创建数据图表
        self.data_canvas = MatplotlibCanvas(self.data_tab, width=12, height=6)
        data_layout.addWidget(self.data_canvas)

        # 预测结果标签页
        self.prediction_tab = QWidget()
        prediction_layout = QVBoxLayout(self.prediction_tab)

        # 创建预测图表
        self.prediction_canvas = MatplotlibCanvas(self.prediction_tab, width=12, height=6)
        prediction_layout.addWidget(self.prediction_canvas)

        # 评估指标标签页
        self.evaluation_tab = QWidget()
        evaluation_layout = QVBoxLayout(self.evaluation_tab)

        # 创建评估指标表格和图表
        evaluation_splitter = QSplitter(Qt.Vertical)

        self.metrics_table = QTableWidget()
        self.metrics_table.setAlternatingRowColors(True)
        self.metrics_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        self.metrics_canvas = MatplotlibCanvas(self.evaluation_tab, width=12, height=6)

        evaluation_splitter.addWidget(self.metrics_table)
        evaluation_splitter.addWidget(self.metrics_canvas)

        evaluation_layout.addWidget(evaluation_splitter)

        # 添加标签页到标签页控件
        self.tab_widget.addTab(self.data_tab, "数据可视化")
        self.tab_widget.addTab(self.prediction_tab, "预测结果")
        self.tab_widget.addTab(self.evaluation_tab, "评估指标")

        # 添加标签页控件到主布局
        main_layout.addWidget(self.tab_widget)

        # 初始化显示
        if self.stations and self.features:
            self.update_plots()

    def update_feature_combo(self):
        """更新特征下拉框"""
        selected_station = self.station_combo.currentText()

        if selected_station in self.split_data['train']:
            # 获取所选站点的特征
            df = self.split_data['train'][selected_station]
            features = list(df.columns)

            # 清空并重新填充特征下拉框
            self.feature_combo.clear()
            self.feature_combo.addItems(features)

    def update_plots(self):
        """更新所有图表"""
        # 获取当前选择
        selected_station = self.station_combo.currentText()
        if not selected_station or self.feature_combo.count() == 0:
            return

        selected_feature = self.feature_combo.currentText()
        selected_model = self.model_combo.currentText()

        # 更新数据可视化
        self.update_data_visualization(selected_station, selected_feature)

        # 更新预测结果
        self.update_prediction_plot(selected_station, selected_feature, selected_model)

        # 更新评估指标
        self.update_evaluation_metrics(selected_station, selected_feature, selected_model)

    def update_data_visualization(self, station, feature):
        """
        更新数据可视化图表

        Args:
            station: 所选断面名称
            feature: 所选特征名称
        """
        # 检查数据是否存在
        if station in self.split_data['train'] and feature in self.split_data['train'][station].columns:
            # 获取站点数据
            train_data = self.split_data['train'][station]

            # 绘制时间序列
            self.data_canvas.plot_time_series(
                train_data,
                feature,
                title=f"{station} - {feature} 时间序列"
            )

    def update_prediction_plot(self, station, feature, model_type):
        """
        更新预测结果图表

        Args:
            station: 所选断面名称
            feature: 所选特征名称
            model_type: 所选模型类型
        """
        # 首先检查是否有预测结果可用
        # 这里我们可以尝试加载对应模型的预测结果文件
        safe_station = station.replace('/', '_').replace('\\', '_').replace(':', '_')
        safe_feature = feature.replace('/', '_').replace('\\', '_').replace(':', '_')

        prediction_file = os.path.join(FIGURES_DIR, f'{model_type}_{safe_station}_{safe_feature}_forecast.png')

        if os.path.exists(prediction_file):
            # 如果有预测结果图像，可以直接显示
            img = plt.imread(prediction_file)
            self.prediction_canvas.ax.clear()
            self.prediction_canvas.ax.imshow(img)
            self.prediction_canvas.ax.axis('off')
            self.prediction_canvas.draw()
        else:
            # 如果没有现成的预测结果，可以尝试组合训练、测试数据展示
            if (station in self.split_data['train'] and station in self.split_data['test'] and
                    feature in self.split_data['train'][station].columns and
                    feature in self.split_data['test'][station].columns):

                train_data = self.split_data['train'][station][feature]
                test_data = self.split_data['test'][station][feature]

                # 这里简单地使用测试数据作为"预测"，实际应用中应该加载真正的预测结果
                predictions = test_data.values

                self.prediction_canvas.plot_prediction_comparison(
                    train_data,
                    test_data,
                    predictions,
                    model_type,
                    feature
                )
            else:
                self.prediction_canvas.ax.clear()
                self.prediction_canvas.ax.text(0.5, 0.5, "没有找到预测结果",
                                               ha='center', va='center', transform=self.prediction_canvas.ax.transAxes)
                self.prediction_canvas.draw()

    def update_evaluation_metrics(self, station, feature, model_type):
        """
        更新评估指标表格和图表

        Args:
            station: 所选断面名称
            feature: 所选特征名称
            model_type: 所选模型类型
        """
        # 检查是否有评估指标数据
        if model_type in self.metrics_data:
            metrics_df = self.metrics_data[model_type]

            # 筛选当前站点和特征的评估指标
            if 'station' in metrics_df.columns and 'feature' in metrics_df.columns:
                filtered_df = metrics_df[(metrics_df['station'] == station) &
                                         (metrics_df['feature'] == feature)]
            else:
                filtered_df = metrics_df

            # 更新表格
            self.update_metrics_table(filtered_df)

            # 更新图表 - 这里可以展示所有模型在当前站点和特征上的对比
            combined_metrics = []

            for model, df in self.metrics_data.items():
                if 'station' in df.columns and 'feature' in df.columns:
                    model_df = df[(df['station'] == station) & (df['feature'] == feature)].copy()
                    if not model_df.empty:
                        model_df['model_type'] = model
                        combined_metrics.append(model_df)

            if combined_metrics:
                combined_df = pd.concat(combined_metrics, ignore_index=True)
                self.metrics_canvas.plot_evaluation_metrics(combined_df)
            else:
                self.metrics_canvas.ax.clear()
                self.metrics_canvas.ax.text(0.5, 0.5, "没有找到评估指标数据",
                                            ha='center', va='center', transform=self.metrics_canvas.ax.transAxes)
                self.metrics_canvas.draw()

    def update_metrics_table(self, metrics_df):
        """
        更新评估指标表格

        Args:
            metrics_df: 包含评估指标的DataFrame
        """
        # 清空表格
        self.metrics_table.clear()

        if metrics_df.empty:
            self.metrics_table.setRowCount(0)
            self.metrics_table.setColumnCount(0)
            return

        # 设置列数和列名
        self.metrics_table.setColumnCount(len(metrics_df.columns))
        self.metrics_table.setHorizontalHeaderLabels(metrics_df.columns)

        # 设置行数
        self.metrics_table.setRowCount(len(metrics_df))

        # 填充数据
        for i, row in metrics_df.iterrows():
            for j, value in enumerate(row):
                item = QTableWidgetItem(str(value))
                self.metrics_table.setItem(i if i < len(metrics_df) else i - len(metrics_df), j, item)


def main():
    """主函数"""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # 使用Fusion样式，更现代的外观

    # 设置字体
    font = QFont()
    font.setPointSize(10)
    app.setFont(font)

    # 创建并显示主窗口
    window = WaterQualityPredictionUI()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()