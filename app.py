import base64
import io
import os
import logging # 引入日志模块

import matplotlib
matplotlib.use('Agg') # 为无图形界面的环境使用 Agg 后端
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from flask import Flask, render_template, request, redirect, url_for
from scipy.special import expit as sigmoid # Sigmoid 函数，即 logistic 函数
from sklearn.metrics import classification_report, f1_score
from sklearn.utils.class_weight import compute_sample_weight # 计算样本权重

# Flask 应用初始化
# 默认情况下，Flask 会在与 app.py 文件同级的名为 'templates' 的文件夹中查找模板。
app = Flask(__name__)
app.secret_key = os.urandom(24) # 用于会话管理

# 配置日志
logging.basicConfig(level=logging.INFO) # 设置日志级别为 INFO

# 全局缓存，用于存储模型和数据
model_cache = {}

# Matplotlib 字体设置
try:
    matplotlib.rcParams['font.family'] = 'sans-serif' # 更通用的字体
    matplotlib.rcParams['axes.unicode_minus'] = False
except Exception as e:
    app.logger.warning(f"Matplotlib 字体设置警告: {e}")

def fig_to_base64(fig):
    """将 Matplotlib 图像转换为 base64 编码的 PNG。"""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def generate_shap_waterfall_base64(shap_values_single_instance):
    """生成 SHAP waterfall 图并以 base64 格式返回。"""
    fig = plt.figure()
    shap.plots.waterfall(shap_values_single_instance, show=False, max_display=15)
    plt.tight_layout()
    base64_str = fig_to_base64(fig)
    return base64_str

@app.route('/', methods=['GET', 'POST'])
def index():
    global model_cache
    app.logger.info(f"访问 '/' 路由，方法: {request.method}")

    if request.method == 'GET':
        # 仅当这不是由 predict_single 重定向回来时才清除缓存
        # 或者在用户明确导航到主页时（比如通过链接或直接输入URL）
        # 检查 model_cache 中是否有 show_results 标志，如果没有，说明是新的会话或直接访问
        if not model_cache.get('show_results_from_upload'):
            app.logger.info("GET 请求到 '/', 清空 model_cache (可能是新会话或直接访问)")
            model_cache = {}
        else:
            app.logger.info("GET 请求到 '/', 保留 model_cache (可能从 predict_single 重定向)")
            # 清除 predict_single 的特定结果，但保留上传和训练的结果
            model_cache.pop('show_single_pred_results', None)
            model_cache.pop('single_pred_input_data_html', None)
            model_cache.pop('single_pred_prob', None)
            model_cache.pop('single_pred_label', None)
            model_cache.pop('single_pred_best_thresh', None)
            model_cache.pop('single_pred_shap_table_html', None)
            model_cache.pop('base64_waterfall_plot', None)
            model_cache.pop('single_pred_error', None)
            # default_values 应保留为上次上传数据的均值或上次预测的输入

        # 确保即使 model_cache 为空，模板也能获取到所有预期的变量（值为 None 或默认值）
        template_vars = {
            'show_results': model_cache.get('show_results_from_upload', False), # 主结果区基于上传
            'show_single_pred_results': model_cache.get('show_single_pred_results', False),
            'filename': model_cache.get('filename'),
            'form_inputs': model_cache.get('form_inputs'),
            'base64_perf_plot': model_cache.get('base64_perf_plot'),
            'metrics_df_html': model_cache.get('metrics_df_html'),
            'best_recommendation_html': model_cache.get('best_recommendation_html'),
            'recommended_threshold_text': model_cache.get('recommended_threshold_text'),
            'base64_global_shap_plot': model_cache.get('base64_global_shap_plot'),
            'default_values': model_cache.get('default_values'),
            'single_pred_input_data_html': model_cache.get('single_pred_input_data_html'),
            'single_pred_prob': model_cache.get('single_pred_prob'),
            'single_pred_label': model_cache.get('single_pred_label'),
            'single_pred_best_thresh': model_cache.get('single_pred_best_thresh'),
            'single_pred_shap_table_html': model_cache.get('single_pred_shap_table_html'),
            'base64_waterfall_plot': model_cache.get('base64_waterfall_plot'),
            'error_message': model_cache.get('error_message'), # 全局错误信息
            'single_pred_error': model_cache.get('single_pred_error') # 单次预测错误信息
        }
        # 渲染前清除一次性消息
        model_cache.pop('error_message', None)
        model_cache.pop('single_pred_error', None)

        return render_template('index.html', **template_vars)


    if request.method == 'POST': # 此 POST 用于文件上传和初始处理
        app.logger.info("POST 请求到 '/', 开始处理上传文件...")
        model_cache = {} # 每次上传新文件时，清空旧的缓存

        if 'file' not in request.files:
            app.logger.error("POST 请求错误: 请求中没有文件部分。")
            model_cache['error_message'] = "请求中没有文件部分。"
            return redirect(url_for('index'))

        file = request.files['file']
        if file.filename == '':
            app.logger.error("POST 请求错误: 未选择文件。")
            model_cache['error_message'] = "未选择文件。"
            return redirect(url_for('index'))

        if file and file.filename.endswith('.csv'):
            try:
                filename = file.filename
                app.logger.info(f"成功接收到文件: {filename}")
                df = pd.read_csv(file)
                model_cache['filename'] = filename

                if "OK_NG" not in df.columns:
                    app.logger.error(f"文件 {filename} 缺少 'OK_NG' 列。")
                    model_cache['error_message'] = "上传的 CSV 文件必须包含 'OK_NG' 列。"
                    return redirect(url_for('index'))
                
                X = df.drop("OK_NG", axis=1).copy()
                try:
                    # 尝试将所有特征列转换为数值型，忽略无法转换的错误，后续SHAP可能处理
                    for col in X.columns:
                        X[col] = pd.to_
