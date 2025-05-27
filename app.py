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
from scipy.special import expit as sigmoid # Sigmoid 函数, 即 logistic 函数
from sklearn.metrics import classification_report, f1_score
from sklearn.utils.class_weight import compute_sample_weight # 计算样本权重

# Flask 应用初始化
app = Flask(__name__)
app.secret_key = os.urandom(24)

# 配置日志
logging.basicConfig(level=logging.INFO)
# 如果使用 Gunicorn, 可以获取其 logger 以统一日志格式，但对于 python app.py 启动不是必需的
# gunicorn_logger = logging.getLogger('gunicorn.error')
# app.logger.handlers.extend(gunicorn_logger.handlers)
app.logger.setLevel(logging.INFO)


# 全局缓存
model_cache = {}

# Matplotlib 字体设置
try:
    matplotlib.rcParams['font.family'] = 'sans-serif'
    matplotlib.rcParams['axes.unicode_minus'] = False
except Exception as e:
    app.logger.warning(f"Matplotlib 字体设置警告: {e}")

def fig_to_base64(fig):
    """将 Matplotlib 图像转换为 base64 编码的 PNG。"""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight')
    plt.close(fig) # 关闭图像, 释放内存
    buf.seek(0) # 将指针移到缓冲区的开头
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def generate_shap_waterfall_base64(shap_values_single_instance):
    """生成 SHAP waterfall 图并以 base64 格式返回。"""
    # shap_values_single_instance 应为单个实例的 shap.Explanation 对象
    fig = plt.figure() # 创建新的图像上下文
    shap.plots.waterfall(shap_values_single_instance, show=False, max_display=15) # max_display 控制显示特征数量
    plt.tight_layout() # 自动调整子图参数, 使之填充整个图像区域
    base64_str = fig_to_base64(fig)
    return base64_str

@app.route('/', methods=['GET', 'POST'])
def index():
    global model_cache
    app.logger.info(f"访问 '/' 路由, 方法: {request.method}")

    if request.method == 'GET':
        # 仅当这不是由其他操作（如 predict_single 或文件上传后的 POST）重定向回来时才考虑清空缓存。
        # 一个简单的判断是检查 model_cache 中是否有标志性数据，如果没有，则认为是全新的访问。
        if not model_cache.get('show_results_from_upload') and not model_cache.get('show_single_pred_results'):
            app.logger.info("GET 请求到 '/', 清空 model_cache (可能是新会话或直接访问)")
            model_cache = {}
        else:
            app.logger.info("GET 请求到 '/', 保留 model_cache (可能从POST或 predict_single 重定向)")
            # 清除一次性的错误消息，这些消息应该只显示一次
            model_cache.pop('error_message_to_display', None)
            model_cache.pop('single_pred_error_to_display', None)


        template_vars = {
            'show_results': model_cache.get('show_results_from_upload', False),
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
            'error_message': model_cache.get('error_message_to_display'), # 从这里获取待显示错误
            'single_pred_error': model_cache.get('single_pred_error_to_display')
        }
        return render_template('index.html', **template_vars)


    if request.method == 'POST': # 此 POST 用于文件上传和初始处理
        app.logger.info("POST 请求到 '/', 开始处理上传文件...")
        model_cache = {} # 每次上传新文件时, 清空旧的缓存

        if 'file' not in request.files:
            app.logger.error("POST 请求错误: 请求中没有文件部分。")
            model_cache['error_message_to_display'] = "请求中没有文件部分。"
            return redirect(url_for('index'))

        file = request.files['file']
        if file.filename == '':
            app.logger.error("POST 请求错误: 未选择文件。")
            model_cache['error_message_to_display'] = "未选择文件。"
            return redirect(url_for('index'))

        if file and file.filename.endswith('.csv'):
            try:
                filename = file.filename
                app.logger.info(f"成功接收到文件: {filename}")
                df = pd.read_csv(file)
                model_cache['filename'] = filename

                if "OK_NG" not in df.columns:
                    app.logger.error(f"文件 {filename} 缺少 'OK_NG' 列。")
                    model_cache['error_message_to_display'] = "上传的 CSV 文件必须包含 'OK_NG' 列。"
                    return redirect(url_for('index'))
                
                X = df.drop("OK_NG", axis=1).copy() # 特征
                try:
                    # 尝试将所有特征列转换为数值型, 忽略无法转换的错误, 后续SHAP可能处理
                    for col in X.columns:
                        X[col] = pd.to_numeric(X[col], errors='coerce')
                    # 可以选择填充NaN, 或让XGBoost处理, 或者在这里报错
                    if X.isnull().values.any():
                        app.logger.warning(f"文件 {filename} 中的特征列在转换为数值型后包含NaN值。将使用均值填充。")
                        X = X.fillna(X.mean()) # 用均值填充NaN
                except Exception as e: # 更通用的异常捕获
                    app.logger.error(f"转换特征为数值型时出错: {e}")
                    model_cache['error_message_to_display'] = f"特征转换错误: {e}"
                    return redirect(url_for('index'))

                y_raw = df["OK_NG"].copy() # 原始目标变量
                if not y_raw.isin(['OK', 'NG']).all(): # 检查是否所有值都在 ['OK', 'NG'] 中
                    app.logger.error(f"文件 {filename} 的 'OK_NG' 列包含无效值。")
                    model_cache['error_message_to_display'] = "列 'OK_NG' 必须只包含 'OK' 或 'NG' 值。"
                    return redirect(url_for('index'))

                ok_label_numeric = 1
                ng_label_numeric = 0
                y_numeric = y_raw.map({'OK': ok_label_numeric, 'NG': ng_label_numeric}).astype(int)
                
                model_cache['feature_names'] = list(X.columns)
                model_cache['X_train_df_for_explainer'] = X.copy() # 用于 SHAP 的背景数据集

                # 样本权重
                weights = compute_sample_weight(class_weight={ng_label_numeric: 1.0, ok_label_numeric: 2.0}, y=y_numeric)

                # XGBoost 分类器
                clf = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                                        objective="binary:logistic", eval_metric="logloss",
                                        random_state=42, use_label_encoder=False)
                clf.fit(X, y_numeric, sample_weight=weights)
                model_cache['clf'] = clf # 缓存训练好的模型
                app.logger.info("XGBoost 模型训练完成。")

                # --- 阈值扫描 ---
                probs = clf.predict_proba(X)[:, ok_label_numeric] # 'OK' 类的概率
                thresholds = np.arange(0.1, 0.91, 0.01) # 扫描的阈值范围
                metrics_list = []
                for t_val in thresholds: # 使用 t_val 避免与内置的 t 冲突
                    preds = (probs >= t_val).astype(int) # 根据阈值进行预测
                    report = classification_report(y_numeric, preds, output_dict=True, zero_division=0,
                                                   labels=[ng_label_numeric, ok_label_numeric], # 指定标签顺序
                                                   target_names=['NG', 'OK']) # 指定标签名称
                    metrics_list.append({
                        "threshold": t_val,
                        "accuracy": report["accuracy"],
                        "NG_recall": report["NG"]["recall"],
                        "OK_recall": report["OK"]["recall"],
                        "OK_precision": report["OK"]["precision"],
                        "f1_score_weighted": f1_score(y_numeric, preds, average='weighted', zero_division=0) # 加权平均 F1
                    })
                metrics_df = pd.DataFrame(metrics_list)
                model_cache['metrics_df_html'] = metrics_df.to_html(classes='table table-sm table-striped table-hover', index=False, float_format='{:.2f}'.format)

                # 筛选满足 OK 召回率 >= 0.95 的阈值
                filtered_metrics = metrics_df[metrics_df["OK_recall"] >= 0.95]
                best_recommendation_df = pd.DataFrame() # 如果没有合适的阈值, 则为空 DataFrame
                if not filtered_metrics.empty:
                    best_recommendation_df = filtered_metrics.sort_values(
                        ["NG_recall", "f1_score_weighted"], ascending=[False, False] # 优先 NG 召回率, 其次 F1
                    ).head(1)
                    # best_thresh 是一个 Series，取第一个元素
                    best_thresh_val = best_recommendation_df["threshold"].values[0]
                    recommended_threshold_text = (
                        f"✅ 推荐分类阈值为：{best_thresh_val:.2f}\n"
                        f"   - NG召回率: {best_recommendation_df['NG_recall'].values[0]:.2f}\n"
                        f"   - OK召回率: {best_recommendation_df['OK_recall'].values[0]:.2f}\n"
                        f"   - F1 分数 (加权): {best_recommendation_df['f1_score_weighted'].values[0]:.2f}"
                    )
                    model_cache['best_recommendation_html'] = best_recommendation_df.to_html(classes='table table-sm table-striped', index=False, float_format='{:.2f}'.format)
                else:
                    best_thresh_val = 0.5 # 如果没有阈值满足条件, 则使用默认值 0.5
                    recommended_threshold_text = "⚠️ 没有找到 OK召回率≥0.95 的推荐阈值。将使用默认阈值0.5进行预测。"
                    model_cache['best_recommendation_html'] = "<p>无推荐阈值满足条件。</p>"
                
                model_cache['best_thresh'] = best_thresh_val # 缓存最佳阈值
                model_cache['recommended_threshold_text'] = recommended_threshold_text.replace("\n", "<br>") # 缓存推荐文本
                app.logger.info(f"阈值扫描完成, 推荐阈值: {best_thresh_val:.2f}")

                # --- 性能图绘制 ---
                fig_perf, ax_perf = plt.subplots(figsize=(10, 6))
                ax_perf.plot(metrics_df["threshold"], metrics_df["NG_recall"], label="NG 召回率", color="red")
                ax_perf.plot(metrics_df["threshold"], metrics_df["OK_recall"], label="OK 召回率", color="green")
                ax_perf.plot(metrics_df["threshold"], metrics_df["f1_score_weighted"], label="F1 分数 (加权)", color="blue")
                if not best_recommendation_df.empty: # 如果找到了推荐阈值, 则在图上标记
                    ax_perf.axvline(x=best_thresh_val, color="purple", linestyle="--", label=f"推荐阈值: {best_thresh_val:.2f}")
                ax_perf.set_xlabel("分类阈值")
                ax_perf.set_ylabel("指标值")
                ax_perf.set_title("XGBoost 模型性能 vs. 阈值")
                ax_perf.legend()
                ax_perf.grid(True)
                plt.tight_layout()
                model_cache['base64_perf_plot'] = fig_to_base64(fig_perf)

                # --- 全局 SHAP 特征重要性图 ---
                explainer_global = shap.Explainer(clf, X) # X 是背景数据集
                shap_values_all = explainer_global(X) # 计算所有训练样本的 SHAP 值
                
                fig_global_shap, _ = plt.subplots() # 为 SHAP bar 图创建新的图像
                shap.plots.bar(shap_values_all, show=False, max_display=15) # show=False 避免在服务器端显示
                plt.tight_layout()
                model_cache['base64_global_shap_plot'] = fig_to_base64(fig_global_shap)
                app.logger.info("性能图和全局 SHAP 图生成完毕。")
                
                model_cache['form_inputs'] = model_cache['feature_names'] # 用于生成预测表单
                model_cache['default_values'] = X.mean().to_dict() # 用于预填充预测表单的默认值
                model_cache['show_results_from_upload'] = True # 标记结果来自文件上传
                model_cache['show_single_pred_results'] = False # 初始时不显示单样本预测结果

                return redirect(url_for('index'))

            except Exception as e:
                app.logger.error(f"处理文件 {filename} 时发生严重错误: {e}", exc_info=True) # 记录详细错误信息
                model_cache = {} # 清理缓存
                model_cache['error_message_to_display'] = f"处理文件时发生严重错误: {str(e)}"
                return redirect(url_for('index'))

        else: # 文件类型不是 CSV
             app.logger.error("POST 请求错误: 文件类型无效, 非CSV文件。")
             model_cache = {}
             model_cache['error_message_to_display'] = "文件类型无效。请上传 CSV 文件。"
             return redirect(url_for('index'))
    
    app.logger.warning("接收到未知类型的请求或意外的流程, 重定向到主页。")
    model_cache = {} # 确保从干净的状态开始
    return redirect(url_for('index'))

@app.route('/predict_single', methods=['POST'])
def predict_single():
    global model_cache
    app.logger.info("POST 请求到 '/predict_single', 开始单一样本预测...")

    if 'clf' not in model_cache:
        app.logger.warning("'/predict_single' 错误: 模型未在缓存中找到。")
        model_cache['error_message_to_display'] = "请先上传并处理一个CSV文件, 然后再进行单一样本预测。"
        return redirect(url_for('index'))

    try:
        clf = model_cache['clf']
        best_thresh = model_cache['best_thresh']
        X_background = model_cache.get('X_train_df_for_explainer')
        feature_names = model_cache.get('feature_names')

        if X_background is None or feature_names is None:
            app.logger.error("'/predict_single' 错误: 缓存中缺少 X_background 或 feature_names。")
            model_cache['single_pred_error_to_display'] = "内部错误：缺少必要的训练数据信息。"
            return redirect(url_for('index'))

        input_data_dict = {}
        current_inputs_for_form = {} # 用于在出错时保留表单值
        for f_name in feature_names:
            form_value = request.form.get(f_name)
            current_inputs_for_form[f_name] = form_value # 总是先记录下来
            if form_value is None or form_value.strip() == "":
                 app.logger.warning(f"单样本预测表单输入错误: 特征 '{f_name}' 的值为空。")
                 model_cache['single_pred_error_to_display'] = f"特征 '{f_name}' 的值不能为空。"
                 model_cache['default_values'] = current_inputs_for_form # 保留用户已输入的值
                 model_cache['show_single_pred_results'] = True # 标记需要显示单预测区域（即使是显示错误）
                 return redirect(url_for('index'))
            try:
                input_data_dict[f_name] = float(form_value)
            except ValueError:
                app.logger.warning(f"单样本预测表单输入错误: 特征 '{f_name}' 的值 '{form_value}' 不是有效数字。")
                model_cache['single_pred_error_to_display'] = f"特征 '{f_name}' 的输入无效, 请输入一个数字。当前值为: '{form_value}'"
                model_cache['default_values'] = current_inputs_for_form
                model_cache['show_single_pred_results'] = True
                return redirect(url_for('index'))
        
        df_input = pd.DataFrame([input_data_dict], columns=feature_names)
        app.logger.info(f"单样本预测输入数据: {df_input.to_dict(orient='records')}")

        # 确保 prob_ok 获取的是单个概率值，而不是数组
        prob_ok_array = clf.predict_proba(df_input)[:, ok_label_numeric] # ok_label_numeric 应该是 1
        if prob_ok_array.ndim > 0: # 如果是数组
            prob_ok_scalar = prob_ok_array[0] # 取第一个元素
        else: # 如果已经是标量
            prob_ok_scalar = prob_ok_array

        pred_label_text = "✅ 合格 (OK)" if prob_ok_scalar >= best_thresh else "❌ 不合格 (NG)"
        app.logger.info(f"单样本预测概率 (OK): {prob_ok_scalar:.3f}, 标签: {pred_label_text}")

        explainer_instance = shap.Explainer(clf, X_background)
        shap_values_instance = explainer_instance(df_input) # 这会返回一个 Explanation 对象

        # generate_shap_waterfall_base64 需要单个样本的 Explanation 对象
        model_cache['base64_waterfall_plot'] = generate_shap_waterfall_base64(shap_values_instance[0])


        # shap_values_instance.values 是一个 (n_samples, n_features) 的数组
        # 对于单个样本预测, n_samples 是 1
        shap_values_for_df = shap_values_instance.values[0]
        base_value_for_df = shap_values_instance.base_values[0]

        shap_df_data = pd.DataFrame({
            "特征": df_input.columns,
            "SHAP 值": shap_values_for_df
        })
        
        predicted_logit = base_value_for_df + np.sum(shap_values_for_df)
        
        impacts = []
        for _, shap_val_feature in enumerate(shap_values_for_df): # 迭代单个样本的SHAP值
            impact = sigmoid(predicted_logit) - sigmoid(predicted_logit - shap_val_feature)
            impacts.append(impact)
        shap_df_data["对概率的影响"] = impacts
        
        shap_df_data = shap_df_data.sort_values(by="SHAP 值", key=abs, ascending=False).head(5)
        app.logger.info("单样本 SHAP 分析完成。")
        
        model_cache['show_results_from_upload'] = True # 确保主结果区保持显示
        model_cache['show_single_pred_results'] = True
        model_cache['single_pred_input_data_html'] = pd.DataFrame([input_data_dict]).to_html(classes='table table-sm table-bordered', index=False, float_format='{:.2f}'.format)
        model_cache['single_pred_prob'] = f"{prob_ok_scalar:.3f}"
        model_cache['single_pred_label'] = pred_label_text
        model_cache['single_pred_best_thresh'] = f"{best_thresh:.2f}" # best_thresh 应该是标量
        model_cache['single_pred_shap_table_html'] = shap_df_data.to_html(classes='table table-sm table-striped table-hover', index=False, float_format='{:.4f}'.format)
        model_cache['default_values'] = df_input.iloc[0].to_dict() # 更新表单的默认值为当前输入
        model_cache.pop('single_pred_error_to_display', None) # 成功后清除单次预测错误

        return redirect(url_for('index'))

    except Exception as e:
        app.logger.error(f"'/predict_single' 发生严重错误: {e}", exc_info=True)
        model_cache['show_results_from_upload'] = True # 尝试保持主结果区
        model_cache['show_single_pred_results'] = True # 仍然尝试显示单预测部分, 但会带有错误
        model_cache['single_pred_error_to_display'] = f"单次预测过程中发生严重错误: {str(e)}"
        return redirect(url_for('index'))

# ---- 关键部分：确保 app.run 在 __main__ 中并配置正确 ----
if __name__ == '__main__':
    # Render 会设置 PORT 环境变量。在本地开发时, 如果没有设置, 则默认为 5000。
    port = int(os.environ.get("PORT", 5000))
    # 对于直接使用 `python app.py` 启动, debug=True 在本地开发时有用。
    # Render 在生产环境中通常会忽略这个 debug 设置, 或者有自己的方式来控制。
    # 重要的是 host='0.0.0.0', 这样 Render 才能从外部访问到容器内的应用。
    app.logger.info(f"应用启动中, 监听地址 0.0.0.0, 端口: {port}")
    app.run(host='0.0.0.0', port=port, debug=False) # 对于Render, debug通常应为False
