import base64
import io
import os
import logging

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from flask import Flask, render_template, request, redirect, url_for
from scipy.special import expit as sigmoid
from sklearn.metrics import classification_report, f1_score
from sklearn.utils.class_weight import compute_sample_weight

app = Flask(__name__)
app.secret_key = os.urandom(24)

logging.basicConfig(level=logging.INFO)
app.logger.setLevel(logging.INFO)

model_cache = {}

try:
    matplotlib.rcParams['font.family'] = 'sans-serif'
    matplotlib.rcParams['axes.unicode_minus'] = False
except Exception as e:
    app.logger.warning(f"Matplotlib 字体设置警告: {e}")

def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def generate_shap_waterfall_base64(shap_values_single_instance):
    fig = plt.figure()
    shap.plots.waterfall(shap_values_single_instance, show=False, max_display=15)
    plt.tight_layout()
    base64_str = fig_to_base64(fig)
    return base64_str

@app.route('/', methods=['GET', 'POST'])
def index():
    global model_cache
    app.logger.info(f"访问 '/' 路由, 方法: {request.method}")

    if request.method == 'GET':
        if not model_cache.get('show_results_from_upload') and not model_cache.get('show_single_pred_results'):
            app.logger.info("GET 请求到 '/', 清空 model_cache (可能是新会话或直接访问)")
            model_cache = {}
        else:
            app.logger.info("GET 请求到 '/', 保留 model_cache (可能从POST或 predict_single 重定向)")
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
            'error_message': model_cache.get('error_message_to_display'),
            'single_pred_error': model_cache.get('single_pred_error_to_display')
        }
        return render_template('index.html', **template_vars)

    if request.method == 'POST':
        app.logger.info("POST 请求到 '/', 开始处理上传文件...")
        model_cache = {}

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
                
                X = df.drop("OK_NG", axis=1).copy()
                try:
                    for col in X.columns:
                        X[col] = pd.to_numeric(X[col], errors='coerce')
                    if X.isnull().values.any():
                        app.logger.warning(f"文件 {filename} 中的特征列在转换为数值型后包含NaN值。将使用均值填充。")
                        X = X.fillna(X.mean())
                except Exception as e:
                    app.logger.error(f"转换特征为数值型时出错: {e}")
                    model_cache['error_message_to_display'] = f"特征转换错误: {e}"
                    return redirect(url_for('index'))

                # --- 修改开始：处理数值型的 OK_NG 列 ---
                y_numeric_from_csv = df["OK_NG"].copy()

                # 确保 OK_NG 列是数值类型 (0 或 1)
                if not pd.api.types.is_numeric_dtype(y_numeric_from_csv) or not y_numeric_from_csv.isin([0, 1]).all():
                    # 尝试转换，如果列中包含非0/1的数值或非数值，则报错
                    try:
                        y_numeric_from_csv = pd.to_numeric(y_numeric_from_csv, errors='raise')
                        if not y_numeric_from_csv.isin([0, 1]).all():
                            raise ValueError("转换后仍包含非0或1的值")
                    except (ValueError, TypeError) as e:
                        invalid_values_original = df["OK_NG"][~df["OK_NG"].astype(str).isin(['0', '1'])].unique()
                        app.logger.error(f"文件 {filename} 的 'OK_NG' 列期望是0或1，但包含无效值。错误: {e}。无效值示例: {invalid_values_original[:5]}")
                        model_cache['error_message_to_display'] = f"列 'OK_NG' 必须只包含数字 0 (不合格) 或 1 (合格)。无效值示例: {', '.join(map(str, invalid_values_original[:3]))}{'...' if len(invalid_values_original) > 3 else ''}"
                        return redirect(url_for('index'))
                
                # 此时 y_numeric_from_csv 已经是包含 0 和 1 的 Series
                y_numeric = y_numeric_from_csv.astype(int) # 确保是整数类型

                # 定义数值标签的含义，这在后续 predict_proba 和 classification_report 中很重要
                # 我们假设 1 代表 "OK" (正类), 0 代表 "NG" (负类)
                ok_label_numeric_val = 1  # OK (合格) 对应的数值
                ng_label_numeric_val = 0  # NG (不合格) 对应的数值
                # --- 修改结束 ---
                
                model_cache['feature_names'] = list(X.columns)
                model_cache['X_train_df_for_explainer'] = X.copy()

                weights = compute_sample_weight(class_weight={ng_label_numeric_val: 1.0, ok_label_numeric_val: 2.0}, y=y_numeric)

                clf = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                                        objective="binary:logistic", eval_metric="logloss",
                                        random_state=42, use_label_encoder=False) # XGBoost可以直接处理数值型y
                clf.fit(X, y_numeric, sample_weight=weights)
                model_cache['clf'] = clf
                app.logger.info("XGBoost 模型训练完成。")

                # --- 修改：阈值扫描和报告中使用数值标签 ---
                # predict_proba 会返回两列，第二列 (索引1) 是正类 (我们定义为 ok_label_numeric_val) 的概率
                probs = clf.predict_proba(X)[:, ok_label_numeric_val]
                thresholds = np.arange(0.1, 0.91, 0.01)
                metrics_list = []
                for t_val in thresholds:
                    preds_numeric = (probs >= t_val).astype(int)
                    report = classification_report(y_numeric, preds_numeric, output_dict=True, zero_division=0,
                                                   labels=[ng_label_numeric_val, ok_label_numeric_val], # 指定标签顺序
                                                   target_names=['NG (0)', 'OK (1)']) # 指定标签名称以供报告
                    metrics_list.append({
                        "threshold": t_val,
                        "accuracy": report["accuracy"],
                        "NG_recall": report["NG (0)"]["recall"], # 使用报告中的键
                        "OK_recall": report["OK (1)"]["recall"], # 使用报告中的键
                        "OK_precision": report["OK (1)"]["precision"],
                        "f1_score_weighted": f1_score(y_numeric, preds_numeric, average='weighted', zero_division=0)
                    })
                metrics_df = pd.DataFrame(metrics_list)
                model_cache['metrics_df_html'] = metrics_df.to_html(classes='table table-sm table-striped table-hover', index=False, float_format='{:.2f}'.format)

                filtered_metrics = metrics_df[metrics_df["OK_recall"] >= 0.95]
                best_recommendation_df = pd.DataFrame()
                if not filtered_metrics.empty:
                    best_recommendation_df = filtered_metrics.sort_values(
                        ["NG_recall", "f1_score_weighted"], ascending=[False, False]
                    ).head(1)
                    best_thresh_val = best_recommendation_df["threshold"].values[0]
                    recommended_threshold_text = (
                        f"✅ 推荐分类阈值为：{best_thresh_val:.2f}\n"
                        f"   - NG召回率: {best_recommendation_df['NG_recall'].values[0]:.2f}\n"
                        f"   - OK召回率: {best_recommendation_df['OK_recall'].values[0]:.2f}\n"
                        f"   - F1 分数 (加权): {best_recommendation_df['f1_score_weighted'].values[0]:.2f}"
                    )
                    model_cache['best_recommendation_html'] = best_recommendation_df.to_html(classes='table table-sm table-striped', index=False, float_format='{:.2f}'.format)
                else:
                    best_thresh_val = 0.5
                    recommended_threshold_text = "⚠️ 没有找到 OK召回率≥0.95 的推荐阈值。将使用默认阈值0.5进行预测。"
                    model_cache['best_recommendation_html'] = "<p>无推荐阈值满足条件。</p>"
                
                model_cache['best_thresh'] = best_thresh_val
                model_cache['recommended_threshold_text'] = recommended_threshold_text.replace("\n", "<br>")
                app.logger.info(f"阈值扫描完成, 推荐阈值: {best_thresh_val:.2f}")
                # --- 修改结束 ---

                fig_perf, ax_perf = plt.subplots(figsize=(10, 6))
                ax_perf.plot(metrics_df["threshold"], metrics_df["NG_recall"], label="NG (0) 召回率", color="red")
                ax_perf.plot(metrics_df["threshold"], metrics_df["OK_recall"], label="OK (1) 召回率", color="green")
                ax_perf.plot(metrics_df["threshold"], metrics_df["f1_score_weighted"], label="F1 分数 (加权)", color="blue")
                if not best_recommendation_df.empty:
                    ax_perf.axvline(x=best_thresh_val, color="purple", linestyle="--", label=f"推荐阈值: {best_thresh_val:.2f}")
                ax_perf.set_xlabel("分类阈值")
                ax_perf.set_ylabel("指标值")
                ax_perf.set_title("XGBoost 模型性能 vs. 阈值")
                ax_perf.legend()
                ax_perf.grid(True)
                plt.tight_layout()
                model_cache['base64_perf_plot'] = fig_to_base64(fig_perf)

                explainer_global = shap.Explainer(clf, X)
                shap_values_all = explainer_global(X)
                fig_global_shap, _ = plt.subplots()
                shap.plots.bar(shap_values_all, show=False, max_display=15)
                plt.tight_layout()
                model_cache['base64_global_shap_plot'] = fig_to_base64(fig_global_shap)
                app.logger.info("性能图和全局 SHAP 图生成完毕。")
                
                model_cache['form_inputs'] = model_cache['feature_names']
                model_cache['default_values'] = X.mean().to_dict()
                model_cache['show_results_from_upload'] = True
                model_cache['show_single_pred_results'] = False

                return redirect(url_for('index'))

            except Exception as e:
                app.logger.error(f"处理文件 {filename} 时发生严重错误: {e}", exc_info=True)
                model_cache = {}
                model_cache['error_message_to_display'] = f"处理文件时发生严重错误: {str(e)}"
                return redirect(url_for('index'))

        else:
             app.logger.error("POST 请求错误: 文件类型无效, 非CSV文件。")
             model_cache = {}
             model_cache['error_message_to_display'] = "文件类型无效。请上传 CSV 文件。"
             return redirect(url_for('index'))
    
    app.logger.warning("接收到未知类型的请求或意外的流程, 重定向到主页。")
    model_cache = {}
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
        best_thresh = model_cache['best_thresh'] # 这是数值阈值
        X_background = model_cache.get('X_train_df_for_explainer')
        feature_names = model_cache.get('feature_names')

        # 定义数值标签的含义
        ok_label_numeric_val = 1
        ng_label_numeric_val = 0


        if X_background is None or feature_names is None:
            app.logger.error("'/predict_single' 错误: 缓存中缺少 X_background 或 feature_names。")
            model_cache['single_pred_error_to_display'] = "内部错误：缺少必要的训练数据信息。"
            return redirect(url_for('index'))

        input_data_dict = {}
        current_inputs_for_form = {}
        for f_name in feature_names:
            form_value = request.form.get(f_name)
            current_inputs_for_form[f_name] = form_value
            if form_value is None or form_value.strip() == "":
                 app.logger.warning(f"单样本预测表单输入错误: 特征 '{f_name}' 的值为空。")
                 model_cache['single_pred_error_to_display'] = f"特征 '{f_name}' 的值不能为空。"
                 model_cache['default_values'] = current_inputs_for_form
                 model_cache['show_single_pred_results'] = True
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

        # --- 修改：单样本预测也基于数值标签 ---
        # predict_proba 返回属于每个类的概率, 第二列 (索引1) 是 ok_label_numeric_val (即1) 的概率
        prob_ok_array = clf.predict_proba(df_input)[:, ok_label_numeric_val]
        prob_ok_scalar = prob_ok_array[0] if prob_ok_array.ndim > 0 else prob_ok_array

        # 预测的数值标签
        predicted_numeric_label = ok_label_numeric_val if prob_ok_scalar >= best_thresh else ng_label_numeric_val
        
        # 转换为文本标签用于显示
        pred_label_text = "✅ 合格 (OK)" if predicted_numeric_label == ok_label_numeric_val else "❌ 不合格 (NG)"
        app.logger.info(f"单样本预测概率 (OK): {prob_ok_scalar:.3f}, 预测数值标签: {predicted_numeric_label}, 文本标签: {pred_label_text}")
        # --- 修改结束 ---

        explainer_instance = shap.Explainer(clf, X_background)
        shap_values_instance = explainer_instance(df_input)

        # --- 修改：确保 SHAP 图是针对预测为 NG (0) 的情况 ---
        if predicted_numeric_label == ng_label_numeric_val: # 如果预测为不合格 (0)
            model_cache['base64_waterfall_plot'] = generate_shap_waterfall_base64(shap_values_instance[0])
            app.logger.info("为不合格样本生成了 SHAP Waterfall 图。")
        else:
            model_cache['base64_waterfall_plot'] = None # 合格样本不显示 SHAP 图
            app.logger.info("合格样本, 不生成 SHAP Waterfall 图。")
        # --- 修改结束 ---


        shap_values_for_df = shap_values_instance.values[0]
        base_value_for_df = shap_values_instance.base_values[0]

        shap_df_data = pd.DataFrame({
            "特征": df_input.columns,
            "SHAP 值": shap_values_for_df
        })
        
        predicted_logit = base_value_for_df + np.sum(shap_values_for_df)
        
        impacts = []
        for _, shap_val_feature in enumerate(shap_values_for_df):
            impact = sigmoid(predicted_logit) - sigmoid(predicted_logit - shap_val_feature)
            impacts.append(impact)
        shap_df_data["对概率的影响"] = impacts
        
        shap_df_data = shap_df_data.sort_values(by="SHAP 值", key=abs, ascending=False).head(5)
        app.logger.info("单样本 SHAP 表格数据分析完成。")
        
        model_cache['show_results_from_upload'] = True
        model_cache['show_single_pred_results'] = True
        model_cache['single_pred_input_data_html'] = pd.DataFrame([input_data_dict]).to_html(classes='table table-sm table-bordered', index=False, float_format='{:.2f}'.format)
        model_cache['single_pred_prob'] = f"{prob_ok_scalar:.3f}"
        model_cache['single_pred_label'] = pred_label_text
        model_cache['single_pred_best_thresh'] = f"{best_thresh:.2f}"
        model_cache['single_pred_shap_table_html'] = shap_df_data.to_html(classes='table table-sm table-striped table-hover', index=False, float_format='{:.4f}'.format)
        model_cache['default_values'] = df_input.iloc[0].to_dict()
        model_cache.pop('single_pred_error_to_display', None)

        return redirect(url_for('index'))

    except Exception as e:
        app.logger.error(f"'/predict_single' 发生严重错误: {e}", exc_info=True)
        model_cache['show_results_from_upload'] = True
        model_cache['show_single_pred_results'] = True
        model_cache['single_pred_error_to_display'] = f"单次预测过程中发生严重错误: {str(e)}"
        return redirect(url_for('index'))

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.logger.info(f"应用启动中, 监听地址 0.0.0.0, 端口: {port}")
    app.run(host='0.0.0.0', port=port, debug=False)
