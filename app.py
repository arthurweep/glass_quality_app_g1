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
            # 'best_recommendation_html': model_cache.get('best_recommendation_html'), # 移除自动推荐相关的HTML
            'recommended_threshold_text': model_cache.get('recommended_threshold_text'), # 这个会显示固定阈值的信息
            'base64_global_shap_plot': model_cache.get('base64_global_shap_plot'),
            'default_values': model_cache.get('default_values'),
            'single_pred_input_data_html': model_cache.get('single_pred_input_data_html'),
            'single_pred_prob': model_cache.get('single_pred_prob'),
            'single_pred_label': model_cache.get('single_pred_label'),
            'single_pred_best_thresh': model_cache.get('single_pred_best_thresh'), # 会显示固定的0.5
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

                y_numeric_from_csv = df["OK_NG"].copy()
                if not pd.api.types.is_numeric_dtype(y_numeric_from_csv) or not y_numeric_from_csv.isin([0, 1]).all():
                    try:
                        y_numeric_from_csv = pd.to_numeric(y_numeric_from_csv, errors='raise')
                        if not y_numeric_from_csv.isin([0, 1]).all():
                            raise ValueError("转换后仍包含非0或1的值")
                    except (ValueError, TypeError) as e_val:
                        invalid_values_original = df["OK_NG"][~df["OK_NG"].astype(str).isin(['0', '1'])].unique()
                        app.logger.error(f"文件 {filename} 的 'OK_NG' 列期望是0或1, 但包含无效值。错误: {e_val}。无效值示例: {invalid_values_original[:5]}")
                        error_msg_part = f"无效值示例: {', '.join(map(str, invalid_values_original[:3]))}{'...' if len(invalid_values_original) > 3 else ''}"
                        model_cache['error_message_to_display'] = f"列 'OK_NG' 必须只包含数字 0 (不合格) 或 1 (合格)。{error_msg_part}"
                        return redirect(url_for('index'))
                
                y_numeric = y_numeric_from_csv.astype(int)
                ok_label_numeric_val = 1
                ng_label_numeric_val = 0
                
                model_cache['feature_names'] = list(X.columns)
                model_cache['X_train_df_for_explainer'] = X.copy()

                weights = compute_sample_weight(class_weight={ng_label_numeric_val: 1.0, ok_label_numeric_val: 2.0}, y=y_numeric)

                clf = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                                        objective="binary:logistic", eval_metric="logloss",
                                        random_state=42, use_label_encoder=False)
                clf.fit(X, y_numeric, sample_weight=weights)
                model_cache['clf'] = clf
                app.logger.info("XGBoost 模型训练完成。")

                # --- 修改：固定阈值为0.5 ---
                fixed_threshold = 0.5
                model_cache['best_thresh'] = fixed_threshold # 使用固定阈值
                # 推荐文本也反映固定阈值
                recommended_threshold_text = f"ℹ️ 分类阈值固定为: {fixed_threshold:.1f}" # 中文信息
                model_cache['recommended_threshold_text'] = recommended_threshold_text
                # best_recommendation_html 可以不设置或设置提示信息
                model_cache['best_recommendation_html'] = f"<p>当前使用的分类阈值为固定的 <strong>{fixed_threshold:.1f}</strong>。</p>" # 中文信息
                app.logger.info(f"分类阈值固定为: {fixed_threshold}")
                # --- 修改结束 ---

                # 仍然计算在固定阈值下的性能指标以供展示
                probs = clf.predict_proba(X)[:, ok_label_numeric_val]
                preds_at_fixed_thresh = (probs >= fixed_threshold).astype(int)
                report_at_fixed_thresh = classification_report(y_numeric, preds_at_fixed_thresh, output_dict=True, zero_division=0,
                                                               labels=[ng_label_numeric_val, ok_label_numeric_val],
                                                               target_names=['NG_Class_0', 'OK_Class_1']) # 英文键名
                
                metrics_list_for_display = [{
                    "threshold": fixed_threshold, # 显示当前使用的固定阈值
                    "accuracy": report_at_fixed_thresh["accuracy"],
                    "NG_recall": report_at_fixed_thresh["NG_Class_0"].get("recall", 0),
                    "OK_recall": report_at_fixed_thresh["OK_Class_1"].get("recall", 0),
                    "OK_precision": report_at_fixed_thresh["OK_Class_1"].get("precision", 0),
                    "f1_score_weighted": f1_score(y_numeric, preds_at_fixed_thresh, average='weighted', zero_division=0)
                }]
                metrics_df_for_display = pd.DataFrame(metrics_list_for_display)
                # 这个HTML表格现在只显示固定阈值下的指标
                model_cache['metrics_df_html'] = metrics_df_for_display.to_html(classes='table table-sm table-striped table-hover', index=False, float_format='{:.2f}'.format)


                # --- 修改：性能图绘制逻辑调整 ---
                # 性能图仍然可以绘制不同阈值下的表现，但会特别标记出我们固定的0.5阈值
                fig_perf, ax_perf = plt.subplots(figsize=(10, 6))
                scan_thresholds = np.arange(0.1, 0.91, 0.01) # 用于绘图的扫描阈值
                ng_recalls_scan = []
                ok_recalls_scan = []
                f1_scores_scan = []

                for t_scan in scan_thresholds:
                    preds_scan = (probs >= t_scan).astype(int)
                    report_scan = classification_report(y_numeric, preds_scan, output_dict=True, zero_division=0,
                                                        labels=[ng_label_numeric_val, ok_label_numeric_val],
                                                        target_names=['NG_Class_0', 'OK_Class_1']) # 英文键名
                    ng_recalls_scan.append(report_scan["NG_Class_0"].get("recall", 0))
                    ok_recalls_scan.append(report_scan["OK_Class_1"].get("recall", 0))
                    f1_scores_scan.append(f1_score(y_numeric, preds_scan, average='weighted', zero_division=0))
                
                ax_perf.plot(scan_thresholds, ng_recalls_scan, label="NG (Class 0) Recall", color="red")
                ax_perf.plot(scan_thresholds, ok_recalls_scan, label="OK (Class 1) Recall", color="green")
                ax_perf.plot(scan_thresholds, f1_scores_scan, label="F1 Score (Weighted)", color="blue")
                # 标记固定的阈值
                ax_perf.axvline(x=fixed_threshold, color="purple", linestyle="--", label=f"Fixed Threshold: {fixed_threshold:.1f}")
                
                ax_perf.set_xlabel("Classification Threshold") # 英文
                ax_perf.set_ylabel("Metric Value") # 英文
                ax_perf.set_title("Model Performance vs. Threshold (Fixed Threshold Applied)") # 英文
                ax_perf.legend(loc='best')
                ax_perf.grid(True)
                plt.tight_layout()
                model_cache['base64_perf_plot'] = fig_to_base64(fig_perf)
                # --- 修改结束 ---


                explainer_global = shap.Explainer(clf, X)
                shap_values_all = explainer_global(X)
                fig_global_shap, _ = plt.subplots()
                shap.plots.bar(shap_values_all, show=False, max_display=15) # SHAP图通常是英文
                plt.tight_layout()
                model_cache['base64_global_shap_plot'] = fig_to_base64(fig_global_shap)
                app.logger.info("性能图和全局 SHAP 图生成完毕。") # 中文日志
                
                model_cache['form_inputs'] = model_cache['feature_names']
                model_cache['default_values'] = X.mean().to_dict()
                model_cache['show_results_from_upload'] = True
                model_cache['show_single_pred_results'] = False

                return redirect(url_for('index'))

            except Exception as e:
                app.logger.error(f"处理文件 {filename} 时发生严重错误: {e}", exc_info=True)
                model_cache = {}
                model_cache['error_message_to_display'] = f"处理文件时发生严重错误: {str(e)}" # 中文错误信息
                return redirect(url_for('index'))

        else:
             app.logger.error("POST 请求错误: 文件类型无效, 非CSV文件。")
             model_cache = {}
             model_cache['error_message_to_display'] = "文件类型无效。请上传 CSV 文件。" # 中文错误信息
             return redirect(url_for('index'))
    
    app.logger.warning("接收到未知类型的请求或意外的流程, 重定向到主页。") # 中文日志
    model_cache = {}
    return redirect(url_for('index'))

@app.route('/predict_single', methods=['POST'])
def predict_single():
    global model_cache
    app.logger.info("POST 请求到 '/predict_single', 开始单一样本预测...") # 中文日志

    if 'clf' not in model_cache or 'best_thresh' not in model_cache: # 确保模型和固定阈值已设置
        app.logger.warning("'/predict_single' 错误: 模型或阈值未在缓存中找到。")
        model_cache['error_message_to_display'] = "请先上传并处理一个CSV文件, 然后再进行单一样本预测。" # 中文错误信息
        return redirect(url_for('index'))

    try:
        clf = model_cache['clf']
        fixed_threshold = model_cache['best_thresh'] # 获取缓存中的固定阈值 (应该是0.5)
        X_background = model_cache.get('X_train_df_for_explainer')
        feature_names = model_cache.get('feature_names')
        ok_label_numeric_val = 1
        ng_label_numeric_val = 0

        if X_background is None or feature_names is None:
            app.logger.error("'/predict_single' 错误: 缓存中缺少 X_background 或 feature_names。")
            model_cache['single_pred_error_to_display'] = "内部错误：缺少必要的训练数据信息。" # 中文错误信息
            return redirect(url_for('index'))

        input_data_dict = {}
        current_inputs_for_form = {}
        for f_name in feature_names:
            form_value = request.form.get(f_name)
            current_inputs_for_form[f_name] = form_value
            if form_value is None or form_value.strip() == "":
                 app.logger.warning(f"单样本预测表单输入错误: 特征 '{f_name}' 的值为空。")
                 model_cache['single_pred_error_to_display'] = f"特征 '{f_name}' 的值不能为空。" # 中文错误信息
                 model_cache['default_values'] = current_inputs_for_form
                 model_cache['show_single_pred_results'] = True
                 return redirect(url_for('index'))
            try:
                input_data_dict[f_name] = float(form_value)
            except ValueError:
                app.logger.warning(f"单样本预测表单输入错误: 特征 '{f_name}' 的值 '{form_value}' 不是有效数字。")
                model_cache['single_pred_error_to_display'] = f"特征 '{f_name}' 的输入无效, 请输入一个数字。当前值为: '{form_value}'" # 中文错误信息
                model_cache['default_values'] = current_inputs_for_form
                model_cache['show_single_pred_results'] = True
                return redirect(url_for('index'))
        
        df_input = pd.DataFrame([input_data_dict], columns=feature_names)
        app.logger.info(f"单样本预测输入数据: {df_input.to_dict(orient='records')}") # 中文日志

        prob_ok_array = clf.predict_proba(df_input)[:, ok_label_numeric_val]
        prob_ok_scalar = prob_ok_array[0] if prob_ok_array.ndim > 0 else prob_ok_array
        
        # 使用固定的阈值进行预测
        predicted_numeric_label = ok_label_numeric_val if prob_ok_scalar >= fixed_threshold else ng_label_numeric_val
        
        pred_label_text = "✅ 合格 (OK)" if predicted_numeric_label == ok_label_numeric_val else "❌ 不合格 (NG)" # 中文标签
        app.logger.info(f"单样本预测概率 (OK): {prob_ok_scalar:.3f}, 预测数值标签: {predicted_numeric_label}, 文本标签: {pred_label_text}") # 中文日志

        if predicted_numeric_label == ng_label_numeric_val:
            explainer_instance = shap.Explainer(clf, X_background)
            shap_values_instance = explainer_instance(df_input)
            model_cache['base64_waterfall_plot'] = generate_shap_waterfall_base64(shap_values_instance[0])
            app.logger.info("为不合格样本生成了 SHAP Waterfall 图。") # 中文日志
        else:
            model_cache['base64_waterfall_plot'] = None
            app.logger.info("合格样本, 不生成 SHAP Waterfall 图。") # 中文日志

        if predicted_numeric_label == ng_label_numeric_val:
            # 确保 explainer_instance 和 shap_values_instance 存在
            if 'explainer_instance' not in locals(): 
                 explainer_instance = shap.Explainer(clf, X_background)
                 shap_values_instance = explainer_instance(df_input)

            shap_values_for_df = shap_values_instance.values[0]
            base_value_for_df = shap_values_instance.base_values[0]
            shap_df_data = pd.DataFrame({
                "Feature": df_input.columns,
                "SHAP Value": shap_values_for_df
            })
            predicted_logit = base_value_for_df + np.sum(shap_values_for_df)
            impacts = []
            for _, shap_val_feature in enumerate(shap_values_for_df):
                impact = sigmoid(predicted_logit) - sigmoid(predicted_logit - shap_val_feature)
                impacts.append(impact)
            shap_df_data["Impact on Probability"] = impacts
            shap_df_data = shap_df_data.sort_values(by="SHAP Value", key=abs, ascending=False).head(5)
            model_cache['single_pred_shap_table_html'] = shap_df_data.to_html(classes='table table-sm table-striped table-hover', index=False, float_format='{:.4f}'.format)
            app.logger.info("单样本 SHAP 表格数据分析完成。") # 中文日志
        else:
            model_cache['single_pred_shap_table_html'] = None

        model_cache['show_results_from_upload'] = True
        model_cache['show_single_pred_results'] = True
        model_cache['single_pred_input_data_html'] = pd.DataFrame([input_data_dict]).to_html(classes='table table-sm table-bordered', index=False, float_format='{:.2f}'.format)
        model_cache['single_pred_prob'] = f"{prob_ok_scalar:.3f}"
        model_cache['single_pred_label'] = pred_label_text
        model_cache['single_pred_best_thresh'] = f"{fixed_threshold:.1f}" # 显示固定的阈值
        model_cache['default_values'] = df_input.iloc[0].to_dict()
        model_cache.pop('single_pred_error_to_display', None)

        return redirect(url_for('index'))

    except Exception as e:
        app.logger.error(f"'/predict_single' 发生严重错误: {e}", exc_info=True)
        model_cache['show_results_from_upload'] = True
        model_cache['show_single_pred_results'] = True
        model_cache['single_pred_error_to_display'] = f"单次预测过程中发生严重错误: {str(e)}" # 中文错误信息
        return redirect(url_for('index'))

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.logger.info(f"应用启动中, 监听地址 0.0.0.0, 端口: {port}") # 中文日志
    app.run(host='0.0.0.0', port=port, debug=False)
