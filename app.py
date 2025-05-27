import base64
import io
import os

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
# from sklearn.preprocessing import LabelEncoder # XGBoost 会自动处理标签编码（如果 y 是字符串）
from sklearn.utils.class_weight import compute_sample_weight # 计算样本权重

# Flask 应用初始化
app = Flask(__name__)
app.secret_key = os.urandom(24) # 用于会话管理（在此示例中未大量使用）

# 全局缓存，用于存储模型和数据（简化版，用于演示）
# 在生产环境的多用户应用中，应以不同方式处理（例如，会话、数据库）
model_cache = {}

# Matplotlib 字体设置 (确保 Render 上有可用字体，或使用系统默认字体)
try:
    # 尝试使用无衬线字体作为通用后备
    matplotlib.rcParams['font.family'] = 'sans-serif'
    matplotlib.rcParams['axes.unicode_minus'] = False # 解决负号显示问题
except Exception as e:
    print(f"字体设置错误: {e}。Matplotlib 将使用默认字体。")

def fig_to_base64(fig):
    """将 Matplotlib 图像转换为 base64 编码的 PNG。"""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight') # bbox_inches='tight' 避免图像边缘被裁剪
    plt.close(fig) # 关闭图像，释放内存
    buf.seek(0) # 将指针移到缓冲区的开头
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def generate_shap_waterfall_base64(shap_values_single_instance):
    """生成 SHAP waterfall 图并以 base64 格式返回。"""
    # shap_values_single_instance 应为单个实例的 shap.Explanation 对象
    fig = plt.figure() # 创建新的图像上下文
    shap.plots.waterfall(shap_values_single_instance, show=False, max_display=15) # max_display 控制显示特征数量
    plt.tight_layout() # 自动调整子图参数，使之填充整个图像区域
    base64_str = fig_to_base64(fig)
    return base64_str

@app.route('/', methods=['GET', 'POST'])
def index():
    global model_cache # 允许修改全局变量 model_cache
    if request.method == 'GET':
        # 清空缓存以重新开始，或当返回上传页面时
        # model_cache = {} # 评论掉这一行，以便在单个预测后返回时保留之前上传和训练的结果
        # 只有在用户明确回到主页（比如点击“上传新文件”链接）时才应该清空
        # 如果这是用户首次访问或通过链接导航到 '/'，则可能是新会话的开始
        if not model_cache.get('show_results'): # 如果不是因为提交了单样本预测而重新加载页面
            model_cache = {}

        return render_template('index.html',
                               show_results=model_cache.get('show_results', False),
                               show_single_pred_results=model_cache.get('show_single_pred_results', False),
                               filename=model_cache.get('filename'),
                               form_inputs=model_cache.get('form_inputs'),
                               base64_perf_plot=model_cache.get('base64_perf_plot'),
                               metrics_df_html=model_cache.get('metrics_df_html'),
                               best_recommendation_html=model_cache.get('best_recommendation_html'),
                               recommended_threshold_text=model_cache.get('recommended_threshold_text'),
                               base64_global_shap_plot=model_cache.get('base64_global_shap_plot'),
                               default_values=model_cache.get('default_values'),
                               single_pred_input_data_html=model_cache.get('single_pred_input_data_html'),
                               single_pred_prob=model_cache.get('single_pred_prob'),
                               single_pred_label=model_cache.get('single_pred_label'),
                               single_pred_best_thresh=model_cache.get('single_pred_best_thresh'),
                               single_pred_shap_table_html=model_cache.get('single_pred_shap_table_html'),
                               base64_waterfall_plot=model_cache.get('base64_waterfall_plot')
                               )


    if request.method == 'POST': # 此 POST 用于文件上传和初始处理
        model_cache = {} # 每次上传新文件时，清空旧的缓存

        if 'file' not in request.files:
            return "请求中没有文件部分。", 400
        file = request.files['file']
        if file.filename == '':
            return "未选择文件。", 400

        if file and file.filename.endswith('.csv'): # 确保是 CSV 文件
            try:
                filename = file.filename
                df = pd.read_csv(file)
                model_cache['filename'] = filename # 缓存文件名

                # --- 数据预处理和模型训练 (来自 Colab 脚本) ---
                if "OK_NG" not in df.columns:
                    return "上传的 CSV 文件必须包含 'OK_NG' 列。", 400
                
                X = df.drop("OK_NG", axis=1).copy() # 特征
                # 确保所有特征列都是数值类型，如果不是，尝试转换或给出错误提示
                try:
                    X = X.astype(float) # 尝试将所有特征列转换为数值型
                except ValueError as e:
                    non_numeric_cols = [col for col in X.columns if not pd.api.types.is_numeric_dtype(X[col])]
                    return f"以下特征列无法转换为数值类型，请检查数据: {', '.join(non_numeric_cols)}. 错误: {e}", 400


                y_raw = df["OK_NG"].copy() # 原始目标变量

                # 确保 'OK_NG' 列包含期望的值
                if not y_raw.isin(['OK', 'NG']).all(): # 检查是否所有值都在 ['OK', 'NG'] 中
                    return "列 'OK_NG' 必须只包含 'OK' 或 'NG' 值。", 400

                # 编码 y: 'OK' 应为 1 (正类), 'NG' 应为 0
                ok_label_numeric = 1
                ng_label_numeric = 0
                y_numeric = y_raw.map({'OK': ok_label_numeric, 'NG': ng_label_numeric}).astype(int)
                
                # 存储特征名称，用于后续的预测表单
                model_cache['feature_names'] = list(X.columns)
                model_cache['X_train_df_for_explainer'] = X.copy() # 用于 SHAP 的背景数据集

                # 样本权重
                # class_weight 参数的键应该是数值标签
                weights = compute_sample_weight(class_weight={ng_label_numeric: 1.0, ok_label_numeric: 2.0}, y=y_numeric)


                # XGBoost 分类器
                clf = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                                        objective="binary:logistic", eval_metric="logloss",
                                        random_state=42, use_label_encoder=False) # use_label_encoder=False 推荐
                clf.fit(X, y_numeric, sample_weight=weights)
                model_cache['clf'] = clf # 缓存训练好的模型

                # --- 阈值扫描 ---
                probs = clf.predict_proba(X)[:, ok_label_numeric] # 'OK' 类的概率
                thresholds = np.arange(0.1, 0.91, 0.01) # 扫描的阈值范围
                metrics_list = []
                for t in thresholds:
                    preds = (probs >= t).astype(int) # 根据阈值进行预测
                    # classification_report 需要 y_true 和 y_pred 都是数值型
                    report = classification_report(y_numeric, preds, output_dict=True, zero_division=0,
                                                   labels=[ng_label_numeric, ok_label_numeric], # 指定标签顺序
                                                   target_names=['NG', 'OK']) # 指定标签名称
                    metrics_list.append({
                        "threshold": t,
                        "accuracy": report["accuracy"],
                        "NG_recall": report["NG"]["recall"],
                        "OK_recall": report["OK"]["recall"],
                        "OK_precision": report["OK"]["precision"],
                        "f1_score_weighted": f1_score(y_numeric, preds, average='weighted', zero_division=0) # 加权平均 F1
                    })
                metrics_df = pd.DataFrame(metrics_list)
                model_cache['metrics_df'] = metrics_df # 缓存指标数据
                model_cache['metrics_df_html'] = metrics_df.to_html(classes='table table-sm table-striped table-hover', index=False, float_format='{:.2f}'.format)


                # 筛选满足 OK 召回率 >= 0.95 的阈值
                filtered_metrics = metrics_df[metrics_df["OK_recall"] >= 0.95]
                best_recommendation_df = pd.DataFrame() # 如果没有合适的阈值，则为空 DataFrame
                if not filtered_metrics.empty:
                    best_recommendation_df = filtered_metrics.sort_values(
                        ["NG_recall", "f1_score_weighted"], ascending=[False, False] # 优先 NG 召回率，其次 F1
                    ).head(1)
                    best_thresh = best_recommendation_df["threshold"].values[0]
                    recommended_threshold_text = (
                        f"✅ 推荐分类阈值为：{best_thresh:.2f}\n"
                        f"   - NG召回率: {best_recommendation_df['NG_recall'].values[0]:.2f}\n"
                        f"   - OK召回率: {best_recommendation_df['OK_recall'].values[0]:.2f}\n"
                        f"   - F1 分数 (加权): {best_recommendation_df['f1_score_weighted'].values[0]:.2f}"
                    )
                    model_cache['best_recommendation_html'] = best_recommendation_df.to_html(classes='table table-sm table-striped', index=False, float_format='{:.2f}'.format)
                else:
                    best_thresh = 0.5 # 如果没有阈值满足条件，则使用默认值 0.5
                    recommended_threshold_text = "⚠️ 没有找到 OK召回率≥0.95 的推荐阈值。将使用默认阈值0.5进行预测。"
                    model_cache['best_recommendation_html'] = "<p>无推荐阈值满足条件。</p>"
                
                model_cache['best_thresh'] = best_thresh # 缓存最佳阈值
                model_cache['best_recommendation_df'] = best_recommendation_df # 缓存最佳推荐的详细指标
                model_cache['recommended_threshold_text'] = recommended_threshold_text.replace("\n", "<br>") # 缓存推荐文本


                # --- 性能图绘制 ---
                fig_perf, ax_perf = plt.subplots(figsize=(10, 6))
                ax_perf.plot(metrics_df["threshold"], metrics_df["NG_recall"], label="NG 召回率", color="red")
                ax_perf.plot(metrics_df["threshold"], metrics_df["OK_recall"], label="OK 召回率", color="green")
                ax_perf.plot(metrics_df["threshold"], metrics_df["f1_score_weighted"], label="F1 分数 (加权)", color="blue")
                if not best_recommendation_df.empty: # 如果找到了推荐阈值，则在图上标记
                    ax_perf.axvline(x=best_thresh, color="purple", linestyle="--", label=f"推荐阈值: {best_thresh:.2f}")
                ax_perf.set_xlabel("分类阈值")
                ax_perf.set_ylabel("指标值")
                ax_perf.set_title("XGBoost 模型性能 vs. 阈值")
                ax_perf.legend()
                ax_perf.grid(True)
                plt.tight_layout()
                base64_perf_plot = fig_to_base64(fig_perf)
                model_cache['base64_perf_plot'] = base64_perf_plot # 缓存性能图的 base64 编码

                # --- 全局 SHAP 特征重要性图 ---
                explainer_global = shap.Explainer(clf, X) # X 是背景数据集
                shap_values_all = explainer_global(X) # 计算所有训练样本的 SHAP 值
                
                fig_global_shap, ax_global_shap = plt.subplots() # 为 SHAP bar 图创建新的图像
                shap.plots.bar(shap_values_all, show=False, max_display=15) # show=False 避免在服务器端显示
                plt.tight_layout()
                base64_global_shap_plot = fig_to_base64(fig_global_shap)
                model_cache['base64_global_shap_plot'] = base64_global_shap_plot # 缓存全局 SHAP 图
                
                model_cache['form_inputs'] = model_cache['feature_names']
                model_cache['default_values'] = X.mean().to_dict()
                model_cache['show_results'] = True
                model_cache['show_single_pred_results'] = False # 初始时不显示单样本预测结果

                # 直接调用 index 的 GET 逻辑来渲染页面
                return redirect(url_for('index'))


            except Exception as e:
                app.logger.error(f"处理文件时发生错误: {e}", exc_info=True) # 记录详细错误信息
                # 清理缓存，防止污染下一次尝试
                model_cache = {}
                return render_template('index.html', error_message=f"处理文件时发生错误: {str(e)}", show_results=False, show_single_pred_results=False)

        else: # 文件类型不是 CSV
             model_cache = {}
             return render_template('index.html', error_message="文件类型无效。请上传 CSV 文件。", show_results=False, show_single_pred_results=False)
    
    # 如果不是 GET 或 POST (理论上不应发生)，则重定向回主页
    # 或者直接渲染初始页面
    model_cache = {} # 确保从干净的状态开始
    return render_template('index.html', show_results=False, show_single_pred_results=False)

@app.route('/predict_single', methods=['POST'])
def predict_single():
    global model_cache
    if 'clf' not in model_cache: # 检查模型是否已训练
        # 如果模型不存在（可能因为用户直接访问此URL或会话丢失），重定向到主页
        return redirect(url_for('index'))

    try:
        clf = model_cache['clf']
        best_thresh = model_cache['best_thresh']
        X_background = model_cache['X_train_df_for_explainer'] # 使用缓存的训练数据作为背景
        feature_names = model_cache['feature_names']

        input_data_dict = {}
        for f_name in feature_names:
            form_value = request.form.get(f_name)
            if form_value is None or form_value.strip() == "":
                 return render_template('index.html', **model_cache, single_pred_error=f"特征 '{f_name}' 的值不能为空。")
            try:
                input_data_dict[f_name] = float(form_value)
            except ValueError:
                return render_template('index.html', **model_cache, single_pred_error=f"特征 '{f_name}' 的输入无效，请输入一个数字。当前值为: '{form_value}'")
        
        df_input = pd.DataFrame([input_data_dict], columns=feature_names)

        prob_ok = clf.predict_proba(df_input)[0][1] # 假设类别 1 是 'OK'
        pred_label_text = "✅ 合格 (OK)" if prob_ok >= best_thresh else "❌ 不合格 (NG)"

        explainer_instance = shap.Explainer(clf, X_background)
        shap_values_instance = explainer_instance(df_input)

        base64_waterfall_plot = generate_shap_waterfall_base64(shap_values_instance[0])

        shap_df_data = pd.DataFrame({
            "特征": df_input.columns,
            "SHAP 值": shap_values_instance.values[0]
        })
        
        base_value_logit = shap_values_instance.base_values[0]
        predicted_logit = base_value_logit + np.sum(shap_values_instance.values[0])
        
        impacts = []
        for _, shap_val_feature in enumerate(shap_values_instance.values[0]):
            impact = sigmoid(predicted_logit) - sigmoid(predicted_logit - shap_val_feature)
            impacts.append(impact)
        shap_df_data["对概率的影响"] = impacts
        
        shap_df_data = shap_df_data.sort_values(by="SHAP 值", key=abs, ascending=False).head(5)
        
        # 更新 model_cache 以包含单次预测的结果
        model_cache['show_single_pred_results'] = True
        model_cache['single_pred_input_data_html'] = pd.DataFrame([input_data_dict]).to_html(classes='table table-sm table-bordered', index=False, float_format='{:.2f}'.format)
        model_cache['single_pred_prob'] = f"{prob_ok:.3f}"
        model_cache['single_pred_label'] = pred_label_text
        model_cache['single_pred_best_thresh'] = f"{best_thresh:.2f}"
        model_cache['single_pred_shap_table_html'] = shap_df_data.to_html(classes='table table-sm table-striped table-hover', index=False, float_format='{:.4f}'.format)
        model_cache['base64_waterfall_plot'] = base64_waterfall_plot
        model_cache['default_values'] = df_input.iloc[0].to_dict() # 更新表单的默认值为当前输入

        return redirect(url_for('index')) # 重定向到 index，index的GET会从cache加载并渲染

    except Exception as e:
        app.logger.error(f"/predict_single 发生错误: {e}", exc_info=True)
        # 尝试保留大部分UI状态，但显示错误
        model_cache['show_single_pred_results'] = True # 仍然尝试显示单预测部分
        model_cache['single_pred_error'] = f"单次预测过程中发生错误: {str(e)}"
        return redirect(url_for('index'))


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000)) # Render 会设置 PORT 环境变量
    # 在 Render 上生产环境应设置 debug=False
    app.run(host='0.0.0.0', port=port, debug=False) # 生产环境推荐 debug=False
