import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# 贝叶斯优化
from hyperopt import fmin, tpe, hp, Trials, space_eval

# 数据划分/交叉验证
from sklearn.model_selection import (
    train_test_split,
    KFold,
    RepeatedKFold,
    cross_val_score
)

# 模型
from sklearn.ensemble import GradientBoostingRegressor

# 评估指标
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# 数据封装
from sklearn.utils import Bunch


# ========================== 1. 自定义评估指标 ==========================
def MAPE(true, pred):
    true = np.array(true)
    pred = np.array(pred)
    return np.mean(np.abs((true - pred) / (np.abs(true) + 1e-8))) * 100


def calc_metrics(y_true, y_pred):
    """统一计算评估指标"""
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = MAPE(y_true, y_pred)
    return r2, rmse, mae, mape


def evaluate_cv_details(X, y, params, cv, n_splits):
    """
    逐折训练并记录每一折指标
    n_splits:
        单次10折时传10
        Repeated 5-Fold时传5
    """
    records = []

    # 保证 X 是 DataFrame，便于 iloc
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    X = X.reset_index(drop=True)

    # 保证 y 是 numpy array
    y = np.array(y)

    for i, (train_idx, val_idx) in enumerate(cv.split(X, y), start=1):
        X_tr = X.iloc[train_idx]
        X_val = X.iloc[val_idx]
        y_tr = y[train_idx]
        y_val = y[val_idx]

        model = GradientBoostingRegressor(**params, random_state=123)
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_val)

        r2, rmse, mae, mape = calc_metrics(y_val, y_pred)

        records.append({
            'split_id': i,
            'repeat': (i - 1) // n_splits + 1,
            'fold': (i - 1) % n_splits + 1,
            'n_val': len(val_idx),
            'R2': r2,
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape
        })

    return pd.DataFrame(records)


def summarize_cv_results(df_metrics):
    """汇总 CV 指标：mean/std/min/max/CV%"""
    summary = df_metrics[['R2', 'RMSE', 'MAE', 'MAPE']].agg(['mean', 'std', 'min', 'max']).T
    summary['CV_%'] = summary['std'] / summary['mean'].abs() * 100
    return summary


# ========================== 2. 读取数据 + 封装数据集 ==========================
df = pd.read_excel('数据集defect frequency.xlsx')

# 目标列
target_col = 'defect frequency'

# 要剔除的列（这里只有目标列）
data_feature = [target_col]

bunch = Bunch(
    data=df.drop(data_feature, axis=1).values,
    target=df[target_col].values,
    feature_names=df.drop(data_feature, axis=1).columns.tolist(),
    target_names=[target_col]
)

X = pd.DataFrame(bunch.data, columns=bunch.feature_names)
y = bunch.target

# 划分训练集/测试集（固定种子，可复现）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1
)

print("=" * 60)
print(f"总样本数: {len(X)}")
print(f"训练集样本数: {len(X_train)}")
print(f"测试集样本数: {len(X_test)}")
print("=" * 60)


# ========================== 3. 基准模型（手动参数训练） ==========================
print("手动参数基准模型结果：")
base_model = GradientBoostingRegressor(
    random_state=123,
    n_estimators=260,
    max_depth=3,
    learning_rate=0.08033607156844087,
    min_samples_split=7,
    subsample=0.8936623050720786,
    min_samples_leaf=6
)

base_model.fit(X_train, y_train)

print(f"训练集 R²: {base_model.score(X_train, y_train):.4f}")
print(f"测试集 R²: {base_model.score(X_test, y_test):.4f}")
print("=" * 60)


# ========================== 4. Hyperopt 贝叶斯优化 ==========================
space = {
    'learning_rate': hp.uniform('learning_rate', 0.02, 0.15),

    # 2 ~ 3，包含 2 和 3
    'max_depth': hp.choice('max_depth', [1,2, 3]),

    # 100 ~ 500
    'n_estimators': hp.choice('n_estimators', list(range(10, 500, 10))),

    # 5 ~ 15
    'min_samples_split': hp.choice('min_samples_split', list(range(4, 16))),

    # 2 ~ 8
    'min_samples_leaf': hp.choice('min_samples_leaf', list(range(2, 9))),

    # 0.7 ~ 1.0
    'subsample': hp.uniform('subsample', 0.7, 1.0)
}


def objective(params):
    model = GradientBoostingRegressor(**params, random_state=123)

    # 调参阶段：仍然使用单次10折
    kfold = KFold(n_splits=10, shuffle=True, random_state=1)

    cv_score = cross_val_score(
        model,
        X_train,
        y_train,
        cv=kfold,
        scoring='r2'
    ).mean()

    return {'loss': -cv_score, 'status': 'ok'}


trials = Trials()
print("开始贝叶斯超参数优化...")

best = fmin(
    fn=objective,
    space=space,
    algo=tpe.suggest,
    max_evals=500,
    trials=trials,
    show_progressbar=True,
    catch_eval_exceptions=True
)

best_params = space_eval(space, best)

print("=" * 60)
print("Hyperopt 最优超参数：")
print(best_params)
print("=" * 60)
''''# ========================== 检查第3折为什么 R2 很低 ==========================
print("\n" + "=" * 60)
print("检查单次10折中第1折的验证样本误差")
print("=" * 60)

kf10 = KFold(n_splits=10, shuffle=True, random_state=1)

for fold, (train_idx, val_idx) in enumerate(kf10.split(X_train, y_train), start=1):
    if fold == 1:
        X_tr = X_train.iloc[train_idx]
        X_val = X_train.iloc[val_idx]
        y_tr = y_train[train_idx]
        y_val = y_train[val_idx]

        model = GradientBoostingRegressor(**best_params, random_state=123)
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_val)

        check_df = X_val.copy()

        # 原始 Excel 行号：pandas index + 2
        check_df.insert(0, 'original_index', X_val.index)
        check_df.insert(1, 'excel_row', X_val.index + 2)

        check_df['y_true'] = y_val
        check_df['y_pred'] = y_pred
        check_df['error'] = y_pred - y_val
        check_df['abs_error'] = np.abs(y_pred - y_val)
        check_df['APE_%'] = np.abs((y_val - y_pred) / (np.abs(y_val) + 1e-8)) * 100

        # 每个样本对总平方误差 SSE 的贡献
        sse_each = (y_val - y_pred) ** 2
        check_df['SSE_contribution_%'] = sse_each / np.sum(sse_each) * 100

        print("\n第1折整体指标：")
        r2, rmse, mae, mape = calc_metrics(y_val, y_pred)
        print(f"R2   = {r2:.6f}")
        print(f"RMSE = {rmse:.6f}")
        print(f"MAE  = {mae:.6f}")
        print(f"MAPE = {mape:.6f}%")

        print("\n第1折验证样本，按绝对误差从大到小排序：")
        print(
            check_df.sort_values('abs_error', ascending=False)
            .to_string(index=False)
        )

        print("\n去掉单个样本后，第1折 R2 的变化：")
        loo_records = []
        for i in range(len(y_val)):
            mask = np.ones(len(y_val), dtype=bool)
            mask[i] = False

            r2_without = r2_score(y_val[mask], y_pred[mask])

            loo_records.append({
                'removed_excel_row': X_val.index[i] + 2,
                'removed_original_index': X_val.index[i],
                'removed_y_true': y_val[i],
                'removed_y_pred': y_pred[i],
                'removed_abs_error': abs(y_pred[i] - y_val[i]),
                'R2_after_removing_this': r2_without
            })

        loo_df = pd.DataFrame(loo_records)
        print(
            loo_df.sort_values('R2_after_removing_this', ascending=False)
            .to_string(index=False)
        )

        break'''


# ========================== 5. 用最优参数训练最终模型 ==========================
best_model = GradientBoostingRegressor(**best_params, random_state=123)
best_model.fit(X_train, y_train)


# ========================== 6. 最优模型测试集评估 ==========================
print("最优模型测试集评估结果：")
y_pred = best_model.predict(X_test)

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
mape = MAPE(y_test, y_pred)

print(f"R² 决定系数: {r2:.4f}")
print(f"RMSE 均方根误差: {rmse:.4f}")
print(f"MAE 平均绝对误差: {mae:.4f}")
print(f"MAPE 平均绝对百分比误差: {mape:.4f}%")
print("=" * 60)
# ========================== 6.1 测试集误差最大样本分析 ==========================
print("\n" + "=" * 60)
print("测试集误差最大样本分析")
print("=" * 60)

# y_pred 已经在第6部分得到：y_pred = best_model.predict(X_test)
error_df = X_test.copy()

# 原始 Excel 行号：pandas index + 2
# 因为 Excel 第1行通常是表头，所以数据从第2行开始
error_df.insert(0, "original_index", X_test.index)
error_df.insert(1, "excel_row", X_test.index + 2)

error_df["y_true"] = y_test
error_df["y_pred"] = y_pred
error_df["error"] = error_df["y_pred"] - error_df["y_true"]
error_df["abs_error"] = np.abs(error_df["error"])
error_df["APE_%"] = np.abs(
    error_df["error"] / (np.abs(error_df["y_true"]) + 1e-8)
) * 100

# 每个测试样本对测试集总平方误差 SSE 的贡献
sse_each = (error_df["y_true"] - error_df["y_pred"]) ** 2
error_df["SSE_contribution_%"] = sse_each / sse_each.sum() * 100

# 按绝对误差从大到小排序
error_df_sorted = error_df.sort_values("abs_error", ascending=False)

print("\n测试集样本按绝对误差从大到小排序：")
print(error_df_sorted.to_string(index=False))

# 保存误差分析结果
error_df_sorted.to_excel("测试集误差分析defect.xlsx", index=False)
print("\n测试集误差分析已保存为：测试集误差分析 defect.xlsx")


# ========================== 6.2 误差最大样本是否集中在某个区域 ==========================
print("\n" + "=" * 60)
print("误差最大样本的设计变量区域分析")
print("=" * 60)

# 取误差最大的前 5 个样本
top_n = 5
top_error_df = error_df_sorted.head(top_n)

feature_cols = X_test.columns.tolist()

print(f"\n误差最大的前 {top_n} 个样本：")
print(
    top_error_df[
        ["excel_row"] + feature_cols + ["y_true", "y_pred", "abs_error", "APE_%", "SSE_contribution_%"]
    ].to_string(index=False)
)

print(f"\n误差最大的前 {top_n} 个样本的变量统计：")
print(top_error_df[feature_cols].describe())

print("\n训练集变量统计：")
print(X_train[feature_cols].describe())

print("\n测试集变量统计：")
print(X_test[feature_cols].describe())

# 保存误差最大样本
top_error_df.to_excel("测试集误差最大样本_top5defect.xlsx", index=False)
print("\n误差最大的前5个样本已保存为：测试集误差最大样本_top5.xlsx")


# ========================== 6.3 测试集真实值 vs 预测值图 ==========================
plt.figure(figsize=(6, 5))

plt.scatter(y_test, y_pred, alpha=0.85)

# 画 y = x 理想预测线
min_value = min(np.min(y_test), np.min(y_pred))
max_value = max(np.max(y_test), np.max(y_pred))

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--",
    linewidth=1.5,
    label="Ideal prediction"
)

plt.xlabel("True bandgap")
plt.ylabel("Predicted bandgap")
plt.title(
    "Test set: True vs Predicted bandgap\n"
    f"R2={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}, MAPE={mape:.2f}%"
)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("测试集_真实值_vs_预测值_defect.png", dpi=300)
plt.show()


# ========================== 6.4 训练集和测试集真实值 vs 预测值对比图 ==========================
y_train_pred = best_model.predict(X_train)

r2_train = r2_score(y_train, y_train_pred)
rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
mae_train = mean_absolute_error(y_train, y_train_pred)
mape_train = MAPE(y_train, y_train_pred)

plt.figure(figsize=(6, 5))

plt.scatter(y_train, y_train_pred, alpha=0.65, label="Train set")
plt.scatter(y_test, y_pred, alpha=0.85, label="Test set")

all_true = np.concatenate([y_train, y_test])
all_pred = np.concatenate([y_train_pred, y_pred])

min_value = min(np.min(all_true), np.min(all_pred))
max_value = max(np.max(all_true), np.max(all_pred))

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--",
    linewidth=1.5,
    label="Ideal prediction"
)

plt.xlabel("True bandgap")
plt.ylabel("Predicted bandgap")
plt.title(
    "True vs Predicted bandgap\n"
    f"Train R2={r2_train:.4f}, Test R2={r2:.4f}"
)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("训练集测试集_真实值_vs_预测值_defect.png", dpi=300)
plt.show()


# ========================== 7. 单次 10 折：查看各折波动 ==========================
print("单次 10 折交叉验证：各折波动分析")

kf10 = KFold(n_splits=10, shuffle=True, random_state=1)

df_10fold = evaluate_cv_details(
    X=X_train,
    y=y_train,
    params=best_params,
    cv=kf10,
    n_splits=10
)

print("\n[单次10折] 各折结果：")
print(df_10fold[['fold', 'n_val', 'R2', 'RMSE', 'MAE', 'MAPE']].to_string(index=False))

summary_10fold = summarize_cv_results(df_10fold)
print("\n[单次10折] 汇总统计（mean/std/min/max/CV%）：")
print(summary_10fold)

print("\n[单次10折] R² 最差的 3 折：")
print(df_10fold.sort_values('R2', ascending=True).head(3)[
    ['fold', 'n_val', 'R2', 'RMSE', 'MAE', 'MAPE']
].to_string(index=False))

print("\n[单次10折] MAPE 最差的 3 折：")
print(df_10fold.sort_values('MAPE', ascending=False).head(3)[
    ['fold', 'n_val', 'R2', 'RMSE', 'MAE', 'MAPE']
].to_string(index=False))

print("=" * 60)


# ========================== 8. 重复交叉验证：查看整体稳定性 ==========================
print("Repeated 5-Fold CV：稳定性分析")

# 这里推荐 5折 × 20次重复
# 因为你样本较少，5折比10折每次验证集稍大，波动会更稳一些
rkf = RepeatedKFold(n_splits=5, n_repeats=20, random_state=42)

df_repeated = evaluate_cv_details(
    X=X_train,
    y=y_train,
    params=best_params,
    cv=rkf,
    n_splits=5
)

print("\n[Repeated 5-Fold] 前10个 split 结果：")
print(df_repeated.head(10).to_string(index=False))

summary_repeated = summarize_cv_results(df_repeated)
print("\n[Repeated 5-Fold] 全部 split 汇总统计（mean/std/min/max/CV%）：")
print(summary_repeated)

# 每次重复（每5折）先求平均，再看重复之间的波动
repeat_mean = df_repeated.groupby('repeat')[['R2', 'RMSE', 'MAE', 'MAPE']].mean()

print("\n[Repeated 5-Fold] 每次重复的平均指标：")
print(repeat_mean)

repeat_summary = repeat_mean.agg(['mean', 'std', 'min', 'max']).T
repeat_summary['CV_%'] = repeat_summary['std'] / repeat_summary['mean'].abs() * 100

print("\n[Repeated 5-Fold] 各次重复之间的汇总统计：")
print(repeat_summary)

print("=" * 60)


# ========================== 9. 可选：保存结果到 Excel ==========================
# 取消下面注释即可导出
# df_10fold.to_excel('单次10折各折结果.xlsx', index=False)
# summary_10fold.to_excel('单次10折汇总统计.xlsx')
# df_repeated.to_excel('Repeated5Fold_全部split结果.xlsx', index=False)
# repeat_mean.to_excel('Repeated5Fold_每次重复平均结果.xlsx')
# repeat_summary.to_excel('Repeated5Fold_汇总统计.xlsx')