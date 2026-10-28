import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error


# =========================
# 1. 设计变量名称和范围
# =========================
feature_names = ["steel_r_ratio", "steel_h_ratio", "a_ratio"]

lower_bounds = np.array([0.5, 0.5, 0.5])
upper_bounds = np.array([0.8, 1.0, 1.0])


# =========================
# 2. defect frequency 的预测模型
# =========================
df_defect_frequency = pd.read_excel("数据集defect frequency.xlsx")

target_name = "defect frequency"
X_defect_frequency = df_defect_frequency[feature_names]
y_defect_frequency = df_defect_frequency[target_name].values

X_train_defect, X_test_defect, y_train_defect, y_test_defect = train_test_split(
    X_defect_frequency,
    y_defect_frequency,
    test_size=0.2,
    random_state=1
)

GBRT_defect_frequency = GradientBoostingRegressor(
    random_state=123,
    n_estimators=420,
    max_depth=3,
    learning_rate=0.10786137850317136,
    min_samples_leaf=3,
    min_samples_split=5,
    subsample=0.7005366209831134

)

GBRT_defect_frequency.fit(X_train_defect, y_train_defect)

y_pred_defect_test = GBRT_defect_frequency.predict(X_test_defect)

r2_defect = r2_score(y_test_defect, y_pred_defect_test)
rmse_defect = np.sqrt(mean_squared_error(y_test_defect, y_pred_defect_test))
mae_defect = mean_absolute_error(y_test_defect, y_pred_defect_test)
mape_defect = mean_absolute_percentage_error(
    y_test_defect,
    y_pred_defect_test
) * 100

print("defect_frequency 模型 feature_names =", feature_names)
print("defect_frequency R2 =", r2_defect)
print("defect_frequency RMSE =", rmse_defect)
print("defect_frequency MAE =", mae_defect)
print("defect_frequency MAPE =", mape_defect, "%")
# 对训练集进行预测
y_pred_defect_train = GBRT_defect_frequency.predict(X_train_defect)

# 构建 DataFrame
defect_train_df = pd.DataFrame({
    "真实值": y_train_defect,
    "预测值": y_pred_defect_train
})
defect_test_df = pd.DataFrame({
    "真实值": y_test_defect,
    "预测值": y_pred_defect_test
})

# 导出 Excel（包含两个 sheet）
with pd.ExcelWriter("defect_frequency_预测效果.xlsx") as writer:
    defect_train_df.to_excel(writer, sheet_name="训练集", index=False)
    defect_test_df.to_excel(writer, sheet_name="测试集", index=False)

print("defect frequency 预测效果已保存：defect_frequency_预测效果.xlsx")

# =========================
# 3. bandgap 的预测模型
# =========================
df_bandgap = pd.read_excel("数据集bandgap.xlsx")

target_name = "bandgap"
X_bandgap = df_bandgap[feature_names]
y_bandgap = df_bandgap[target_name].values

X_train_bandgap, X_test_bandgap, y_train_bandgap, y_test_bandgap = train_test_split(
    X_bandgap,
    y_bandgap,
    test_size=0.2,
    random_state=1
)

GBRT_bandgap = GradientBoostingRegressor(
    random_state=123,
    n_estimators=430,
    max_depth=2,
    learning_rate=0.14661217949969257,
    min_samples_split=7,
    min_samples_leaf=3,
    subsample=0.9431248918637649
)

GBRT_bandgap.fit(X_train_bandgap, y_train_bandgap)

y_pred_bandgap_test = GBRT_bandgap.predict(X_test_bandgap)

r2_bandgap = r2_score(y_test_bandgap, y_pred_bandgap_test)
rmse_bandgap = np.sqrt(mean_squared_error(y_test_bandgap, y_pred_bandgap_test))
mae_bandgap = mean_absolute_error(y_test_bandgap, y_pred_bandgap_test)
mape_bandgap = mean_absolute_percentage_error(
    y_test_bandgap,
    y_pred_bandgap_test
) * 100

print("bandgap 模型 feature_names =", feature_names)
print("bandgap R2 =", r2_bandgap)
print("bandgap RMSE =", rmse_bandgap)
print("bandgap MAE =", mae_bandgap)
print("bandgap MAPE =", mape_bandgap, "%")

# 对训练集进行预测
y_pred_bandgap_train = GBRT_bandgap.predict(X_train_bandgap)

# 构建 DataFrame
bandgap_train_df = pd.DataFrame({
    "真实值": y_train_bandgap,
    "预测值": y_pred_bandgap_train
})
bandgap_test_df = pd.DataFrame({
    "真实值": y_test_bandgap,
    "预测值": y_pred_bandgap_test
})

# 导出 Excel
with pd.ExcelWriter("bandgap_预测效果.xlsx") as writer:
    bandgap_train_df.to_excel(writer, sheet_name="训练集", index=False)
    bandgap_test_df.to_excel(writer, sheet_name="测试集", index=False)

print("bandgap 预测效果已保存：bandgap_预测效果.xlsx")
# =========================
# 4. mass 计算公式
# =========================
def calculate_mass(steel_ratio, h_ratio, a_ratio):
    mass = (
        392.35 * a_ratio ** 2
        + 1559.49 * steel_ratio ** 2 * h_ratio
        + 94.5 * a_ratio
        + 7.85
    )
    return mass


# =========================
# 5. NSGA-II 多目标优化问题
# =========================
class MyProblem(ElementwiseProblem):
    def __init__(self):
        super().__init__(
            n_var=3,
            n_obj=3,
            xl=lower_bounds,
            xu=upper_bounds
        )

    def _evaluate(self, x, out, *args, **kwargs):
        steel_ratio = x[0]
        h_ratio = x[1]
        a_ratio = x[2]

        x_input = pd.DataFrame([x], columns=feature_names)

        predicted_bandgap = GBRT_bandgap.predict(x_input)[0]
        predicted_defect = GBRT_defect_frequency.predict(x_input)[0]

        mass = calculate_mass(
            steel_ratio=steel_ratio,
            h_ratio=h_ratio,
            a_ratio=a_ratio
        )

        # pymoo 默认是最小化
        # 目标1：最小 mass
        # 目标2：最小 defect frequency
        # 目标3：最大 bandgap，所以写成最小化 -bandgap
        out["F"] = [
            mass,
            predicted_defect,
            -predicted_bandgap
        ]


# =========================
# 6. 创建优化问题和算法
# =========================
problem = MyProblem()

algorithm = NSGA2(
    pop_size=100,
    eliminate_duplicates=True
)


# =========================
# 7. 开始优化
# =========================
result = minimize(
    problem=problem,
    algorithm=algorithm,
    termination=("n_gen", 200),
    seed=1,
    verbose=True
)


# =========================
# 8. 读取 Pareto 结果
# =========================
unique_F, unique_indices = np.unique(result.F, axis=0, return_index=True)
unique_X = result.X[unique_indices]

mass_values = unique_F[:, 0]
defect_values = unique_F[:, 1]
bandgap_values = -unique_F[:, 2]

pareto_df = pd.DataFrame(unique_X, columns=feature_names)
pareto_df["mass"] = mass_values
pareto_df["defect_frequency"] = defect_values
pareto_df["bandgap"] = bandgap_values

pareto_df = pareto_df.sort_values(by="mass", ascending=True)

print("去重后的 Pareto 结果：")
print(pareto_df)

pareto_df.to_excel("Pareto优化结果.xlsx", index=False)
print("Pareto 优化结果已保存为：Pareto优化结果.xlsx")


# =========================
# 9. 可视化 1：代理模型预测精度图
# =========================

# defect frequency: true vs predicted
plt.figure(figsize=(6, 5))
plt.scatter(y_test_defect, y_pred_defect_test, alpha=0.8)

min_value = min(y_test_defect.min(), y_pred_defect_test.min())
max_value = max(y_test_defect.max(), y_pred_defect_test.max())
plt.plot([min_value, max_value], [min_value, max_value], linestyle="--")

plt.xlabel("True defect frequency")
plt.ylabel("Predicted defect frequency")
plt.title(
    "Prediction accuracy of defect frequency\n"
    f"R2={r2_defect:.4f}, RMSE={rmse_defect:.4f}, MAE={mae_defect:.4f}"
)
plt.tight_layout()
plt.savefig("模型精度_defect_frequency.png", dpi=300)
plt.show()


# bandgap: true vs predicted
plt.figure(figsize=(6, 5))
plt.scatter(y_test_bandgap, y_pred_bandgap_test, alpha=0.8)

min_value = min(y_test_bandgap.min(), y_pred_bandgap_test.min())
max_value = max(y_test_bandgap.max(), y_pred_bandgap_test.max())
plt.plot([min_value, max_value], [min_value, max_value], linestyle="--")

plt.xlabel("True bandgap")
plt.ylabel("Predicted bandgap")
plt.title(
    "Prediction accuracy of bandgap\n"
    f"R2={r2_bandgap:.4f}, RMSE={rmse_bandgap:.4f}, MAE={mae_bandgap:.4f}"
)
plt.tight_layout()
plt.savefig("模型精度_bandgap.png", dpi=300)
plt.show()


# =========================
# 10. 可视化 2：Pareto 三维前沿图
# =========================
fig = plt.figure(figsize=(7, 6))
ax = fig.add_subplot(111, projection="3d")

scatter = ax.scatter(
    pareto_df["mass"],
    pareto_df["defect_frequency"],
    pareto_df["bandgap"],
    c=pareto_df["bandgap"],
    s=40,
    alpha=0.85
)

ax.set_xlabel("Mass")
ax.set_ylabel("Defect frequency")
ax.set_zlabel("Bandgap")
ax.set_title("3D Pareto front")

fig.colorbar(scatter, ax=ax, label="Bandgap")

plt.tight_layout()
plt.savefig("Pareto三维前沿图.png", dpi=300)
plt.show()


# =========================
# 11. 可视化 3：Pareto 二维投影图
# =========================

# mass vs defect frequency
plt.figure(figsize=(6, 5))
plt.scatter(
    pareto_df["mass"],
    pareto_df["defect_frequency"],
    c=pareto_df["bandgap"],
    alpha=0.85
)
plt.xlabel("Mass")
plt.ylabel("Defect frequency")
plt.title("Pareto projection: Mass vs Defect frequency")
plt.colorbar(label="Bandgap")
plt.tight_layout()
plt.savefig("Pareto二维投影_mass_defect.png", dpi=300)
plt.show()


# mass vs bandgap
plt.figure(figsize=(6, 5))
plt.scatter(
    pareto_df["mass"],
    pareto_df["bandgap"],
    c=pareto_df["defect_frequency"],
    alpha=0.85
)
plt.xlabel("Mass")
plt.ylabel("Bandgap")
plt.title("Pareto projection: Mass vs Bandgap")
plt.colorbar(label="Defect frequency")
plt.tight_layout()
plt.savefig("Pareto二维投影_mass_bandgap.png", dpi=300)
plt.show()


# defect frequency vs bandgap
plt.figure(figsize=(6, 5))
plt.scatter(
    pareto_df["defect_frequency"],
    pareto_df["bandgap"],
    c=pareto_df["mass"],
    alpha=0.85
)
plt.xlabel("Defect frequency")
plt.ylabel("Bandgap")
plt.title("Pareto projection: Defect frequency vs Bandgap")
plt.colorbar(label="Mass")
plt.tight_layout()
plt.savefig("Pareto二维投影_defect_bandgap.png", dpi=300)
plt.show()


# =========================
# 12. 可视化 4：Pareto 解中设计变量分布
# =========================
plt.figure(figsize=(7, 5))

plt.scatter(
    pareto_df["steel_r_ratio"],
    pareto_df["steel_h_ratio"],
    c=pareto_df["a_ratio"],
    alpha=0.85
)

plt.xlabel("steel_r_ratio")
plt.ylabel("steel_h_ratio")
plt.title("Distribution of Pareto design variables")
plt.colorbar(label="a_ratio")
plt.tight_layout()
plt.savefig("Pareto设计变量分布图.png", dpi=300)
plt.show()


# =========================
# 13. 可视化 5：平行坐标图
# =========================
parallel_df = pareto_df[
    [
        "steel_r_ratio",
        "steel_h_ratio",
        "a_ratio",
        "mass",
        "defect_frequency",
        "bandgap"
    ]
].copy()

# 归一化到 0-1，方便画平行坐标
parallel_df_norm = (
    parallel_df - parallel_df.min()
) / (
    parallel_df.max() - parallel_df.min()
)

plt.figure(figsize=(9, 5))

for i in range(len(parallel_df_norm)):
    plt.plot(
        parallel_df_norm.columns,
        parallel_df_norm.iloc[i],
        alpha=0.35
    )

plt.ylabel("Normalized value")
plt.title("Parallel coordinate plot of Pareto solutions")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("Pareto平行坐标图.png", dpi=300)
plt.show()
# =========================
# =========================
# 14. 选取用于仿真验证的代表性 Pareto 解
# =========================

from sklearn.neighbors import NearestNeighbors

validation_df = pareto_df.copy()


# ---------- 1. 归一化函数 ----------
def minmax_norm(series):
    if series.max() == series.min():
        return series * 0
    return (series - series.min()) / (series.max() - series.min())


# mass 越小越好
validation_df["mass_norm"] = minmax_norm(validation_df["mass"])

# defect_frequency 越小越好
validation_df["defect_norm"] = minmax_norm(validation_df["defect_frequency"])

# bandgap 越大越好，所以转成“越小越好”的归一化形式
validation_df["bandgap_norm"] = (
    validation_df["bandgap"].max() - validation_df["bandgap"]
) / (
    validation_df["bandgap"].max() - validation_df["bandgap"].min()
)

# 防止 bandgap 全部一样导致除零
validation_df["bandgap_norm"] = validation_df["bandgap_norm"].fillna(0)


# ---------- 2. 计算折中得分 ----------
# 加权折中：低 defect frequency 和大 bandgap 更重要，mass 次之
w_mass = 0.2
w_defect = 0.6
w_bandgap = 0.2

validation_df["compromise_score"] = np.sqrt(
    w_mass * validation_df["mass_norm"] ** 2
    + w_defect * validation_df["defect_norm"] ** 2
    + w_bandgap * validation_df["bandgap_norm"] ** 2
)
# 为了和你之前输出习惯一致，保留 distance_to_ideal 这个名字
validation_df["distance_to_ideal"] = validation_df["compromise_score"]


# ---------- 3. 计算距离变量边界的距离 ----------
lb = pd.Series(lower_bounds, index=feature_names)
ub = pd.Series(upper_bounds, index=feature_names)

for name in feature_names:
    validation_df[f"{name}_boundary_distance"] = np.minimum(
        validation_df[name] - lb[name],
        ub[name] - validation_df[name]
    ) / (ub[name] - lb[name])

validation_df["min_boundary_distance"] = validation_df[
    [f"{name}_boundary_distance" for name in feature_names]
].min(axis=1)


# ---------- 4. 计算每个 Pareto 解到最近训练样本的距离 ----------
# 距离越大，说明该 Pareto 解离已有样本越远，代理模型越可能不可靠
X_train_all = X_bandgap[feature_names].values
X_pareto = validation_df[feature_names].values

nbrs = NearestNeighbors(n_neighbors=1).fit(X_train_all)
distances, indices = nbrs.kneighbors(X_pareto)

validation_df["nearest_train_distance"] = distances[:, 0]


# ---------- 5. 选取代表性方案 ----------
candidates = []

def add_candidate(scheme_name, idx):
    row = validation_df.loc[idx].copy()
    row["scheme"] = scheme_name
    candidates.append(row)


# 方案1：最小质量方案
idx_min_mass = validation_df["mass"].idxmin()
add_candidate("S1_minimum_mass", idx_min_mass)

# 方案2：最小 defect frequency 方案
idx_min_defect = validation_df["defect_frequency"].idxmin()
add_candidate("S2_minimum_defect_frequency", idx_min_defect)

# 方案3：最大 bandgap 方案
idx_max_bandgap = validation_df["bandgap"].idxmax()
add_candidate("S3_maximum_bandgap", idx_max_bandgap)

# 方案4：折中最优方案，距离理想点最近
idx_compromise = validation_df["compromise_score"].idxmin()
add_candidate("S4_best_compromise", idx_compromise)

# 方案5：Pareto 前沿中部方案，按 mass 排序取中间
mass_sorted_df = validation_df.sort_values(by="mass")
idx_middle = mass_sorted_df.index[len(mass_sorted_df) // 2]
add_candidate("S5_middle_pareto_solution", idx_middle)


# ---------- 6. 增加一个不贴边界的安全折中方案 ----------
# 因为你之前的折中解 steel_h_ratio 基本贴着 0.5 下边界
safe_df = validation_df[
    (validation_df["steel_r_ratio"] > 0.52) &
    (validation_df["steel_r_ratio"] < 0.78) &
    (validation_df["steel_h_ratio"] > 0.55) &
    (validation_df["steel_h_ratio"] < 0.95) &
    (validation_df["a_ratio"] > 0.52) &
    (validation_df["a_ratio"] < 0.95)
].copy()

if len(safe_df) > 0:
    idx_safe_compromise = safe_df["compromise_score"].idxmin()
    add_candidate("S6_safe_compromise_not_near_boundary", idx_safe_compromise)


# ---------- 7. 汇总代表性方案 ----------
selected_df = pd.DataFrame(candidates)

# 如果不同方案选到了同一个 Pareto 点，去重
selected_df = selected_df.drop_duplicates(
    subset=feature_names,
    keep="first"
)

# 如果去重后少于 5 个，再从折中得分较小的点里面补足
if len(selected_df) < 5:
    existing_points = (
        selected_df[feature_names]
        .round(8)
        .astype(str)
        .agg("_".join, axis=1)
        .tolist()
    )

    validation_df["point_key"] = (
        validation_df[feature_names]
        .round(8)
        .astype(str)
        .agg("_".join, axis=1)
    )

    supplement_df = validation_df[
        ~validation_df["point_key"].isin(existing_points)
    ].sort_values(by="compromise_score")

    need_num = 5 - len(selected_df)

    for i in range(min(need_num, len(supplement_df))):
        row = supplement_df.iloc[i].copy()
        row["scheme"] = f"S_extra_{i + 1}"
        selected_df = pd.concat(
            [selected_df, row.to_frame().T],
            ignore_index=True
        )


# ---------- 8. 整理输出列 ----------
output_columns = [
    "scheme",
    "steel_r_ratio",
    "steel_h_ratio",
    "a_ratio",
    "mass",
    "defect_frequency",
    "bandgap",
    "compromise_score",
    "distance_to_ideal",
    "min_boundary_distance",
    "nearest_train_distance"
]

selected_df = selected_df[output_columns]

# 保证数值列是数值类型
numeric_columns = [
    "steel_r_ratio",
    "steel_h_ratio",
    "a_ratio",
    "mass",
    "defect_frequency",
    "bandgap",
    "compromise_score",
    "distance_to_ideal",
    "min_boundary_distance",
    "nearest_train_distance"
]

selected_df[numeric_columns] = selected_df[numeric_columns].apply(
    pd.to_numeric,
    errors="coerce"
)
# ---------- 8.1 规定输出精度 ----------
# 设计变量保留 5 位小数，便于后续仿真建模
parameter_columns = [
    "steel_r_ratio",
    "steel_h_ratio",
    "a_ratio"
]

selected_df[parameter_columns] = selected_df[parameter_columns].round(5)

# 如果你希望目标值和辅助指标也统一保留 5 位小数，可以保留下面这段
other_numeric_columns = [
    "mass",
    "defect_frequency",
    "bandgap",
    "compromise_score",
    "distance_to_ideal",
    "min_boundary_distance",
    "nearest_train_distance"
]

selected_df[other_numeric_columns] = selected_df[other_numeric_columns].round(5)


# ---------- 9. 输出结果 ----------
print("建议用于仿真验证的代表性 Pareto 方案：")
print(selected_df)

selected_df.to_excel("用于仿真验证的代表性Pareto方案.xlsx", index=False)

print("已保存：用于仿真验证的代表性Pareto方案.xlsx")


# ---------- 10. 单独保存折中方案 ----------
best_compromise = selected_df[
    selected_df["scheme"] == "S4_best_compromise"
].copy()

best_compromise.to_excel("最终推荐折中最优方案.xlsx", index=False)

print("已保存：最终推荐折中最优方案.xlsx")
# =========================
# 15. 围绕 5 个代表性 Pareto 方案进行轻量局部加密采样
# 每个代表方案附近只生成 3 个新点
# =========================

base_schemes = [
    "S1_minimum_mass",
    "S2_minimum_defect_frequency",
    "S3_maximum_bandgap",
    "S4_best_compromise",
    "S5_middle_pareto_solution"
]

base_points_df = selected_df[selected_df["scheme"].isin(base_schemes)].copy()

lb = np.array(lower_bounds)
ub = np.array(upper_bounds)

# 局部扰动幅度
delta_r = 0.015
delta_h = 0.025
delta_a = 0.025

local_samples = []

for _, row in base_points_df.iterrows():

    center = row[feature_names].values.astype(float)
    scheme_name = row["scheme"]

    candidate_points = []

    # 点1：steel_r_ratio 正向扰动
    p1 = center.copy()
    p1[0] += delta_r
    candidate_points.append(("r_plus", p1))

    # 点2：steel_h_ratio 正向扰动
    # 尤其适合 h_ratio 贴近 0.5 下边界的情况
    p2 = center.copy()
    p2[1] += delta_h
    candidate_points.append(("h_plus", p2))

    # 点3：a_ratio 正向扰动
    p3 = center.copy()
    p3[2] += delta_a
    candidate_points.append(("a_plus", p3))

    for sample_type, point in candidate_points:
        point = np.clip(point, lb, ub)

        local_samples.append({
            "source_scheme": scheme_name,
            "sample_type": sample_type,
            "steel_r_ratio": point[0],
            "steel_h_ratio": point[1],
            "a_ratio": point[2]
        })

local_samples_df = pd.DataFrame(local_samples)

# 参数保留 5 位小数
local_samples_df[feature_names] = local_samples_df[feature_names].round(5)

# 去重
local_samples_df = local_samples_df.drop_duplicates(
    subset=feature_names,
    keep="first"
).reset_index(drop=True)

print("轻量局部加密样本：")
print(local_samples_df)

local_samples_df.to_excel(
    "五个代表性Pareto方案轻量局部加密样本.xlsx",
    index=False
)

print("已保存：五个代表性Pareto方案轻量局部加密样本.xlsx")