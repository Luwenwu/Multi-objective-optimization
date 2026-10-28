import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor

# 1. 读取数据
df = pd.read_excel('数据集mass.xlsx')

# 2. 指定目标列
target_name = 'mass'

# 3. 构造输入和输出
X = df.drop(columns=[target_name])
y = df[target_name].values

# 4. 保存特征名顺序（后面 NSGA-II 要用）
feature_names = X.columns.tolist()

# 5. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1
)

# 6. 训练 mass 代理模型
GBRT_mass = GradientBoostingRegressor(
    random_state=123,
    n_estimators=310,
    max_depth=2,
    learning_rate=0.22120855450771604,
    min_samples_split=4
)

GBRT_mass.fit(X_train, y_train)

# 7. 简单测试一下能不能预测
test_input = pd.DataFrame([[0.8, 0.7, 0.9]], columns=feature_names)
pred_mass = GBRT_mass.predict(test_input)[0]

print("feature_names =", feature_names)
print("预测的 mass =", pred_mass)