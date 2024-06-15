import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 生成示例数据
np.random.seed(42)
n_samples = 1000
data = pd.DataFrame({
    'feature_1': np.random.normal(loc=0, scale=1, size=n_samples),
    'feature_2': np.random.normal(loc=5, scale=2, size=n_samples),
    'feature_3': np.random.randint(0, 100, size=n_samples),
    'feature_4': np.random.choice(['A', 'B', 'C'], size=n_samples),
    'feature_5': np.random.choice([np.nan, 1, 2, 3], size=n_samples, p=[0.1, 0.3, 0.3, 0.3])
})
data.loc[::10, 'feature_1'] = np.random.uniform(10, 15, size=n_samples // 10)
data = pd.concat([data, data.sample(50)])
data = data.drop_duplicates()

# 缺失值比例
missing_data = data.isnull().sum()
missing_data_ratio = missing_data / len(data) * 100
print(f"缺失值:\n{missing_data}\n")
print(f"缺失值比例 (%):\n{missing_data_ratio}\n")
for column, ratio in missing_data_ratio.items():
    if ratio == 0:
        print(f"{column}: 无缺失值，数据质量最好。")
    elif ratio < 5:
        print(f"{column}: 缺失值比例低，数据质量较好。")
    elif ratio < 20:
        print(f"{column}: 缺失值比例中等，可能需要处理。")
    else:
        print(f"{column}: 缺失值比例高，需要处理。\n")

# 异常值检测（以3倍标准差为准）
def detect_outliers(df):
    outliers = pd.DataFrame(columns=df.columns)
    for column in df.select_dtypes(include=['float64', 'int64']).columns:
        mean = df[column].mean()
        std = df[column].std()
        outliers[column] = ((df[column] - mean).abs() > 3 * std)
    return outliers.sum()

outliers = detect_outliers(data)
outliers_ratio = outliers / len(data) * 100
print(f"异常值:\n{outliers}\n")
print(f"异常值比例 (%):\n{outliers_ratio}\n")
for column, ratio in outliers_ratio.items():
    if ratio == 0:
        print(f"{column}: 无异常值，数据质量最好。")
    elif ratio < 1:
        print(f"{column}: 异常值比例低，数据质量较好。")
    elif ratio < 5:
        print(f"{column}: 异常值比例中等，可能需要处理。")
    else:
        print(f"{column}: 异常值比例高，需要处理。\n")

# 数据分布（直方图）
for column in data.select_dtypes(include=['float64', 'int64']).columns:
    plt.figure(figsize=(10, 6))
    sns.histplot(data[column], kde=True)
    plt.title(f'{column}的分布')
    plt.show()

# 重复值检测
duplicate_rows = data.duplicated().sum()
duplicate_ratio = duplicate_rows / len(data) * 100
print(f"重复值数量: {duplicate_rows}")
print(f"重复值比例 (%): {duplicate_ratio}\n")
if duplicate_ratio == 0:
    print("无重复值，数据质量最好。")
elif duplicate_ratio < 1:
    print("重复值比例低，数据质量较好。")
else:
    print("重复值比例较高，需要处理。\n")

# 数据类型检查
data_types = data.dtypes
print(f"数据类型:\n{data_types}\n")
print("数据类型一致性检查：")
for column, dtype in data_types.items():
    if dtype == 'object':
        print(f"{column}: 可能是分类变量或文本。")
    elif dtype in ['int64', 'float64']:
        print(f"{column}: 数值变量。")
    else:
        print(f"{column}: 数据类型为{dtype}，需要进一步检查。\n")

# 唯一值数量
unique_counts = data.nunique()
print(f"唯一值数量:\n{unique_counts}\n")
print("唯一值数量检查：")
for column, count in unique_counts.items():
    print(f"{column}: 唯一值数量为{count}。")

# 相关性分析
numeric_data = data.select_dtypes(include=['float64', 'int64'])
correlation_matrix = numeric_data.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('特征相关性矩阵')
plt.show()
