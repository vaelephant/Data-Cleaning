import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import glob
from flask import Flask, jsonify, render_template, request
from datetime import datetime

# 创建Flask应用
app = Flask(__name__)

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 定义函数以读取不同格式的数据文件
def read_data(file_path):
    file_extension = os.path.splitext(file_path)[1]
    try:
        if file_extension == '.txt':
            return pd.read_csv(file_path, sep='\t')
        elif file_extension == '.csv':
            return pd.read_csv(file_path)
        elif file_extension == '.json':
            return pd.read_json(file_path)
        elif file_extension in ['.xls', '.xlsx']:
            return pd.read_excel(file_path)
        elif file_extension == '.h5' or file_extension == '.hdf5':
            return pd.read_hdf(file_path)
        elif file_extension == '.parquet':
            return pd.read_parquet(file_path)
        elif file_extension == '.feather':
            return pd.read_feather(file_path)
        elif file_extension in ['.pkl', '.pickle']:
            return pd.read_pickle(file_path)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def detect_outliers(df):
    outliers = pd.DataFrame(columns=df.columns)
    for column in df.select_dtypes(include=['float64', 'int64']).columns:
        mean = df[column].mean()
        std = df[column].std()
        outliers[column] = ((df[column] - mean).abs() > 3 * std)
    return outliers.sum()

def generate_statistics(data, filename):
    # 检查和处理字典类型列
    for column in data.select_dtypes(include=['object']):
        if data[column].apply(type).eq(dict).any():
            data[column] = data[column].apply(lambda x: str(x))

    # 创建文件夹
    folder_name = os.path.splitext(filename)[0] + "_" + os.path.splitext(filename)[1][1:]
    folder_path = os.path.join('static', folder_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # 统计信息
    stats = {}
    stats['total_rows'] = len(data)
    stats['total_columns'] = data.shape[1]
    stats['dataset_name'] = filename
    stats['folder_path'] = folder_name
    stats['top_rows'] = data.head(5).to_dict(orient='records')

    # 缺失值分析
    missing_data = data.isnull().sum()
    missing_data_ratio = missing_data / len(data) * 100
    stats['missing_data_info'] = []
    for column, ratio in missing_data_ratio.items():
        if ratio == 0:
            info = f"{column}: 无缺失值，数据质量最好。"
        elif ratio < 5:
            info = f"{column}: 缺失值比例低，数据质量较好。可以使用插值法或均值填补。"
        elif ratio < 20:
            info = f"{column}: 缺失值比例中等，可能需要处理。建议使用插值法、均值填补或删除缺失值较多的行。"
        else:
            info = f"{column}: 缺失值比例高，需要处理。考虑删除该特征或使用高级填补方法（如KNN填补）。"
        stats['missing_data_info'].append(info)
        print(info)

    # 缺失值图表
    plt.figure(figsize=(10, 6))
    sns.barplot(x=missing_data.index, y=missing_data.values)
    plt.xticks(rotation=90)
    plt.title('缺失值数量')
    plt.xlabel('特征')
    plt.ylabel('缺失值数量')
    plt.savefig(os.path.join(folder_path, 'missing_values.png'))
    plt.close()

    # 异常值检测
    outliers = detect_outliers(data)
    outliers_ratio = outliers / len(data) * 100
    stats['outliers_info'] = []
    for column, ratio in outliers_ratio.items():
        if ratio == 0:
            info = f"{column}: 无异常值，数据质量最好。"
        elif ratio < 1:
            info = f"{column}: 异常值比例低，数据质量较好。可以选择保留或删除异常值。"
        elif ratio < 5:
            info = f"{column}: 异常值比例中等，可能需要处理。建议使用IQR法删除异常值或进行修正。"
        else:
            info = f"{column}: 异常值比例高，需要处理。考虑使用变换方法或删除异常值。"
        stats['outliers_info'].append(info)
        print(info)

    # 异常值图表
    for column in data.select_dtypes(include=['float64', 'int64']).columns:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=data[column])
        plt.title(f'{column}的箱线图')
        plt.savefig(os.path.join(folder_path, f'boxplot_{column}.png'))
        plt.close()

    # 数据分布图（直方图）
    for column in data.select_dtypes(include=['float64', 'int64']).columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(data[column], kde=True)
        plt.title(f'{column}的分布')
        plt.savefig(os.path.join(folder_path, f'distribution_{column}.png'))
        plt.close()

    # 重复值检测
    duplicate_rows = data.duplicated().sum()
    duplicate_ratio = duplicate_rows / len(data) * 100
    if duplicate_ratio == 0:
        stats['duplicate_info'] = "无重复值，数据质量最好。"
    elif duplicate_ratio < 1:
        stats['duplicate_info'] = "重复值比例低，数据质量较好。可以选择保留或删除重复值。"
    else:
        stats['duplicate_info'] = "重复值比例较高，需要处理。建议删除重复值。"
    stats['duplicate_rows'] = duplicate_rows
    stats['duplicate_ratio'] = duplicate_ratio
    print(f"重复值数量: {duplicate_rows}")
    print(f"重复值比例 (%): {duplicate_ratio}\n{stats['duplicate_info']}")

    # 数据类型检查
    data_types = data.dtypes
    stats['data_types_info'] = []
    for column, dtype in data_types.items():
        if dtype == 'object':
            info = f"{column}: 可能是分类变量或文本。"
        elif dtype in ['int64', 'float64']:
            info = f"{column}: 数值变量。"
        else:
            info = f"{column}: 数据类型为{dtype}，需要进一步检查。"
        stats['data_types_info'].append(info)
        print(info)

    # 唯一值数量
    unique_counts = data.nunique()
    stats['unique_counts_info'] = [f"{column}: 唯一值数量为{count}。" for column, count in unique_counts.items()]
    for info in stats['unique_counts_info']:
        print(info)

    # 唯一值数量图表
    plt.figure(figsize=(10, 6))
    sns.barplot(x=unique_counts.index, y=unique_counts.values)
    plt.xticks(rotation=90)
    plt.title('唯一值数量')
    plt.xlabel('特征')
    plt.ylabel('唯一值数量')
    plt.savefig(os.path.join(folder_path, 'unique_values.png'))
    plt.close()

    # 相关性分析
    numeric_data = data.select_dtypes(include=['float64', 'int64'])
    if not numeric_data.empty:
        correlation_matrix = numeric_data.corr()
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
        plt.title('特征相关性矩阵')
        plt.savefig(os.path.join(folder_path, 'correlation_matrix.png'))
        plt.close()

    return stats

# 从dataset文件夹中读取数据文件
data_folder_path = 'dataset'
data_files = glob.glob(os.path.join(data_folder_path, '*'))

datasets_statistics = {}
for file_path in data_files:
    filename = os.path.basename(file_path)
    data = read_data(file_path)
    if data is not None:
        datasets_statistics[filename] = generate_statistics(data, filename)

@app.route('/data')
def get_data():
    dataset_name = request.args.get('dataset', list(datasets_statistics.keys())[0])
    return jsonify(datasets_statistics[dataset_name])

@app.route('/')
def index():
    dataset_names = list(datasets_statistics.keys())
    current_dataset = request.args.get('dataset', dataset_names[0])
    stats = datasets_statistics[current_dataset]
    return render_template('index.html', dataset_names=dataset_names, stats=stats, current_dataset=current_dataset)

if __name__ == '__main__':
    app.run(debug=True)
