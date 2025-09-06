import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier  # 随机森林分类器
from sklearn.model_selection import train_test_split  # 数据集分割
from sklearn.metrics import classification_report, accuracy_score  # 模型评估指标
import matplotlib.pyplot as plt  # 数据可视化

data_q4 = pd.read_excel('女胎检测数据.xlsx')

# 新增怀孕时间（天）列
def week2day(week: str):
    try:
        w = 0
        d = 0
        if '+' in week:
            WAD = week.split("+")
            w = int(WAD[0].strip("w").strip())
            d = int(WAD[1].strip("d").strip())
        elif 'w' in week:
            w = int(week.strip('w').strip())
            d = 0
        elif 'd' in week:
            w = 0
            d = int(week.strip('d').strip())
        else:
            # 如果格式不符合预期，返回NaN或默认值
            return pd.NaT
        return 7 * w + d
    except (ValueError, AttributeError):
        # 处理转换错误
        return pd.NaT

data_q4["怀孕时间"] = data_q4["检测孕周"].apply(week2day)

new_columns = {
    'T13异常': np.nan,
    'T18异常': np.nan,
    'T21异常': np.nan,
    '二分类异常': np.nan,
    'T13T18异常': np.nan,
    'T13T21异常': np.nan,
    'T18T21异常': np.nan
}

# 使用 assign 方法批量添加空列
data_q4 = data_q4.assign(**new_columns)

# 先把空值填成空字符串，方便后面统一处理
data_q4['染色体的非整倍体'] = data_q4['染色体的非整倍体'].fillna('')

# 构造三个新列，默认 0
for col in ['T13异常', 'T18异常', 'T21异常']:
    data_q4[col] = 0

# 根据 AB 列内容打 1
data_q4.loc[data_q4['染色体的非整倍体'].str.contains('T13', na=False), 'T13异常'] = 1
data_q4.loc[data_q4['染色体的非整倍体'].str.contains('T18', na=False), 'T18异常'] = 1
data_q4.loc[data_q4['染色体的非整倍体'].str.contains('T21', na=False), 'T21异常'] = 1

# 更新“二分类异常”列
data_q4['二分类异常'] = np.where((data_q4['T13异常'] == 1) | (data_q4['T18异常'] == 1) | (data_q4['T21异常'] == 1), 1,
                                 0)

# 更新“T13T18异常”列
data_q4['T13T18异常'] = np.where((data_q4['T13异常'] == 1) & (data_q4['T18异常'] == 1), 1, 0)

# 更新“T13T21异常”列
data_q4['T13T21异常'] = np.where((data_q4['T13异常'] == 1) & (data_q4['T21异常'] == 1), 1, 0)

# 更新“T18T21异常”列
data_q4['T18T21异常'] = np.where((data_q4['T18异常'] == 1) & (data_q4['T21异常'] == 1), 1, 0)

# 定义特征列名列表（用于模型输入的变量）
features = [
    '检测抽血次数', '孕妇BMI', '在参考基因组上比对的比例', '重复读段的比例', '年龄',
    '13号染色体的Z值', '18号染色体的Z值', '21号染色体的Z值', 'X染色体的Z值', 'X染色体浓度',
    '13号染色体的GC含量', '18号染色体的GC含量', '21号染色体的GC含量', '怀孕时间']
# 删除任何特征列或目标列中包含缺失值的行
data_q4 = data_q4.dropna(subset=features)

# 定义目标变量列名
target1 = '二分类异常'
target2 = 'T13异常'
target3 = 'T18异常'
target4 = 'T21异常'
target5 = 'T13T18异常'
target6 = 'T13T21异常'
target7 = 'T18T21异常'
targets = [target1, target2, target3, target4, target5, target6, target7]
# 提取特征矩阵X（所有特征列）
X = data_q4[features]

# 创建一个包含多个模型的列表,初始化随机森林分类器
models = []
for i in range(7):
    # 为每个模型使用相同的参数
    rf = RandomForestClassifier(
        n_estimators=100,  # 森林中树的数量
        max_features='sqrt',  # 每棵树考虑的特征数
        bootstrap=True,  # 使用bootstrap采样构建每棵树
        oob_score=True,  # 计算袋外分数
        random_state=42,  # 固定随机状态
        class_weight='balanced'  # 自动平衡类别权重
    )
    models.append(rf)
rf1 = models[0]
rf2 = models[1]
rf3 = models[2]
rf4 = models[3]
rf5 = models[4]
rf6 = models[5]
rf7 = models[6]

# 创建相关列表存放训练数据
X_train_list = []
X_test_list = []
y_train_list = []
y_test_list = []
y_pred_list = []

# 提取目标向量，制成列表，备用
y_list = []
for i in range(7):
    y_list.append(data_q4[targets[i]])

def train_model(rf, target):
    # 提取目标向量y（目标列）
    y = data_q4[target]

    # 将数据划分为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.3,  # 测试集占30%
        random_state=42,  # 随机种子确保结果可复现
        stratify=y  # 保持训练集和测试集中目标变量的类别比例相同
    )

    # 在训练集上训练模型
    rf.fit(X_train, y_train)

    # 在测试集上进行预测
    y_pred = rf.predict(X_test)

    X_train_list.append(X_train)
    X_test_list.append(X_test)
    y_train_list.append(y_train)
    y_test_list.append(y_test)
    y_pred_list.append(y_pred)


def plot_feature_importances(rf, target, index):
    # 打印分类报告（包含精确度、召回率、F1分数等指标）
    print("\n模型评估报告:")
    print(classification_report(y_test_list[index], y_pred_list[index], zero_division=np.nan))

    # 计算并打印准确率
    print("准确率:", accuracy_score(y_test_list[index], y_pred_list[index]))

    # 打印袋外分数（OOB Score）- 随机森林特有的评估指标
    print("袋外分数:", rf.oob_score_)

    # 创建特征重要性DataFrame
    feature_importances = pd.DataFrame({
        'Feature': features,  # 特征名称
        'Importance': rf.feature_importances_  # 特征重要性得分
    }).sort_values(by='Importance')  # 按重要性排序

    plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
    # 创建水平条形图展示特征重要性
    plt.figure(figsize=(18, 6))  # 设置图形大小
    plt.barh(
        feature_importances['Feature'],  # y轴：特征名称
        feature_importances['Importance'],  # x轴：重要性得分
        color='#1f77b4'  # 条形颜色
    )
    plt.xlabel('重要性')  # x轴标签
    plt.title(target + '特征重要性排序')  # 图表标题
    plt.tight_layout()  # 自动调整子图参数
    plt.show()  # 显示图形

# 训练“二分类异常”随机森林模型
train_model(rf1, target1)
# 二分类异常判定
plot_feature_importances(rf1, target1, 0)

# 训练“T13异常”随机森林模型
train_model(rf2, target2)
# T13异常判定
plot_feature_importances(rf2, target2, 1)

# 训练“T18异常”随机森林模型
train_model(rf3, target3)
# T18异常判定
plot_feature_importances(rf3, target3, 2)

# 训练“T21异常”随机森林模型
train_model(rf4, target4)
# T21异常判定
plot_feature_importances(rf4, target4, 3)

# 训练“T13T18异常”随机森林模型
train_model(rf5, target5)
# T13T18异常判定
plot_feature_importances(rf5, target5, 4)

# 训练“T13T21异常”随机森林模型
train_model(rf6, target6)
# T13T21异常判定
plot_feature_importances(rf6, target6, 5)

# 训练“T18T21异常”随机森林模型
train_model(rf7, target7)
# T18T21异常判定
plot_feature_importances(rf7, target7, 6)

# 将'染色体非整倍体'指标转化为列表，便于整体的分析
chromosome_list = data_q4['染色体的非整倍体'].tolist()

# 整合七个随机森林，创建分层决策体系
def hierarchicalmodel_accuracy():
    correct = 0
    for i in range(len(y_list[0])):
        sample = X.iloc[i:i + 1]
        true_label = chromosome_list[i]
        pred_label = None

        # === 第一层决策：组合异常检测 ===
        if rf5.predict(sample)[0] == 1:
            pred_label = 'T13T18'
        elif rf6.predict(sample)[0] == 1:
            pred_label = 'T13T21'
        elif rf7.predict(sample)[0] == 1:
            pred_label = 'T18T21'

        # === 第二层决策：单染色体检测 ===
        if pred_label is None:  # 确保上层未决策才进入本层
            if rf2.predict(sample)[0] == 1:
                pred_label = 'T13'
            elif rf3.predict(sample)[0] == 1:
                pred_label = 'T18'
            elif rf4.predict(sample)[0] == 1:
                pred_label = 'T21'

        # === 第三层决策：二分类检测 ===
        if pred_label is None:  # 确保上层未决策才进入本层
            pred_label = '可能存在其他异常' if rf1.predict(sample)[0] == 1 else ''

        # 验证
        if pred_label == true_label:
            correct += 1

    return correct / len(y_list[0])


# 计算准确率
accuracy = hierarchicalmodel_accuracy()
print(f"\n\n严格分层准确率: {accuracy:.2%}")