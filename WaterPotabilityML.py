# 导入pandas库，用于数据处理
import pandas as pd
# 导入numpy库，用于数值计算
import numpy as np
# 导入warnings库，用于忽略警告信息
import warnings
warnings.filterwarnings("ignore")  # 忽略所有警告信息，避免输出干扰

# 导入matplotlib库，用于数据可视化
import matplotlib
import matplotlib.pyplot as plt

# 自动适配中文字体（macOS推荐使用“AppleGothic”）
matplotlib.rcParams['font.sans-serif'] = ['AppleGothic', 'Arial Unicode MS', 'STHeiti', 'SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# 导入sklearn中的预处理和模型选择工具
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OrdinalEncoder  # 数据预处理相关
from sklearn.model_selection import train_test_split, GridSearchCV  # 数据集划分和网格搜索

# 导入sklearn中的各种机器学习模型
from sklearn.linear_model import LogisticRegression  # 逻辑回归
from sklearn.neighbors import KNeighborsClassifier   # K近邻
from sklearn.tree import DecisionTreeClassifier      # 决策树
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier  # 集成模型

# 导入sklearn中的评估指标
from sklearn.metrics import accuracy_score, make_scorer  # 准确率和自定义评分函数

# 读取数据集，文件名为'1.csv'
df = pd.read_csv('1.csv')  # 读取csv文件，返回DataFrame对象

# 创建序数编码器，将Potability列中的类别（如是/否）转换为数字（0/1）
m = OrdinalEncoder()  # 创建OrdinalEncoder对象
# 对Potability列进行编码，并转换为int类型
df['Potability'] = m.fit_transform(df['Potability'].values.reshape(-1, 1)).astype('int')

# 删除包含缺失值的行，axis=0表示按行删除，how='any'表示只要有缺失就删
df = df.dropna(axis=0, how='any')

# 将特征和标签分开，x为特征，y为标签（目标变量）
y = df['Potability']  # 取出Potability列作为标签
df_features = df.drop('Potability', axis=1)  # 删除Potability列，剩下的作为特征
x = df_features  # 赋值给x，便于后续使用

# 划分训练集和测试集，比例为7:3，stratify=y保证分布一致
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1, stratify=y)

# 定义要使用的模型及其参数
models = [
    ("LR", LogisticRegression(max_iter=1000, C=1)),  # 逻辑回归
    ("KNN", KNeighborsClassifier(n_neighbors=10)),   # K近邻
    ("DTC", DecisionTreeClassifier(max_depth=3, criterion='gini')),  # 决策树
    ("RF", RandomForestClassifier(n_estimators=60, max_depth=9, random_state=1, class_weight='balanced')),  # 随机森林
    ("ADA", AdaBoostClassifier(learning_rate=0.01, n_estimators=100)),  # AdaBoost
    ("GBDT", GradientBoostingClassifier(learning_rate=0.01, n_estimators=100))  # 梯度提升树
]
accuracies = []  # 用于存储每个模型的准确率
names = []       # 用于存储每个模型的名称

# 遍历每个模型，进行训练和评估
for name, model in models:
    model.fit(x_train, y_train)  # 用训练集训练模型
    result = model.predict(x_test)  # 用测试集进行预测
    accuracy = accuracy_score(y_test, result)  # 计算预测的准确率
    print(f'{name}模型的准确率为{accuracy}')  # 输出模型名称和准确率
    accuracies.append(accuracy)  # 保存准确率
    names.append(name)           # 保存模型名称

# 用最后一个模型对全量数据进行预测，并保存到Excel文件
# 注意：这里只保存了最后一个模型的预测结果，如需保存所有模型结果可自行扩展
df['预测结果'] = model.predict(x)  # 用最后一个模型对所有数据进行预测
# 保存结果到Excel文件，文件名为'预测结果.xlsx'
df.to_excel('预测结果.xlsx')

# 绘制各模型准确率的折线图
plt.plot(names, accuracies)  # 横坐标为模型名称，纵坐标为准确率
plt.title('各模型准确率')    # 设置图表标题
plt.savefig('a.jpg', dpi=300)  # 保存图片为a.jpg，分辨率为300dpi
plt.show()  # 显示图表

# 定义Logistic回归的参数网格，用于后续网格搜索调参
parameters = {
    'C': np.arange(0.01, 0.1, 0.01),         # 正则化参数C的取值范围
    'max_iter': np.arange(10, 500, 10)       # 最大迭代次数的取值范围
}
scoring_fnc = make_scorer(accuracy_score)    # 以准确率作为评分标准

# 创建网格搜索对象，使用5折交叉验证
grid = GridSearchCV(LogisticRegression(), parameters, scoring=scoring_fnc, cv=5)
# 在训练集上进行网格搜索，寻找最优参数
grid = grid.fit(x_train, y_train)
reg = grid.best_estimator_  # 获取最优模型
print(reg.get_params())      # 输出最优模型的参数
y_pre = reg.predict(x_test)  # 用最优模型对测试集进行预测
print('优化后模型在测试集预测的准确率', accuracy_score(y_test, y_pre))  # 输出优化后模型的准确率 