# 饮用水可饮用性机器学习项目

本项目使用Python和scikit-learn实现了饮用水可饮用性（Potability）预测，涵盖数据预处理、模型训练、评估、调参等完整流程，并为初学者添加了详细注释。

## 功能特点
- 数据预处理（缺失值删除、标签编码）
- 多种分类模型：逻辑回归、KNN、决策树、随机森林、AdaBoost、梯度提升树
- 模型准确率对比与可视化
- 逻辑回归网格搜索调参
- 预测结果保存为Excel，准确率图像保存为图片

## 文件结构
- `WaterPotabilityML.py`：主程序脚本（含详细注释）
- `1.csv`：数据集文件（需与脚本放在同一目录）
- `预测结果.xlsx`：模型预测结果输出文件
- `a.jpg`：模型准确率折线图

## 环境依赖
- Python 3.x
- pandas
- numpy
- matplotlib
- scikit-learn

安装依赖包命令：
```bash
pip install pandas numpy matplotlib scikit-learn
```

## 运行方法
1. 确保`1.csv`数据文件与脚本在同一目录。
2. 运行主程序：
   ```bash
   python WaterPotabilityML.py
   ```
3. 查看输出文件：`预测结果.xlsx` 和 `a.jpg`

## 注意事项
- 脚本注释详细，适合初学者学习参考。
- macOS下如出现字体警告，脚本已自动适配常见中文字体。
- 数据集需包含名为`Potability`的目标列。

## 作者
- 由AI助手整理优化，适合初学者学习。 