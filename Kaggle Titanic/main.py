import numpy as np 
import pandas as pd 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

import os
#数据加载
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, 'titanic')

for dirname, _, filenames in os.walk(data_dir):
    for filename in filenames:
        print(os.path.join(dirname, filename))

train_data = pd.read_csv(os.path.join(data_dir, 'train.csv'))

test_data = pd.read_csv(os.path.join(data_dir, 'test.csv'))

y = train_data["Survived"]

#数据预处理

#补充缺失年龄
#提取头衔
train_data['Title'] = train_data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
test_data['Title'] = test_data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

#按照头衔分组，用每组的中位数填充年龄
train_data['Age'] = train_data.groupby('Title')['Age'].transform(lambda x: x.fillna(x.median()))
test_data['Age'] = test_data.groupby('Title')['Age'].transform(lambda x: x.fillna(x.median()))

#其他缺失的年龄用整体中位数填
train_data['Age'].fillna(train_data['Age'].median(), inplace=True)
test_data['Age'].fillna(train_data['Age'].median(), inplace=True)

#补充缺失的上船点
train_data['Embarked'].fillna('S', inplace=True)
test_data['Embarked'].fillna('S', inplace=True)

#补充缺失的票价
test_data['Fare'].fillna(train_data['Fare'].median(), inplace=True)
train_data['Fare'].fillna(train_data['Fare'].median(), inplace=True)

#计算家庭规模还有是不是一个人
for dataset in [train_data, test_data]:
    dataset['RelativeCount'] = dataset['SibSp'] + dataset['Parch']
    dataset['IsAlone'] = 0
    dataset.loc[dataset['RelativeCount'] == 0, 'IsAlone'] = 1

#剔除其他无关特征
train_data = train_data.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])
test_data = test_data.drop(columns=['Name', 'Ticket', 'Cabin'])

#准备最终用于模型训练的数据特征
features = ["Pclass", "Sex", "SibSp", "Parch", "Age", "Embarked", "Fare", "RelativeCount", "IsAlone"]
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

#不同模型评估
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

#随机森林评估
model_forest = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model_forest.fit(X_train, y_train)
pred_val_forest = model_forest.predict(X_val)
print("随机森林模型")
print(f"准确率: {accuracy_score(y_val, pred_val_forest):.4f}")
print(classification_report(y_val, pred_val_forest))

#逻辑回归评估
from sklearn.linear_model import LogisticRegression
model_logistic = LogisticRegression(max_iter=1000, random_state=1)
model_logistic.fit(X_train, y_train)
pred_val_logistic = model_logistic.predict(X_val)
print("逻辑回归模型")
print(f"准确率: {accuracy_score(y_val, pred_val_logistic):.4f}")
print(classification_report(y_val, pred_val_logistic))

#K近邻评估
from sklearn.neighbors import KNeighborsClassifier
model_knn = KNeighborsClassifier(n_neighbors=5)
model_knn.fit(X_train, y_train)
pred_val_knn = model_knn.predict(X_val)
print("K近邻模型")
print(f"准确率: {accuracy_score(y_val, pred_val_knn):.4f}")
print(classification_report(y_val, pred_val_knn))

#支持向量机评估
from sklearn.svm import SVC
model_svm = SVC(kernel='linear', random_state=1)
model_svm.fit(X_train, y_train)
pred_val_svm = model_svm.predict(X_val)
print("支持向量机模型")
print(f"准确率: {accuracy_score(y_val, pred_val_svm):.4f}")
print(classification_report(y_val, pred_val_svm))


#选出最佳模型
models_dict = {
    "Random Forest": (model_forest, accuracy_score(y_val, pred_val_forest)),
    "Logistic Regression": (model_logistic, accuracy_score(y_val, pred_val_logistic)),
    "KNN": (model_knn, accuracy_score(y_val, pred_val_knn)),
    "SVM": (model_svm, accuracy_score(y_val, pred_val_svm))
}

best_model_name = max(models_dict, key=lambda k: models_dict[k][1])
best_model = models_dict[best_model_name][0]
best_accuracy = models_dict[best_model_name][1]

print(f"The best model is {best_model_name} with Accuracy = {best_accuracy:.4f}")

# ==================== 超参数调优 (GridSearchCV) ====================
from sklearn.model_selection import GridSearchCV

if best_model_name == "Random Forest":
    print("开始对 Random Forest 进行超参数调优 (GridSearchCV)...")
    # 1. 定义想要测试的参数字典 (网格)
    param_grid = {
        'n_estimators': [50, 100, 200],      # 测试3种树的数量
        'max_depth': [3, 5, 7, 10],          # 测试4种最大深度
        'min_samples_split': [2, 5, 10]      # 测试分裂内部节点所需的最小样本数
    }
    
    # 2. 初始化 GridSearchCV
    # cv=5 表示5折交叉验证, n_jobs=-1 表示使用所有CPU核心加速
    grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=1), 
                               param_grid=param_grid, 
                               cv=5, 
                               scoring='accuracy', 
                               n_jobs=-1)
    
    # 3. 在全量数据上进行网格搜索
    grid_search.fit(X, y)
    
    print(f"调优完成！最佳参数组合: {grid_search.best_params_}")
    print(f"调优后的最高交叉验证得分: {grid_search.best_score_:.4f}")
    
    # 获取调过优的最强模型
    best_model = grid_search.best_estimator_
else:
    print("Training the best model with 100% of the data...")
    best_model.fit(X, y)

#重新训练 / 预测 (最佳模型)
predictions_best = best_model.predict(X_test)

#保存结果
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions_best})
output.to_csv(os.path.join(current_dir, 'submission.csv'), index=False)
print(f"Best model ({best_model_name}) submission was successfully saved as submission.csv!")

