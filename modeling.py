import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, RocCurveDisplay
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

#读取数据
df = pd.read_parquet(Path('stage1_integrated.parquet'))

# 特征/目标
y = df['target_48h']
X = df.drop(columns=['target_48h', 'lead_id', 'contract_signed_time', 'first_touch_time'])
X = pd.get_dummies(X, drop_first=True)          

#训练/验证划分
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y)

#建模
clf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)
clf.fit(X_train, y_train)

#评估
y_pred_proba = clf.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_pred_proba)
print(f'ROC-AUC = {auc:.4f}')

#可视化
fig, ax = plt.subplots(1, 3, figsize=(15, 4.5), dpi=110)

#ROC
RocCurveDisplay.from_estimator(clf, X_test, y_test, ax=ax[0])
ax[0].set_title('ROC Curve')

#特征重要性
importances = pd.Series(clf.feature_importances_, index=X.columns)
importances.nlargest(15).plot(kind='barh', ax=ax[1])
ax[1].set_title('Top-15 Feature Importance')

#按分数分桶转化率
df_test = X_test.copy()
df_test['score'] = y_pred_proba
df_test['target'] = y_test
df_test['bucket'] = pd.qcut(df_test['score'], 5, labels=False)
conv = df_test.groupby('bucket')['target'].agg(['mean', 'count'])
conv['mean'].plot(kind='bar', ax=ax[2])
ax[2].set_title('Conversion Rate by Score Bucket')

plt.tight_layout()
plt.savefig('stage3_report.png')
plt.show()

#保存模型
import joblib
joblib.dump(clf, 'g48_model.joblib')
print('模型已保存到 g48_model.joblib')

m = joblib.load('g48_model.joblib')
print("模型需要输入", len(m.feature_names_in_), "个特征")
print("顺序：", list(m.feature_names_in_))
