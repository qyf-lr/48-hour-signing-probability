"""
阶段 4：单文件脚本，输入 30 min 线索数据，返回 G48 分数
用法示例：
python stage4_score.py lead_12345 2 1 0 0 1 0 0 0 0 0 0 0
"""
import sys, joblib, pandas as pd
MODEL = joblib.load('g48_model.joblib')
COLS = MODEL.feature_names_in_            # 训练时的列顺序
def score_lead(lead_id, *vals):
    """vals 与 COLS 对齐的 30 min 特征值，缺失补 0"""
    row = pd.DataFrame([vals], columns=COLS, dtype=float).fillna(0)
    p = MODEL.predict_proba(row)[0,1]
    print(f'{lead_id} G48={p:.3f}')
if __name__ == '__main__':
    if len(sys.argv) < 2: sys.exit('Usage: python stage4_score.py lead_id <feature1> ...')
    score_lead(*sys.argv[1:])

