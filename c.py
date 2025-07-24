import pandas as pd,os,numpy as np
from pathlib import Path

#导入数据（parse_dates作用是把日期类型转为datetime64类型，方便后续做时间运算）
leads = pd.read_csv("leads_raw.csv",parse_dates=['first_touch_time'])
events =pd.read_csv('events.csv',parse_dates=["event_time"])
emails =pd.read_csv('email_engagement.csv',parse_dates=['email_sent_time'])
calls =pd.read_csv('sales_activity.csv',parse_dates=['call_time'])
contracts  = pd.read_csv('contracts.csv', parse_dates=['contract_signed_time'])

#事件表---宽特征
event_pivot =(events
              .assign(dummy=1)
              .pivot_table(index='lead_id',
                           columns='event_type',
                           values='dummy',
                           aggfunc='sum',
                           fill_value=0)
              .add_prefix('event_')
              .reset_index())
#邮件互动---宽特征
email_agg =(emails
            .groupby('lead_id')
            .agg(
                 email_sent_cnt=('opened','size'),
                 email_open_cnt=('opened','sum'),
                 email_click_cnt=('clicked','sum')
                )
            .reset_index())
#销售电话---宽特征
call_agg =(calls
           .groupby('lead_id')
           .agg(
                called_cnt=('call_outcome','size'),
                called_duration_mean=('call_duration','mean'),
                called_interested=('call_outcome',lambda x:(x=='interested').sum())
               )
           .reset_index())

#合同
contracts =contracts.rename(
    columns={'contract_signed_time': 'contract_signed_time',
             'contract_value': 'contract_value'})

#按lead_id逐级合并成一张宽表
df=(leads
    .merge(event_pivot,on='lead_id',how='left')
    .merge(email_agg,on='lead_id',how='left')
    .merge(call_agg,on='lead_id',how='left')
    .merge(contracts,on='lead_id',how='left')
    )



df['target_48h']=((df['contract_signed_time']-df['first_touch_time'])<=pd.Timedelta(hours=48)).astype(int)



#缺失值处理
#保证缺失值统一，都为浮点类型
df['contract_value']=pd.to_numeric(df['contract_value'],errors='coerce')
#删除合同金额缺失（未签单）
df=df.dropna(subset=['contract_value'])
#连续值--median
num_cols=df.select_dtypes(include='number').columns
for c in num_cols:
    df[c]=df[c].fillna(df[c].median())
#类别型变量--Unknown
cat_cols=df.select_dtypes(exclude='number').columns
for c in cat_cols:
    if c!='lead_id':
        df[c]=df[c].fillna('Unknown')
#保存结果
df.to_parquet('stage1_integrated',index=False)
print('阶段一数据收集完成',len(df))

events30=events.merge(leads[['lead_id','first_touch_time']],on='lead_id',how='left')
mask30=(events30['event_time']-events30['first_touch_time'])<=pd.Timedelta(minutes=30)
events30=events30[mask30]

ev30 = (events30
        .groupby('lead_id')['event_type']
        .value_counts()
        .unstack(fill_value=0)
        .add_suffix('_30min'))

# 邮件 30 min 行为
email30 = emails.merge(leads[['lead_id', 'first_touch_time']], on='lead_id')
em_mask30 = (email30['email_sent_time'] - email30['first_touch_time']).dt.total_seconds().between(0, 30 * 60)
email30 = email30[em_mask30]
email30_feat = (email30
                .groupby('lead_id')
                .agg(email_sent_30min=('opened', 'size'),
                     email_opened_30min=('opened', 'sum'),
                     email_clicked_30min=('clicked', 'sum')))

# 合并 30 min 特征
df = df.merge(ev30,      left_on='lead_id', right_index=True, how='left')
df = df.merge(email30_feat, left_on='lead_id', right_index=True, how='left')
# 9. 将缺失 30 min 特征补 0
thirty_cols = [c for c in df.columns if c.endswith('_30min')]
df[thirty_cols] = df[thirty_cols].fillna(0)



#转数值，非法值变 NaN
df['form_length'] = pd.to_numeric(df['form_length'], errors='coerce')

#用分桶前，把 NaN 先填掉
df['form_length'] = df['form_length'].fillna(df['form_length'].median())

#再分桶
df['form_length_bucket'] = pd.cut(
    df['form_length'],
    bins=[0, 3, 5, 7, 10],
    labels=['short', 'mid', 'long', 'very_long']
)



#保存中间结果
out_path = Path('stage1_integrated.parquet')
df.to_parquet(out_path, index=False)
print('阶段一完成，行数:', len(df))
print('目标变量分布:\n', df['target_48h'].value_counts())
print('缺失值检查:\n', df.isna().sum().sort_values(ascending=False).head())


print(df.info())
print(df['target_48h'].value_counts())
print(df.filter(regex='_30min$').head())
