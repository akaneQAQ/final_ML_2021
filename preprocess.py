import re
import numpy as np
import pandas as pd

path = 'C:/Users/shq20/Desktop/dataset_finalproject'
df = pd.read_csv(path + '/vaers_jan_nov_2021.csv')

df_vaccine = df[['VAERS_ID', 'VAX_MANU', 'AGE_YRS', 'SEX', 'DISABLE', 'OTHER_MEDS', 'CUR_ILL',
                 'NUMDAYS', 'HOSPITAL', 'HOSPDAYS', 'DIED', 'L_THREAT', 'HISTORY']]
df_vaccine = df_vaccine.drop_duplicates()
df_vaccine = df_vaccine[df_vaccine['NUMDAYS'] <= 50]
df_vaccine.drop(index=df_vaccine[df_vaccine['VAX_MANU'] == 'UNKNOWN MANUFACTURER'].index, inplace=True)
df_vaccine.dropna(subset=["AGE_YRS", 'VAX_MANU', 'SEX'])
df_vaccine = df_vaccine[df_vaccine['AGE_YRS'] > 12]
df_vaccine.reset_index(drop=True, inplace=True)
df_history = df_vaccine['HISTORY']
## 数据预处理
df_vaccine.fillna(0, inplace=True)
df_vaccine.replace('Y', 1, inplace=True)
df_vaccine.replace('MODERNA', 0, inplace=True)
df_vaccine.replace('PFIZER\BIONTECH', 1, inplace=True)
df_vaccine.replace('JANSSEN', 2, inplace=True)
df_vaccine['SEX'].replace('M', 0, inplace=True)
df_vaccine['SEX'].replace('F', 1, inplace=True)
df_vaccine['SEX'].replace('U', 0.5, inplace=True)

moderna = []
pfizer = []
meds = []
ill = []
for index, row in df_vaccine.iterrows():
    if row['VAX_MANU'] == 0:
        moderna.append(1)
        pfizer.append(0)
    elif row['VAX_MANU'] == 1:
        moderna.append(0)
        pfizer.append(1)
    else:
        moderna.append(0)
        pfizer.append(0)
    if row['OTHER_MEDS'] != 0:
        meds.append(1)
    else:
        meds.append(0)
    if row['CUR_ILL'] == 0:
        ill.append(0)
    elif str(row['CUR_ILL']).lower().find('none') >= 0:
        ill.append(0)
    else:
        ill.append(1)

df_vaccine['MODERNA'] = moderna
df_vaccine['PFIZER\BIONTECH'] = pfizer
df_vaccine['OTHER_MEDS'] = meds
df_vaccine['CUR_ILL'] = ill

## 预处理HISTORY
history_ori = df_history.astype('str')
dict_history = {}
for index, line in history_ori.items():
    split = re.split(r'[;,\s,/]', line.lower())
    for item in split:
        if item in dict_history:
            dict_history[item] += 1
        else:
            dict_history[item] = 1
dict_history_sorted = sorted(dict_history.items(), key=lambda x:x[1], reverse=True)
# asthma/hypertension/diabetes/allergy/anxiety/depression/arthritis/thyroid/kidney
# hyperlipidemia(cholesterol)/heart disease/pain(algia)/cancer/obesity/migraines/covid/......
## 变成16feature的0-1矩阵
h1 = history_ori[history_ori.str.contains('asthma', case=False)].index  # 哮喘
h2 = history_ori[history_ori.str.contains('hypertension|blood pressure', case=False)].index  # 高血压
h3 = history_ori[history_ori.str.contains('diabete', case=False)].index  # 糖尿病
h4 = history_ori[history_ori.str.contains('allergy|allergies', case=False)].index  # 过敏
h5 = history_ori[history_ori.str.contains('anxiety', case=False)].index  # 焦虑症
h6 = history_ori[history_ori.str.contains('depression', case=False)].index  # 抑郁症
h7 = history_ori[history_ori.str.contains('thyroid', case=False)].index  # 甲状腺疾病
h8 = history_ori[history_ori.str.contains('hyperlipidemia|cholesterol', case=False)].index  # 高血脂
h9 = history_ori[history_ori.str.contains('heart', case=False)].index  # 心脏病
h10 = history_ori[history_ori.str.contains('pain|algia', case=False)].index  # 疼痛
h11 = history_ori[history_ori.str.contains('cancer', case=False)].index  # 癌症
h12 = history_ori[history_ori.str.contains('obesity', case=False)].index  # 肥胖
h13 = history_ori[history_ori.str.contains('migraines', case=False)].index  # 偏头痛
h14 = history_ori[history_ori.str.contains('covid', case=False)].index  # 新冠
h15 = history_ori[history_ori.str.contains('arthritis', case=False)].index  # 关节炎
h16 = history_ori[history_ori.str.contains('kidney', case=False)].index  # 肝脏疾病

df_histo = pd.DataFrame(np.zeros((len(history_ori), 16)),
                        columns=['asthma', 'hypertension', 'diabete', 'allergy', 'anxiety',
                                 'depression', 'thyroid', 'hyperlipidemia', 'heart', 'pain',
                                 'cancer', 'obesity', 'migraines', 'covid', 'arthritis', 'kidney'])

df_histo['asthma'][h1] = 1
df_histo['hypertension'][h2] = 1
df_histo['diabete'][h3] = 1
df_histo['allergy'][h4] = 1
df_histo['anxiety'][h5] = 1
df_histo['depression'][h6] = 1
df_histo['thyroid'][h7] = 1
df_histo['hyperlipidemia'][h8] = 1
df_histo['heart'][h9] = 1
df_histo['pain'][h10] = 1
df_histo['cancer'][h11] = 1
df_histo['obesity'][h12] = 1
df_histo['migraines'][h13] = 1
df_histo['covid'][h14] = 1
df_histo['arthritis'][h15] = 1
df_histo['kidney'][h16] = 1
dataf = pd.concat([df_vaccine, df_histo], axis=1)
infl = np.asarray(dataf[['MODERNA', 'PFIZER\BIONTECH', 'AGE_YRS', 'SEX', 'DISABLE',
                'OTHER_MEDS', 'CUR_ILL', 'asthma', 'hypertension', 'diabete', 'allergy',
                'anxiety', 'depression', 'thyroid', 'hyperlipidemia', 'heart', 'pain',
                'cancer', 'obesity', 'migraines', 'covid', 'arthritis', 'kidney']])
resp = np.asarray(dataf['NUMDAYS'], dtype=np.float)
m = np.linalg.inv(infl.T@infl)@infl.T@resp
xde = []
for i in range(len(infl)):
    if np.abs(np.sum(m * infl[i])-resp[i]) >= 50:
        xde.append(i)
dataf.drop(xde, inplace=True)
dataf.reset_index(drop=True, inplace=True)

outputpath = path + "/vaers_preprocessed_0.csv"
dataf.to_csv(outputpath)

