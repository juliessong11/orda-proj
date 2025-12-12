# train_model.py
import pandas as pd
import numpy as np
import re
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier
import joblib

# ---------------------------
# LOAD DATA
# ---------------------------
df = pd.read_csv("happy_data.csv")
df.replace([
    ".i:  Inapplicable", ".n:  No answer", ".s:  Skipped on Web",
    ".y:  Not available in this year", ".d:  Do not Know/Cannot Choose",
    ".r:  Refused"
], np.nan, inplace=True)

df_clean = df.copy()

# Drop columns with >60% NaNs
for col in df.columns:
    if df[col].isna().mean() > 0.6:
        df_clean.drop(columns=[col], inplace=True)

df_clean.drop(columns=['year', 'id_', 'divorce'], inplace=True, errors='ignore')
df_clean.dropna(subset=['happy'], inplace=True)

# Encode education
edu_cat_list = []
for val in df_clean['educ']:
    if pd.isna(val):
        edu_cat_list.append(np.nan)
    elif 'grade' in str(val).lower():
        num = re.search(r'\d+', str(val))
        if num:
            grade = int(num.group())
            if grade <= 6: edu_cat_list.append('elementary')
            elif grade <= 8: edu_cat_list.append('middle')
            elif grade <= 11: edu_cat_list.append('some_high_school')
            elif grade == 12: edu_cat_list.append('high_school')
            else: edu_cat_list.append(np.nan)
        else:
            edu_cat_list.append(np.nan)
    elif 'college' in str(val).lower():
        num = re.search(r'\d+', str(val))
        if num:
            yrs = int(num.group())
            if yrs < 4: edu_cat_list.append('some_college')
            elif 4 <= yrs <= 5: edu_cat_list.append('college')
            else: edu_cat_list.append('6+_college')
        else:
            edu_cat_list.append('college')
    else:
        edu_cat_list.append(np.nan)
df_clean['educ_cat'] = edu_cat_list

# Binary features
df_clean["is_religious"] = (df_clean["relig"].notna() & (df_clean["relig"] != "None")).astype(int)
binary_cols = [
    'wrkstat','marital','race','reg16','family16','relig','reliten','hapmar',
    'health','satjob','satfin','finalter','gender1','educ_cat','life'
]
df_clean = pd.get_dummies(df_clean, columns=binary_cols, drop_first=True, dummy_na=True)

# Happiness mapping
def map_happy(v):
    return {'Not too happy':1, 'Pretty happy':5, 'Very happy':10}.get(v, np.nan)
y = df_clean['happy'].apply(map_happy)

# Numeric conversion
df_clean['childs'] = df_clean['childs'].replace({'8 or more':8, '.d:  Do not Know/Cannot Choose':np.nan}).astype(float)
df_clean['age'] = df_clean['age'].replace({'89 or older':89}).astype(float)

# Income midpoints
def income_midpoint_simple(val):
    if pd.isna(val): return np.nan
    nums = [float(n) for n in re.findall(r'\d+', str(val).replace(',',''))]
    if len(nums) == 2: return np.mean(nums)
    elif len(nums) == 1: return nums[0]
    else: return np.nan
df_clean['income_mid'] = df_clean['income'].apply(income_midpoint_simple)
df_clean['rincome_mid'] = df_clean['rincome'].apply(income_midpoint_simple)

# Impute missing numeric values
numeric_cols = ['income_mid','age','childs','rincome_mid']
imputer = SimpleImputer(strategy='mean')
df_clean[numeric_cols] = imputer.fit_transform(df_clean[numeric_cols])

# ---------------------------
# TRAIN FULL MODEL
# ---------------------------
print("Training Model")
x = df_clean.drop(columns=['happy','educ','income','rincome'])
y = df_clean['happy'].apply(map_happy)

g_clf = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=4, random_state=42)
g_clf.fit(x, y, sample_weight=df_clean['wtssps'])

# ---------------------------
# SAVE MODEL AND COLUMNS
# ---------------------------
joblib.dump(g_clf, "g_clf.pkl")         # pre-trained model
joblib.dump(list(x.columns), "columns.pkl")  # column names for user input alignment

print("Model training complete and saved!")
