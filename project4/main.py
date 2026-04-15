import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import sys

# ---------------------------------------------------------
# 터미널 출력을 파일로 저장하여 이메일로 첨부하기 위한 설정
# ---------------------------------------------------------
class Logger(object):
    def __init__(self, filename="project4/data/analysis_log.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = Logger()

# 1. 데이터 불러오기 (상대 경로로 수정)
file_path = "project4/data/5_PAproject_5_4_rater.xlsx"
df = pd.read_excel(file_path)

# ---------------------------------------------------------
# [추가된 해결 코드] 데이터 타입 호환성 문제 해결
# ---------------------------------------------------------
# 범주형 변수들을 'category' 타입으로 변환 (Patsy 호환성 확보)
categorical_cols = ['department', 'job_level', 'rater_id']
for col in categorical_cols:
    df[col] = df[col].astype('category')

# 결측치가 있으면 HLM 모델이 돌아가지 않으므로 제거 (선택 사항)
df = df.dropna(subset=['rating_score', 'performance_true', 'goal_difficulty'])
# ---------------------------------------------------------

print("--- [1] 데이터 로드 및 타입 변환 완료 ---")
print(df.info()) # 데이터 타입 확인

# 2. 기술통계
rater_stats = df.groupby('rater_id', observed=True)['rating_score'].agg(['count', 'mean', 'std']).reset_index()
print("\n--- [2] 평가자별 기술통계 ---")
print(rater_stats)

# 3. ANOVA 분석
groups = [group['rating_score'].values for name, group in df.groupby('rater_id', observed=True)]
f_stat, p_val = stats.f_oneway(*groups)
print(f"\n--- [3] ANOVA 결과: F={f_stat:.4f}, p-value={p_val:.4f} ---")

# 4. HLM (혼합 효과 모형) 적합
# C() 문법을 제거하고 범주형으로 변환된 변수를 직접 사용하거나 유지합니다.
model_formula = "rating_score ~ performance_true + goal_difficulty + age + tenure_years + department + job_level"
hlm_model = smf.mixedlm(model_formula, df, groups=df["rater_id"])
hlm_result = hlm_model.fit()

print("\n--- [4] HLM 분석 요약 ---")
print(hlm_result.summary())

# 5. ICC 계산
var_resid = hlm_result.scale
var_rater = float(hlm_result.cov_re.iloc[0, 0])
icc = var_rater / (var_rater + var_resid)
print(f"\n--- [5] ICC: {icc:.4f} ---")

# 6. 평가자별 Random Effect (Bias) 추출
random_effects = hlm_result.random_effects
bias_df = pd.DataFrame.from_dict(random_effects, orient='index').reset_index()
bias_df.columns = ['rater_id', 'random_effect']

def judge_bias(x):
    if x > 0.1: return 'Leniency (관대화)'
    elif x < -0.1: return 'Severity (엄격화)'
    else: return 'Neutral (중립)'

bias_df['bias_type'] = bias_df['random_effect'].apply(judge_bias)
print("\n--- [6] 평가자별 편향 분석 ---")
print(bias_df.sort_values(by='random_effect', ascending=False))

# 7. 평가점수 보정값 계산
df = df.merge(bias_df, on='rater_id', how='left')
df['adjusted_rating_score'] = df['rating_score'] - df['random_effect']

print("\n--- [7] 보정된 평가 점수 샘플 ---")
print(df[['employee_id', 'rater_id', 'rating_score', 'random_effect', 'adjusted_rating_score']].head())

# 엑셀 저장 (상대 경로로 수정)
output_path = "project4/data/adjusted_results.xlsx"
df.to_excel(output_path, index=False)
print(f"\n--- 분석 결과가 '{output_path}'에 저장되었습니다. ---")
