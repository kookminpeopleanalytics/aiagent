import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # GUI 없는 환경(GitHub Actions)에서 반드시 필요
import matplotlib.pyplot as plt
import seaborn as sns
import koreanize_matplotlib  # 한글 폰트 설정 (Malgun Gothic 대체)

# 1. 데이터 로드 (저장소 루트 기준 상대 경로)
file_path = 'project3/data/2_PAproject_2_3_EDA.csv'
df = pd.read_csv(file_path)

# 이탈 여부 파생 변수 생성 (Voluntary/Involuntary → 1, 재직 중 → 0)
df['Is_Terminated'] = df['Status'].apply(lambda x: 1 if x in ['Voluntary', 'Involuntary'] else 0)

# 결과 이미지 저장 경로
heatmap_path = 'project3/data/heatmap_attrition.png'
barplot_path = 'project3/data/barplot_performance.png'

# ---------------------------------------------------------
# 2. 부서(Department)와 직무(Job_Role)별 이탈률 히트맵
# ---------------------------------------------------------
heatmap_data = df.pivot_table(
    index='Department', columns='Job_Role',
    values='Is_Terminated', aggfunc='mean'
) * 100

plt.figure(figsize=(12, 8))
sns.heatmap(heatmap_data, annot=True, fmt=".1f", cmap='YlOrRd',
            linewidths=.5, cbar_kws={'label': '이탈률 (%)'})
plt.title('부서 및 직무별 이탈률(Attrition Rate) 히트맵 (%)', fontsize=16, pad=20)
plt.xlabel('직무 (Job Role)', fontsize=12)
plt.ylabel('부서 (Department)', fontsize=12)
plt.tight_layout()
plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"히트맵 저장 완료: {heatmap_path}")

# ---------------------------------------------------------
# 3. 성과등급(Performance_Rating)별 이탈률 막대그래프
# ---------------------------------------------------------
rating_attrition = df.groupby('Performance_Rating')['Is_Terminated'].mean() * 100

plt.figure(figsize=(10, 6))
ax = sns.barplot(x=rating_attrition.index, y=rating_attrition.values, palette='viridis')

plt.title('성과등급(Performance Rating)별 이탈률 (%)', fontsize=16, pad=20)
plt.xlabel('성과등급 (Rating)', fontsize=12)
plt.ylabel('이탈률 (%)', fontsize=12)
plt.ylim(0, max(rating_attrition.values) * 1.2)

for i, v in enumerate(rating_attrition.values):
    plt.text(i, v + 0.5, f'{v:.1f}%', ha='center', fontweight='bold', fontsize=11)

plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(barplot_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"막대그래프 저장 완료: {barplot_path}")
