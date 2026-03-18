import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import koreanize_matplotlib

# 1. 데이터 로드
# 프로젝트 폴더 내의 실제 경로를 사용합니다.
file_path = r'data/2_PAproject_2_4_machine.csv'
try:
    df = pd.read_csv(file_path)
    print("데이터를 성공적으로 불러왔습니다.")
except FileNotFoundError:
    print(f"파일을 찾을 수 없습니다: {file_path}")
    exit(1)

# 2. 독립변수(X)와 종속변수(y) 설정
X = df[['Department', 'Performance_Rating', 'Salary', 'Work_Hours']]
y = df['Left']

# 3. 전처리 파이프라인 설정
categorical_features = ['Department']
numeric_features = ['Performance_Rating', 'Salary', 'Work_Hours']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# 4. SVM 모델 구성
svm_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', SVC(kernel='rbf', probability=True, random_state=42))
])

# 5. 모델 학습
svm_model.fit(X, y)
print("모델 학습이 완료되었습니다.")

# 6. 배치 예측 수행
# 작업 폴더 내의 '2_PAproject_2_4_machine_prediction.csv' 파일을 읽어옵니다.
prediction_file_path = r'data/2_PAproject_2_4_machine_prediction.csv'
try:
    predict_df = pd.read_csv(prediction_file_path)
    print(f"예측용 데이터를 성공적으로 불러왔습니다: {prediction_file_path}")
except FileNotFoundError:
    print(f"예측용 파일을 찾을 수 없습니다: {prediction_file_path}")
    exit(1)

# 모델을 사용하여 예측 수행
predictions = svm_model.predict(predict_df)
pred_probas = svm_model.predict_proba(predict_df)

# 결과 데이터프레임 생성 (한글 예측 결과 및 확률 추가)
predict_df['Prediction'] = np.where(predictions == 1, '이직(Left)', '잔류(Stay)')
predict_df['Attrition_Probability (%)'] = np.round(pred_probas[:, 1] * 100, 2)

# 7. 결과 출력 및 저장
print("\n=== 배치 예측 결과 ===")
print(predict_df[['Department', 'Prediction', 'Attrition_Probability (%)']])

# 결과를 새로운 CSV 파일로 저장
output_file_path = r'data/2_PAproject_2_4_machine_results.csv'
predict_df.to_csv(output_file_path, index=False, encoding='utf-8-sig')
print(f"\n예측 결과가 '{output_file_path}'에 저장되었습니다.")
