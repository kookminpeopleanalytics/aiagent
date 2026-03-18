import pandas as pd
import numpy as np
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

def send_email(subject, html_body):
    # 환경 변수에서 설정 로드
    smtp_server = os.environ.get('SMTP_SERVER', 'smtp.gmail.com')
    smtp_port = int(os.environ.get('SMTP_PORT', 587))
    smtp_user = os.environ.get('SMTP_USER')
    smtp_password = os.environ.get('SMTP_PASSWORD')
    receiver_email = os.environ.get('RECEIVER_EMAIL')

    if not all([smtp_user, smtp_password, receiver_email]):
        print("이메일 설정 정보가 부족합니다. GitHub Secrets 설정을 확인하세요.")
        return

    msg = MIMEMultipart()
    msg['From'] = smtp_user
    msg['To'] = receiver_email
    msg['Subject'] = subject

    # HTML 본문 추가
    msg.attach(MIMEText(html_body, 'html', 'utf-8'))

    try:
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(smtp_user, smtp_password)
        server.send_message(msg)
        server.quit()
        print(f"분석 결과 이메일이 {receiver_email}로 성공적으로 전송되었습니다.")
    except Exception as e:
        print(f"이메일 전송 중 오류 발생: {e}")

# 1. 데이터 로드
file_path = os.path.join('data', '2_PAproject_2_4_machine.csv')
try:
    df = pd.read_csv(file_path)
    print(f"데이터를 성공적으로 불러왔습니다: {file_path}")
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

# 6. 다중 직원 예측 (2_PAproject_2_4_machine_prediction.csv 활용)
predict_file_path = os.path.join('data', '2_PAproject_2_4_machine_prediction.csv')
try:
    predict_df = pd.read_csv(predict_file_path)
    print(f"예측 대상 데이터를 성공적으로 불러왔습니다: {predict_file_path}")
except FileNotFoundError:
    print(f"예측 대상 파일을 찾을 수 없습니다: {predict_file_path}")
    exit(1)

# 모델을 사용하여 예측 수행
predictions = svm_model.predict(predict_df)
pred_probas = svm_model.predict_proba(predict_df)

# 7. 결과 출력 및 HTML 이메일 내용 구성
html_content = """
<html>
<head>
    <style>
        table { border-collapse: collapse; width: 100%; max-width: 800px; }
        th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
        th { background-color: #f2f2f2; color: #333; }
        .stay { color: #2ecc71; font-weight: bold; }
        .leave { color: #e74c3c; font-weight: bold; }
        h2 { color: #2c3e50; }
    </style>
</head>
<body>
    <h2>📊 직원 퇴직 확률 예측 결과 보고서</h2>
    <p>데이터 변경에 따라 자동으로 실행된 분석 결과입니다.</p>
    <table>
        <tr>
            <th>순번</th>
            <th>부서</th>
            <th>성과 등급</th>
            <th>급여</th>
            <th>근무 시간</th>
            <th>예측 결과</th>
            <th>이직 확률</th>
        </tr>
"""

for i, (pred, proba) in enumerate(zip(predictions, pred_probas)):
    status = "이직(Left)" if pred == 1 else "잔류(Stay)"
    status_class = "leave" if pred == 1 else "stay"
    leave_prob = proba[1] * 100
    emp_info = predict_df.iloc[i]
    
    html_content += f"""
        <tr>
            <td>{i+1}</td>
            <td>{emp_info['Department']}</td>
            <td>{emp_info['Performance_Rating']}</td>
            <td>{emp_info['Salary']}</td>
            <td>{emp_info['Work_Hours']}</td>
            <td class="{status_class}">{status}</td>
            <td>{leave_prob:.2f}%</td>
        </tr>
    """

html_content += """
    </table>
    <p style="color: #7f8c8d; font-size: 0.9em; margin-top: 20px;">
        ※ 이 메일은 GitHub Actions를 통해 자동으로 발송되었습니다.
    </p>
</body>
</html>
"""

# 콘솔에도 출력
print("\n" + html_content[:500] + "...") # 요약 출력

# 이메일 전송
send_email("[분석 결과] 직원 퇴직 확률 예측 리포트", html_content)
