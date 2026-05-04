import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from google import genai
import os
import requests
import sys

# ---------------------------------------------------------
# 터미널 출력을 파일로 저장하여 Discord로 보내기 위한 설정
# ---------------------------------------------------------
class Logger(object):
    def __init__(self, filename="project5/data/analysis_log.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# 데이터 저장 폴더가 없을 경우 생성
os.makedirs("project5/data", exist_ok=True)
sys.stdout = Logger()

# 1. 파일 경로 설정 및 데이터 로드 (GitHub Actions용 상대 경로로 수정)
file_path = "project5/data/6_PAproject_6_2_Leadership.xlsx"

try:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {file_path}")

    df = pd.read_excel(file_path)
    # 결측치 제거 (분석의 정확성을 위해)
    df = df.dropna(subset=['Employee_Group', 'Pre_Training_Score', 'Post_Training_Score'])
    print("데이터 로드 및 전처리 완료!")

    # 2. ANOVA 분석 (통제 변수 없음)
    groups = [group['Post_Training_Score'] for name, group in df.groupby('Employee_Group')]
    f_stat_anova, p_val_anova = stats.f_oneway(*groups)
    
    anova_res = f"ANOVA 결과: F={f_stat_anova:.4f}, p-value={p_val_anova:.4f}"

    # 3. ANCOVA 분석 (Pre_Training_Score를 공변량으로 통제)
    # 공식: Post_Score ~ Employee_Group + Pre_Score
    model = ols('Post_Training_Score ~ C(Employee_Group) + Pre_Training_Score', data=df).fit()
    ancova_table = sm.stats.anova_lm(model, typ=2) # Type 2 Sum of Squares
    
    # ANCOVA에서 Employee_Group의 유의성 추출
    f_stat_ancova = ancova_table.loc['C(Employee_Group)', 'F']
    p_val_ancova = ancova_table.loc['C(Employee_Group)', 'PR(>F)']
    
    ancova_res = f"ANCOVA 결과 (사전점수 통제): F={f_stat_ancova:.4f}, p-value={p_val_ancova:.4f}"

    # 결과 요약
    full_result_str = f"""
    [분석 결과 요약]
    1. {anova_res}
    2. {ancova_res}
    """
    print(full_result_str)

    # 4. Gemini API를 통한 비교 해석 (환경변수 사용)
    GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
    if not GOOGLE_API_KEY:
        print("GOOGLE_API_KEY 환경변수가 설정되지 않았습니다.")
        gemini_response_text = "API 키 미설정으로 AI 해석을 수행할 수 없습니다."
    else:
        client = genai.Client(api_key=GOOGLE_API_KEY)
        
        prompt = f"""
        당신은 고급 통계 컨설턴트입니다. 다음 ANOVA와 ANCOVA 분석 결과를 비교하여 해석해 주세요.
        
        [데이터 정보]
        - 종속변수: 교육 후 점수 (Post_Training_Score)
        - 독립변수: 직원 그룹 (Employee_Group)
        - 공변량: 교육 전 점수 (Pre_Training_Score)
        
        [분석 결과]
        {full_result_str}
        
        [요청 사항]
        1. ANOVA와 ANCOVA 결과의 차이점을 설명해 주세요. (예: 사전 점수를 통제했을 때 그룹 간 차이가 여전히 유의미한지 등)
        2. 사전 점수(Pre_Training_Score)가 사후 점수에 미치는 영향이 컸는지 추론해 주세요.
        3. 이 결과를 바탕으로, 교육 프로그램이 모든 그룹에 공정하게 효과적이었는지, 아니면 특정 그룹에 치중되었는지 인사이트를 제공해 주세요.
        4. 분석 결과를 경영진에게 보고하는 형태의 요약 문장을 만들어 주세요.
        
        한국어로 전문적이면서도 이해하기 쉽게 설명해 주세요.
        """

        print("Gemini API에 비교 분석을 요청 중입니다...")
        
        response = client.models.generate_content(
            model="gemini-2.5-flash", 
            contents=prompt
        )
        gemini_response_text = response.text

        print("\n" + "="*60)
        print("Gemini의 ANOVA vs ANCOVA 비교 해석")
        print("="*60)
        print(gemini_response_text)

    # ---------------------------------------------------------
    # 5. Discord Webhook으로 결과 전송
    # ---------------------------------------------------------
    sys.stdout.flush()
    DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL")
    
    if DISCORD_WEBHOOK_URL:
        def send_discord_files(webhook_url, message, file_paths):
            payload = {"content": message}
            files = {}
            opened_files = []
            try:
                for i, path in enumerate(file_paths):
                    if os.path.exists(path):
                        f = open(path, "rb")
                        opened_files.append(f)
                        files[f"file{i}"] = (os.path.basename(path), f)
                return requests.post(webhook_url, data=payload, files=files)
            finally:
                for f in opened_files:
                    f.close()

        print("\nDiscord로 결과를 전송 중입니다...")
        msg = "🎓 **[Project 5] ANOVA vs ANCOVA 리더십 교육 효과 분석 완료**\nGemini의 AI 해석과 상세 로그를 확인해 주세요."
        log_path = "project5/data/analysis_log.txt"
        res = send_discord_files(DISCORD_WEBHOOK_URL, msg, [log_path])
        
        if res.status_code in [200, 204]:
            print("[Discord 전송 성공]")
        else:
            print(f"[Discord 전송 실패: {res.status_code}]")
    else:
        print("\n[DISCORD_WEBHOOK_URL 미설정으로 Discord 전송 스킵]")

except Exception as e:
    print(f"오류가 발생했습니다: {e}")
