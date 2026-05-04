import pandas as pd
import numpy as np
import os
import sys
import requests
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
from scipy.sparse.linalg import svds

# ==========================================
# 0. 하이퍼파라미터 및 설정
# ==========================================
# GitHub Actions를 위한 상대 경로로 수정
EXCEL_PATH = "project6/data/6_PAproject_6_4_course.xlsx"
TOP_N = 5  # 추천 과목 수
ALPHA = 0.6  # 하이브리드 기본 CF 가중치 (동적 조정됨)
N_FACTORS = 5  # SVD 잠재 요인 수

# ---------------------------------------------------------
# 터미널 출력을 파일로 저장하여 Discord로 보내기 위한 설정
# ---------------------------------------------------------
class Logger(object):
    def __init__(self, filename="project6/data/analysis_log.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# 데이터 저장 폴더가 없을 경우 생성
os.makedirs("project6/data", exist_ok=True)
sys.stdout = Logger()

# ==========================================
# 1. 데이터 로드 및 전처리 모듈
# ==========================================
def load_data(path):
    """엑셀 파일에서 5개 시트를 로드하고 유효성을 검증합니다."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {path}")

    sheets = ['courses', 'employees', 'ratings_train', 'ratings_test', 'recommend_target']
    data = {}

    with pd.ExcelFile(path) as xls:
        for sheet in sheets:
            if sheet not in xls.sheet_names:
                raise ValueError(f"시트가 존재하지 않습니다: {sheet}")
            data[sheet] = pd.read_excel(xls, sheet_name=sheet)

    return data


def build_rating_matrix(df_ratings, df_employees, df_courses):
    """직원 x 교육과정 평점 행렬을 생성합니다."""
    matrix = df_ratings.pivot(index='emp_id', columns='course_id', values='rating').fillna(0)

    # 모든 직원과 모든 과목이 행렬에 포함되도록 재정렬
    all_emps = df_employees['emp_id'].unique()
    all_courses = df_courses['course_id'].unique()

    matrix = matrix.reindex(index=all_emps, columns=all_courses, fill_value=0)
    return matrix


# ==========================================
# 2. CollaborativeFilter 클래스 (SVD)
# ==========================================
class CollaborativeFilter:
    def __init__(self, rating_matrix, n_factors=5):
        self.matrix = rating_matrix
        self.n_factors = n_factors
        self.preds_df = None

    def fit(self):
        # Row Centering (사용자 평균 기반 정규화)
        user_ratings_mean = np.mean(self.matrix.values, axis=1)
        matrix_centered = self.matrix.values - user_ratings_mean.reshape(-1, 1)

        # SVD 분해
        U, sigma, Vt = svds(matrix_centered, k=self.n_factors)
        sigma = np.diag(sigma)

        # 행렬 복원 및 평균 다시 더하기
        svd_preds = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)
        self.preds_df = pd.DataFrame(svd_preds, columns=self.matrix.columns, index=self.matrix.index)

    def recommend(self, emp_id, top_n=5, exclude_rated=True):
        if emp_id not in self.preds_df.index:
            return pd.Series(dtype=float)

        user_preds = self.preds_df.loc[emp_id].copy()

        if exclude_rated:
            # 이미 수강한(rating > 0) 과목은 -inf로 마스킹
            already_rated = self.matrix.loc[emp_id]
            user_preds[already_rated > 0] = -np.inf

        return user_preds.sort_values(ascending=False).head(top_n)


# ==========================================
# 3. ContentFilter 클래스 (CB)
# ==========================================
class ContentFilter:
    def __init__(self, df_employees, df_courses, rating_matrix):
        self.df_emp = df_employees
        self.df_course = df_courses
        self.rating_matrix = rating_matrix
        self.course_sim_matrix = None

    def recommend(self, emp_id, top_n=5, exclude_rated=True):
        # CB 점수 (0~1) - 본 시나리오에선 예시 로직 유지
        all_courses = self.rating_matrix.columns
        # 시드 고정하여 재현성 확보
        np.random.seed(42)
        scores = pd.Series(np.random.rand(len(all_courses)), index=all_courses)

        if exclude_rated:
            already_rated = self.rating_matrix.loc[emp_id]
            scores = scores[already_rated == 0]

        return scores.sort_values(ascending=False).head(top_n)

    def get_all_scores(self, emp_id):
        """정규화를 위해 모든 과목에 대한 점수 반환"""
        all_courses = self.rating_matrix.columns
        np.random.seed(42)
        scores = pd.Series(np.random.rand(len(all_courses)), index=all_courses)
        return scores


# ==========================================
# 4. HybridRecommender 클래스
# ==========================================
class HybridRecommender:
    def __init__(self, cf_model, cb_model, rating_matrix):
        self.cf = cf_model
        self.cb = cb_model
        self.rating_matrix = rating_matrix
        self.scaler = MinMaxScaler()

    def recommend(self, emp_id, top_n=5, exclude_rated=True, base_alpha=0.6):
        # 사전 필터링: 수강 과목 제외
        rated_items = self.rating_matrix.loc[emp_id]
        unrated_idx = rated_items[rated_items == 0].index if exclude_rated else rated_items.index

        # 각 모델 점수 획득
        cf_scores = self.cf.preds_df.loc[emp_id][unrated_idx]
        cb_scores = self.cb.get_all_scores(emp_id)[unrated_idx]

        # MinMax Scaling 필수
        cf_norm = self.scaler.fit_transform(cf_scores.values.reshape(-1, 1)).flatten()
        cb_norm = self.scaler.fit_transform(cb_scores.values.reshape(-1, 1)).flatten()

        # 동적 가중치 (Dynamic Alpha)
        num_rated = (self.rating_matrix.loc[emp_id] > 0).sum()
        effective_alpha = 0.2 if num_rated <= 2 else base_alpha

        # 가중합 계산
        hybrid_scores = (effective_alpha * cf_norm) + ((1 - effective_alpha) * cb_norm)

        result = pd.Series(hybrid_scores, index=unrated_idx).sort_values(ascending=False).head(top_n)
        return result, effective_alpha


# ==========================================
# 5. 평가 함수
# ==========================================
def evaluate_models(cf_model, rating_train, rating_test):
    # RMSE 계산 (CF 기준)
    test_data = rating_test[rating_test['rating'] > 0]
    y_true = []
    y_pred = []

    for _, row in test_data.iterrows():
        eid, cid, rat = row['emp_id'], row['course_id'], row['rating']
        if eid in cf_model.preds_df.index and cid in cf_model.preds_df.columns:
            y_true.append(rat)
            y_pred.append(cf_model.preds_df.loc[eid, cid])

    rmse = np.sqrt(mean_squared_error(y_true, y_pred)) if y_true else 0
    return rmse


# ==========================================
# 6. Main 실행부
# ==========================================
def main():
    print("══" * 30)
    print("   교육과정 추천 시스템 가동 (GitHub Actions Mode)")
    print("══" * 30)

    # 데이터 로드
    try:
        data = load_data(EXCEL_PATH)
    except Exception as e:
        print(f"[오류] 데이터를 로드할 수 없습니다: {e}")
        return

    # 행렬 빌드 및 모델 학습
    rating_matrix = build_rating_matrix(data['ratings_train'], data['employees'], data['courses'])

    cf = CollaborativeFilter(rating_matrix, n_factors=N_FACTORS)
    cf.fit()

    cb = ContentFilter(data['employees'], data['courses'], rating_matrix)

    hybrid = HybridRecommender(cf, cb, rating_matrix)

    # 평가 출력
    rmse = evaluate_models(cf, data['ratings_train'], data['ratings_test'])
    print(f"시스템 성능지표 (CF RMSE): {rmse:.4f}")
    print("──" * 30)

    # 추천 대상자 처리
    target_emps = data['recommend_target']
    final_recommendations = []

    for _, row in target_emps.iterrows():
        emp_id = row['emp_id']
        num_rated = (rating_matrix.loc[emp_id] > 0).sum()

        # 정보 출력
        emp_info = data['employees'][data['employees']['emp_id'] == emp_id].iloc[0]
        print(f"[{emp_id}] {emp_info['dept']} | {emp_info['grade']} | 근속 {emp_info['tenure']}년 | 기수강 {num_rated}건")

        # 추천 결과 요약 (콘솔 출력용)
        hyb_res, alpha_used = hybrid.recommend(emp_id, top_n=TOP_N, base_alpha=ALPHA)
        print("  하이브리드 추천 결과:")
        for i, (cid, score) in enumerate(hyb_res.items(), 1):
            c_name = data['courses'][data['courses']['course_id'] == cid]['name'].values[0]
            print(f"    {i}. [{cid}] {c_name[:15]:<15} {score:.3f}")

            # 엑셀 저장용 데이터 축적 (Top-3)
            if i <= 3:
                final_recommendations.append({
                    'emp_id': emp_id,
                    'rank': i,
                    'course_id': cid,
                    'course_name': c_name,
                    'score': score,
                    'type': 'Hybrid'
                })
        print("──" * 30)

    # 엑셀 파일 저장
    df_final = pd.DataFrame(final_recommendations)
    output_xlsx = "project6/data/recommendations.xlsx"
    df_final.to_excel(output_xlsx, index=False)
    print(f"\n[알림] 하이브리드 Top-3 추천 결과가 저장되었습니다: {output_xlsx}")

    # ---------------------------------------------------------
    # 7. Discord Webhook으로 결과 전송
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
        msg = "🎯 **[Project 6] 교육과정 추천 시스템 분석 완료**\n직원별 하이브리드 추천 결과와 성능 지표를 확인해 주세요."
        log_path = "project6/data/analysis_log.txt"
        res = send_discord_files(DISCORD_WEBHOOK_URL, msg, [log_path, output_xlsx])
        
        if res.status_code in [200, 204]:
            print("[Discord 전송 성공]")
        else:
            print(f"[Discord 전송 실패: {res.status_code}]")
    else:
        print("\n[DISCORD_WEBHOOK_URL 미설정으로 Discord 전송 스킵]")


if __name__ == "__main__":
    main()
