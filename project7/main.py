import pandas as pd
import numpy as np
import os
import sys
import requests
from pathlib import Path

# ---------------------------------------------------------
# 터미널 출력을 파일로 저장하여 Discord로 보내기 위한 설정
# ---------------------------------------------------------
class Logger(object):
    def __init__(self, filename="project7/data/analysis_log.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# 데이터 저장 폴더가 없을 경우 생성
os.makedirs("project7/data", exist_ok=True)
sys.stdout = Logger()

# 1. 파일 경로 설정 (GitHub Actions용 상대 경로로 수정)
file_path = "project7/data/7_PAproject_7_3_SNA.xlsx"

def run_sna_analysis(path):
    # 파일 로드
    try:
        if not os.path.exists(path):
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {path}")
        
        # 지정된 시트명으로 데이터 읽기
        emp_df = pd.read_excel(path, sheet_name='employees')
        edge_df = pd.read_excel(path, sheet_name='edges')
    except Exception as e:
        print(f"파일을 읽는 중 오류가 발생했습니다: {e}")
        return

    # 2. 데이터 필터링 및 전처리
    target_time = '2025Q2'
    filtered_edges = edge_df[
        (edge_df['time_id'] == target_time) & 
        (edge_df['tie_binary'] == 1) & 
        (edge_df['source'] != edge_df['target'])
    ].copy()

    # 3. SNA 지표 계산
    out_degree = filtered_edges.groupby('source')['target'].nunique().rename('out_degree')
    in_degree = filtered_edges.groupby('target')['source'].nunique().rename('in_degree')
    
    sent_int = filtered_edges.groupby('source')['interaction_count'].sum()
    recv_int = filtered_edges.groupby('target')['interaction_count'].sum()
    total_interaction = sent_int.add(recv_int, fill_value=0).rename('total_interaction')

    conn_list = pd.concat([
        filtered_edges[['source', 'target']].rename(columns={'source': 'me', 'target': 'other'}),
        filtered_edges[['target', 'source']].rename(columns={'target': 'me', 'source': 'other'})
    ])
    total_unique_connections = conn_list.groupby('me')['other'].nunique().rename('total_unique_connections')

    # 4. 결과 데이터 통합
    metrics_df = pd.concat([total_unique_connections, out_degree, in_degree, total_interaction], axis=1)
    final_df = emp_df.merge(metrics_df, left_on='employee_id', right_index=True, how='left')
    
    cols_to_fill = ['total_unique_connections', 'out_degree', 'in_degree', 'total_interaction']
    final_df[cols_to_fill] = final_df[cols_to_fill].fillna(0).astype(int)

    # 5. 정렬 및 순위 부여
    final_df = final_df.sort_values(
        by=['total_unique_connections', 'total_interaction', 'out_degree'], 
        ascending=False
    ).reset_index(drop=True)
    
    final_df.insert(0, 'rank', final_df.index + 1)

    # 6. 결과 저장
    output_filename = f"project7/data/SNA_Analysis_Result_{target_time}.csv"
    final_df.to_csv(output_filename, index=False, encoding='utf-8-sig')

    # 7. 출력 결과 리포트
    print("-" * 50)
    print(f"분석 기준 시점: {target_time}")
    print(f"분석 대상 인원: {len(emp_df)}명")
    print(f"활성화된 고유 관계 수: {len(filtered_edges)}건")
    print("-" * 50)
    print("\n[네트워크 분석 상위 10명 결과]")
    display_cols = ['rank', 'employee_id', 'name', 'department', 'team', 'total_unique_connections', 'total_interaction']
    print(final_df[display_cols].head(10).to_string(index=False))
    
    if not final_df.empty:
        top_1 = final_df.iloc[0]
        print("\n" + "=" * 50)
        print(f"[최고 연결 핵심 인물(Hub) 요약]")
        print(f"분석 결과, 우리 조직에서 가장 연결성이 높은 인물은 {top_1['department']} {top_1['team']}팀의 {top_1['name']}님입니다.")
        print(f"이 직원은 총 {top_1['total_unique_connections']}명의 동료와 {top_1['total_interaction']}회의 상호작용을 기록하고 있습니다.")
        print("=" * 50)

    print(f"\n상세 분석 결과가 '{output_filename}' 파일로 저장되었습니다.")

    # ---------------------------------------------------------
    # 8. Discord Webhook으로 결과 전송
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
        msg = "🕸️ **[Project 7] 조직 소셜 네트워크 분석(SNA) 완료**\n2025Q2 기준 네트워크 핵심 인물(Hub) 분석 결과와 상세 리포트를 확인해 주세요."
        log_path = "project7/data/analysis_log.txt"
        res = send_discord_files(DISCORD_WEBHOOK_URL, msg, [log_path, output_filename])
        
        if res.status_code in [200, 204]:
            print("[Discord 전송 성공]")
        else:
            print(f"[Discord 전송 실패: {res.status_code}]")
    else:
        print("\n[DISCORD_WEBHOOK_URL 미설정으로 Discord 전송 스킵]")

# 스크립트 실행
if __name__ == "__main__":
    run_sna_analysis(file_path)
