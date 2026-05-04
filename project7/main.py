import pandas as pd
import numpy as np
import os
import sys
import requests
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import koreanize_matplotlib
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

# 1. 파일 경로 설정 (GitHub Actions용 상대 경로)
file_path = "project7/data/7_PAproject_7_3_SNA.xlsx"

def run_sna_analysis(path):
    print("══" * 25)
    print("   조직 소셜 네트워크 분석(SNA) 시스템 가동")
    print("══" * 25)

    # 데이터 로드
    try:
        if not os.path.exists(path):
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {path}")
        
        emp_df = pd.read_excel(path, sheet_name='employees')
        edge_df = pd.read_excel(path, sheet_name='edges')
    except Exception as e:
        print(f"파일을 읽는 중 오류가 발생했습니다: {e}")
        return

    # 2. 데이터 전처리 및 지표 계산
    target_time = '2025Q2'
    # 시각화용 데이터 (모든 Tie 포함)
    viz_edges = edge_df[(edge_df['time_id'] == target_time) & 
                        (edge_df['source'] != edge_df['target'])].copy()
    
    # 지표 계산용 데이터 (tie_binary == 1 필터링 유지)
    metric_edges = viz_edges[viz_edges['tie_binary'] == 1].copy()

    # SNA 지표 계산
    out_degree_s = metric_edges.groupby('source')['target'].nunique().rename('out_degree')
    in_degree_s = metric_edges.groupby('target')['source'].nunique().rename('in_degree')
    
    sent_int = metric_edges.groupby('source')['interaction_count'].sum()
    recv_int = metric_edges.groupby('target')['interaction_count'].sum()
    total_interaction = sent_int.add(recv_int, fill_value=0).rename('total_interaction')

    conn_list = pd.concat([
        metric_edges[['source', 'target']].rename(columns={'source': 'me', 'target': 'other'}),
        metric_edges[['target', 'source']].rename(columns={'target': 'me', 'source': 'other'})
    ])
    total_unique_connections = conn_list.groupby('me')['other'].nunique().rename('total_unique_connections')

    metrics_df = pd.concat([total_unique_connections, out_degree_s, in_degree_s, total_interaction], axis=1)
    final_df = emp_df.merge(metrics_df, left_on='employee_id', right_index=True, how='left')
    
    cols_to_fill = ['total_unique_connections', 'out_degree', 'in_degree', 'total_interaction']
    final_df[cols_to_fill] = final_df[cols_to_fill].fillna(0).astype(int)
    final_df = final_df.sort_values(by=['total_unique_connections', 'total_interaction'], ascending=False).reset_index(drop=True)
    final_df.insert(0, 'rank', final_df.index + 1)

    csv_output = f"project7/data/SNA_Analysis_Result_{target_time}.csv"
    final_df.to_csv(csv_output, index=False, encoding='utf-8-sig')

    # 3. NetworkX 시각화
    print(f"\n[시각화 생성 중... 시점: {target_time}]")
    G = nx.Graph()
    for _, row in emp_df.iterrows():
        G.add_node(row['employee_id'], name=row['name'], department=row['department'], team=row['team'])

    # Undirected weight 집계
    viz_edges['node_a'] = viz_edges.apply(lambda x: min(x['source'], x['target']), axis=1)
    viz_edges['node_b'] = viz_edges.apply(lambda x: max(x['source'], x['target']), axis=1)
    df_grouped = viz_edges.groupby(['node_a', 'node_b'], as_index=False)['interaction_count'].sum()

    for _, row in df_grouped.iterrows():
        G.add_edge(row['node_a'], row['node_b'], weight=row['interaction_count'])

    # 고립 노드 제거
    G.remove_nodes_from(list(nx.isolates(G)))

    # 레이아웃 및 스타일 설정
    plt.figure(figsize=(15, 10))
    pos = nx.spring_layout(G, k=0.5, seed=42)

    depts = list(emp_df['department'].unique())
    color_palette = plt.cm.get_cmap('Set3', len(depts))
    dept_colors = {dept: color_palette(i) for i, dept in enumerate(depts)}
    
    node_colors = [dept_colors[G.nodes[n]['department']] for n in G.nodes()]
    degrees = dict(G.degree())
    node_sizes = [v * 100 for v in degrees.values()]
    
    weights = [G[u][v]['weight'] for u, v in G.edges()]
    edge_widths = [w * 0.5 for w in weights] 

    top_15 = sorted(degrees, key=degrees.get, reverse=True)[:15]
    labels = {n: G.nodes[n]['name'] for n in top_15}

    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.3, edge_color='gray')
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.9)
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=10) # koreanize_matplotlib 적용됨

    handles = [mpatches.Patch(color=color, label=dept) for dept, color in dept_colors.items()]
    plt.legend(handles=handles, title="Departments", loc='upper right', bbox_to_anchor=(1.2, 1))
    plt.title(f"{target_time} Social Network Analysis Map", fontsize=15)
    plt.axis('off')
    
    png_output = "project7/data/sna_network_map.png"
    plt.savefig(png_output, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"네트워크 지도 저장 완료: {png_output}")

    # 리포트 출력
    print("-" * 50)
    print(f"분석 기준 시점: {target_time}")
    print(f"분석 대상 인원: {len(emp_df)}명")
    print("-" * 50)
    print("\n[네트워크 분석 상위 5명]")
    print(final_df[['rank', 'name', 'department', 'total_unique_connections', 'total_interaction']].head(5).to_string(index=False))

    # ---------------------------------------------------------
    # 4. Discord Webhook으로 결과 전송
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
        msg = "🕸️ **[Project 7] 조직 소셜 네트워크 분석(SNA) 완료**\n네트워크 지도 이미지와 상세 지표 리포트를 확인해 주세요."
        log_path = "project7/data/analysis_log.txt"
        res = send_discord_files(DISCORD_WEBHOOK_URL, msg, [log_path, csv_output, png_output])
        
        if res.status_code in [200, 204]:
            print("[Discord 전송 성공]")
        else:
            print(f"[Discord 전송 실패: {res.status_code}]")
    else:
        print("\n[DISCORD_WEBHOOK_URL 미설정으로 Discord 전송 스킵]")

if __name__ == "__main__":
    run_sna_analysis(file_path)

# 스크립트 실행
if __name__ == "__main__":
    run_sna_analysis(file_path)
