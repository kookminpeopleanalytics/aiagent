import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import koreanize_matplotlib
import os
import sys
import requests

# ---------------------------------------------------------
# 터미널 출력을 파일로 저장하여 Discord로 보내기 위한 설정
# ---------------------------------------------------------
class Logger(object):
    def __init__(self, filename="project8/data/analysis_log.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# 데이터 저장 폴더가 없을 경우 생성
os.makedirs("project8/data", exist_ok=True)
sys.stdout = Logger()

# 1. 데이터 로드 (GitHub Actions용 상대 경로)
file_path = 'project8/data/7_PAproject_7_3_SNA.xlsx'

def main():
    print("══" * 25)
    print("   조직 SNA 시각화 분석 시스템 가동")
    print("══" * 25)

    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")

        # 시트별 데이터 읽기
        df_employees = pd.read_excel(file_path, sheet_name='employees')
        df_edges = pd.read_excel(file_path, sheet_name='edges')

        # 2. 데이터 전처리 (Edges)
        target_time = '2025Q2'
        df_filtered = df_edges[(df_edges['time_id'] == target_time) & 
                               (df_edges['source'] != df_edges['target'])].copy()

        df_filtered['node_a'] = df_filtered.apply(lambda x: min(x['source'], x['target']), axis=1)
        df_filtered['node_b'] = df_filtered.apply(lambda x: max(x['source'], x['target']), axis=1)
        df_grouped = df_filtered.groupby(['node_a', 'node_b'], as_index=False)['interaction_count'].sum()

        # 3. NetworkX 그래프 생성
        G = nx.Graph()
        for _, row in df_employees.iterrows():
            G.add_node(row['employee_id'], name=row['name'], department=row['department'], team=row['team'])

        for _, row in df_grouped.iterrows():
            G.add_edge(row['node_a'], row['node_b'], weight=row['interaction_count'])

        G.remove_nodes_from(list(nx.isolates(G)))

        # 4. 시각화 설정 준비
        plt.figure(figsize=(15, 10))
        pos = nx.spring_layout(G, k=0.5, seed=42)

        departments = list(df_employees['department'].unique())
        color_map_palette = plt.cm.get_cmap('Set3', len(departments))
        dept_color_dict = {dept: color_map_palette(i) for i, dept in enumerate(departments)}
        node_colors = [dept_color_dict[G.nodes[n]['department']] for n in G.nodes()]

        degrees = dict(G.degree())
        node_sizes = [v * 100 for v in degrees.values()] 

        weights = [G[u][v]['weight'] for u, v in G.edges()]
        edge_widths = [w * 0.5 for w in weights] 

        top_15_nodes = sorted(degrees, key=degrees.get, reverse=True)[:15]
        labels = {n: G.nodes[n]['name'] for n in top_15_nodes}

        # 5. 그래프 그리기
        nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.3, edge_color='gray')
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.9)
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=10) # koreanize_matplotlib 적용

        legend_handles = [mpatches.Patch(color=color, label=dept) for dept, color in dept_color_dict.items()]
        plt.legend(handles=legend_handles, title="Departments", loc='upper right', bbox_to_anchor=(1.2, 1))

        plt.title(f"{target_time} Social Network Analysis Map", fontsize=15)
        plt.axis('off')
        plt.tight_layout()
        
        png_output = "project8/data/sna_network_map.png"
        plt.savefig(png_output, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\n네트워크 지도 저장 완료: {png_output}")

        # ---------------------------------------------------------
        # 6. Discord Webhook으로 결과 전송
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

            print("Discord로 결과를 전송 중입니다...")
            msg = "🎨 **[Project 8] 조직 SNA 시각화 분석 완료**\n부서별 연결 구조를 시각화한 네트워크 지도를 확인해 주세요."
            log_path = "project8/data/analysis_log.txt"
            res = send_discord_files(DISCORD_WEBHOOK_URL, msg, [log_path, png_output])
            
            if res.status_code in [200, 204]:
                print("[Discord 전송 성공]")
            else:
                print(f"[Discord 전송 실패: {res.status_code}]")
        else:
            print("\n[DISCORD_WEBHOOK_URL 미설정으로 Discord 전송 스킵]")

    except Exception as e:
        print(f"오류가 발생했습니다: {e}")

if __name__ == "__main__":
    main()
