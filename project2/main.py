import time
import os
from google import genai

# 1. 환경변수에서 API 키 읽기 (GitHub Secrets에 GOOGLE_API_KEY로 등록 필요)
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY 환경변수가 설정되지 않았습니다. GitHub Secrets를 확인해주세요.")

client = genai.Client(api_key=GOOGLE_API_KEY)

# 2. 동영상 파일 경로 설정 (저장소 루트 기준 상대 경로)
video_path = "project2/data/kim.mp4"

# 3. 분석 결과를 저장할 파일 경로
result_path = "project2/data/analysis_result.txt"

video_file = None

try:
    # 4. 파일 업로드
    print("동영상을 업로드하는 중...")
    video_file = client.files.upload(file=video_path)
    print(f"업로드 완료: {video_file.uri}")

    # 5. 동영상 처리 대기 (Google 서버에서 처리하는 시간)
    print("Google 서버에서 동영상을 처리 중입니다. 잠시만 기다려주세요...")
    while video_file.state.name == "PROCESSING":
        print(".", end="", flush=True)
        time.sleep(5)  # 5초마다 상태 확인
        video_file = client.files.get(name=video_file.name)
    print()

    # 처리 실패 시 예외 발생
    if video_file.state.name == "FAILED":
        raise ValueError("동영상 처리에 실패했습니다. 파일 형식을 확인해주세요.")
    print("동영상 처리 완료!")

    # 6. Gemini 모델로 동영상 분석
    model_id = "gemini-2.5-flash"
    prompt = "이 동영상에서 인물이 어떤 행동을 하고 있는지 시간 흐름에 따라 상세히 분석해줘."

    print(f"AI({model_id})가 동영상을 분석하고 있습니다...")
    response = client.models.generate_content(
        model=model_id,
        contents=[video_file, prompt]
    )

    # 7. 결과 출력 및 파일 저장 (이메일 첨부용)
    result_text = response.text
    print("\n==========[분석 결과] ==========")
    print(result_text)
    print("=================================")

    with open(result_path, "w", encoding="utf-8") as f:
        f.write(result_text)
    print(f"\n분석 결과가 '{result_path}'에 저장되었습니다.")

finally:
    # 8. 보안 및 용량 관리를 위해 Google 서버에서 파일 삭제
    if video_file:
        try:
            client.files.delete(name=video_file.name)
            print("\nGoogle AI Studio 서버에서 동영상 파일이 안전하게 삭제되었습니다.")
        except Exception as e:
            print(f"\n파일 삭제 중 오류 발생: {e}")
