import os
import json
import copy

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

app = FastAPI(title="Vision Inference API")

# =========================
# JSON 결과 로드
# =========================
with open(
    "./results/시연용/risk_results(test).json",
    "r",
    encoding="utf-8"
) as f:
    DEMO_RESULTS = json.load(f)

# =========================
# 영상 분석 API
# =========================
@app.post("/analyze/vision")
async def analyze_vision(
    file: UploadFile = File(...)
):
    try:
        # -------------------------
        # 파일 이름 추출
        # -------------------------
        filename = os.path.splitext(
            file.filename
        )[0].strip()

        # -------------------------
        # JSON 결과 탐색
        # -------------------------
        matched_result = None

        for item in DEMO_RESULTS:
            if item["video"] == filename:
                # 원본 데이터를 훼손하지 않기 위해 깊은 복사
                matched_result = copy.deepcopy(item)
                break

        # -------------------------
        # 결과 반환 (성공)
        # -------------------------
        if matched_result:
            # 💡 [핵심] 스프링부트 자바 DTO가 에러(500)나지 않도록 데이터 규격을 포맷팅합니다.
            # 1. 배열 형태인 events를 "손 반복 행동, 문 조작 행동" 형태의 하나의 문자열로 합치기
            if isinstance(matched_result.get("events"), list):
                matched_result["events"] = ", ".join(matched_result["events"])

            # 2. 자바 및 프론트엔드 변수명 규칙(CamelCase) 안전장치용 데이터 복사 세팅
            matched_result["riskScore"] = matched_result.get("risk_score", 0)
            matched_result["riskLevel"] = matched_result.get("risk_level", "NORMAL")

            return JSONResponse(
                status_code=200,
                content={
                    "status": "success",
                    "result": matched_result
                }
            )

        # -------------------------
        # 없는 영상 (예외 처리)
        # -------------------------
        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "result": {
                    "video": filename,
                    "events": "분석 결과 없음",
                    "risk_score": 0,
                    "riskScore": 0,
                    "risk_level": "NORMAL",
                    "riskLevel": "NORMAL"
                }
            }
        )

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": str(e)
            }
        )

# =========================
# 서버 실행
# =========================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000
    )