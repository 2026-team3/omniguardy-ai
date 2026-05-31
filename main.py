import os
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

app = FastAPI(title="Vision Mock API")

DEMO_DATABASE={
    "노크": {
        "video": "노크",
        "events": "비정상 상체 방향",
        "risk_score": 15,
        "risk_level": "NORMAL"
    },
    "뒤로지나감":{
        "video": "뒤로지나감",
        "events": "일반 이동, 손 반복 행동, 문 조작 행동, 비정상 상체 방향",
        "risk_score": 43,
        "risk_level": "LOW"
    },
    "비번틀림반복": {
        "video": "비번틀림반복",
        "events": "손 반복 행동, 비정상 상체 방향, 수상한 체류",
        "risk_score": 50,
        "risk_level": "MIDDLE"
    },
    "사용자뒤접근": {
        "video": "사용자뒤접근",
        "events": "손 반복 행동, 사용자 뒤 접근",
        "risk_score": 90,
        "risk_level": "HIGH"
    },
    "정상(문열고 들어감)": {
        "video": "정상(문열고 들어감)",
        "events": "일반 이동",
        "risk_score": 3,
        "risk_level": "NORMAL"
    },
    "카메라가림": {
        "video": "카메라가림",
        "events": "문 조작 행동, 비정상 상체 방향, 카메라 가림",
        "risk_score": 50,
        "risk_level": "MIDDLE"
    },
    "택배배달": {
        "video": "택배배달",
        "events": "일반 이동, 비정상 상체 방향",
        "risk_score": 18,
        "risk_level": "NORMAL"
    },
    "택배배달2": {
        "video": "택배배달2",
        "events": "일반 이동, 비정상 상체 방향",
        "risk_score": 18,
        "risk_level": "NORMAL"
    }
}

@app.post("/analyze/vision")
async def analyze_vision(file: UploadFile = File(...)):
    try:
        # 1. 파일 이름에서 확장자(.mp4, .mov 등) 제거하고 순수 이름만 추출
        filename_attr = os.path.splitext(file.filename)[0]
        
        # 공백 제거 및 혹시 모를 매핑 매칭 확률 높이기 키 보정
        clean_key = filename_attr.strip()
        
        # 특정 단어가 파일명에 포함되어 있으면 매핑되도록 유연하게 처리
        matched_result = None
        for key in DEMO_DATABASE.keys():
            if key in clean_key:
                matched_result = DEMO_DATABASE[key]
                break

        # 2. 매칭되는 시연 영상 데이터가 있을 경우 바로 JSON 리턴
        if matched_result:
            return JSONResponse(status_code=200, content=matched_result)
        
        # 3. 예외 상황: 데이터베이스에 없는 새 영상이 업로드되었을 때 디폴트 기본값 반환
        else:
            default_result = {
                "video": file.filename,
                "events": "미등록 시연 영상 (분석 대기)",
                "risk_score": 0,
                "risk_level": "NORMAL"
            }
            return JSONResponse(status_code=200, content=default_result)

    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)