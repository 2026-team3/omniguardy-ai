import cv2

# =========================
# 영상 경로
# =========================
video_path = "./videos/test/시연용/사용자뒤접근.mp4"

# =========================
# ROI 좌표
# (직접 조절)
# =========================
ROI_X1 = 150
ROI_Y1 = 100

ROI_X2 = 1350
ROI_Y2 = 1250

# =========================
# 영상 열기
# =========================
cap = cv2.VideoCapture(video_path)

# 첫 프레임 읽기
ret, frame = cap.read()

if not ret:
    print("영상 읽기 실패")
    exit()

# =========================
# ROI 사각형 그리기
# =========================
cv2.rectangle(
    frame,
    (ROI_X1, ROI_Y1),
    (ROI_X2, ROI_Y2),
    (0, 0, 255),   # 빨간색
    3
)

# 텍스트 표시
cv2.putText(
    frame,
    "ROI AREA",
    (ROI_X1, ROI_Y1 - 10),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (0, 0, 255),
    2
)

# =========================
# 화면 출력
# =========================
cv2.imshow("ROI Check", frame)

cv2.waitKey(0)

cap.release()
cv2.destroyAllWindows()