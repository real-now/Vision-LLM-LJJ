"""
[Vision 프로세스]  vision_on_request.py

카메라에서 person과 주변 객체를 YOLO11n(INT8 Engine)으로 계속 탐지하다가,
음성 프로세스가 request.json을 갱신하면 그 시점의 ROI를 잘라 저장한다.

    (음성 질문 감지) -> request.json -> ROI 캡처 -> roi.jpg + roi_meta.json

    l : 수동 캡처 (마이크 없이 테스트할 때)
    q : 종료

실행:
    source .venv/bin/activate
    python vision_on_request.py
"""

import json
import os

import cv2
from ultralytics import YOLO

# ============================ 설정 ============================

YOLO_MODEL_PATH = "src/models/YOLO/yolo11n_int8.engine"

REQUEST_PATH = "src/output/request.json"  # 음성 프로세스 -> Vision

ROI_IMAGE_PATH = "src/output/roi.jpg"  # Vision -> LLM
TEMP_ROI_IMAGE_PATH = "src/output/roi_tmp.jpg"
ROI_META_PATH = "src/output/roi_meta.json"
TEMP_ROI_META_PATH = "src/output/roi_meta_tmp.json"

PERSON_CLASS = 0
CONF_TH = 0.25
IOU_TH = 0.5

INSIDE_RATIO_TH = 0.55
MAX_AREA_RATIO = 0.6
HAND_BAND = 0.55

EXCLUDE_CLASSES = {
    "chair", "couch", "bed", "dining table", "toilet",
    "tv", "bench", "refrigerator", "oven", "sink",
}

ROI_PADDING = 20
MIN_ROI_SIZE = 320

FLIP_VERTICAL = True

PIPELINE = (
    "nvarguscamerasrc sensor-id=0 ! "
    "video/x-raw(memory:NVMM), "
    "width=1280, height=720, framerate=30/1 ! "
    "nvvidconv ! "
    "video/x-raw, format=BGRx ! "
    "videoconvert ! "
    "video/x-raw, format=BGR ! "
    "queue leaky=downstream max-size-buffers=1 ! "
    "appsink drop=true max-buffers=1 sync=false"
)


# ======================= 보조 함수 =======================


def box_area(box):
    x1, y1, x2, y2 = box
    return max(x2 - x1, 0) * max(y2 - y1, 0)


def intersection_area(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)

    return max(x2 - x1, 0) * max(y2 - y1, 0)


def find_held_object(person, objects):
    """점수 = 사람 안에 들어간 비율 x confidence x 손 높이 근접도"""
    px1, py1, px2, py2 = person["bbox"]
    person_area = box_area(person["bbox"])
    person_height = max(py2 - py1, 1)

    hand_y = py1 + person_height * HAND_BAND

    best = None
    best_score = 0.0

    for obj in objects:
        if obj["class"] == "person":
            continue
        if obj["class"] in EXCLUDE_CLASSES:
            continue

        obj_area = box_area(obj["bbox"])

        if obj_area <= 0:
            continue
        if obj_area > person_area * MAX_AREA_RATIO:
            continue

        inside_ratio = intersection_area(person["bbox"], obj["bbox"]) / obj_area

        if inside_ratio < INSIDE_RATIO_TH:
            continue

        ox1, oy1, ox2, oy2 = obj["bbox"]
        obj_cy = (oy1 + oy2) / 2
        band_weight = 1.0 - min(abs(obj_cy - hand_y) / person_height, 1.0)

        score = inside_ratio * obj["confidence"] * (0.3 + 0.7 * band_weight)

        if score > best_score:
            best_score = score
            best = obj

    return best, best_score


def crop_roi(image, box, padding=ROI_PADDING):
    height, width = image.shape[:2]

    x1, y1, x2, y2 = box

    x1 = max(0, int(x1) - padding)
    y1 = max(0, int(y1) - padding)
    x2 = min(width, int(x2) + padding)
    y2 = min(height, int(y2) + padding)

    roi = image[y1:y2, x1:x2].copy()

    if roi.size == 0:
        return None

    roi_height, roi_width = roi.shape[:2]
    longest = max(roi_height, roi_width)

    if longest < MIN_ROI_SIZE:
        scale = MIN_ROI_SIZE / longest
        roi = cv2.resize(
            roi,
            (int(roi_width * scale), int(roi_height * scale)),
            interpolation=cv2.INTER_CUBIC,
        )

    return roi


def save_roi(roi_image, meta):
    """이미지를 먼저, 메타 JSON을 나중에 저장한다. (LLM은 메타만 감시)"""
    if not cv2.imwrite(TEMP_ROI_IMAGE_PATH, roi_image):
        raise RuntimeError("ROI 이미지 저장에 실패했습니다.")

    os.replace(TEMP_ROI_IMAGE_PATH, ROI_IMAGE_PATH)

    with open(TEMP_ROI_META_PATH, "w", encoding="utf-8") as file:
        json.dump(meta, file, ensure_ascii=False, indent=4)

    os.replace(TEMP_ROI_META_PATH, ROI_META_PATH)


def capture_roi(frame, person, held, score, question):
    """요청 시점의 프레임에서 ROI를 만들어 저장한다."""
    if person is None:
        # 사람이 없어도 LLM 쪽이 무한정 기다리지 않도록 응답은 보낸다.
        save_roi(
            frame,
            {
                "mode": "none",
                "hint": None,
                "question": question,
                "score": 0.0,
            },
        )
        print("\n[ROI] 사람이 탐지되지 않았습니다.")
        return

    if held:
        roi_image = crop_roi(frame, held["bbox"])
        meta = {
            "mode": "object",
            "hint": held["class"],
            "question": question,
            "score": round(score, 3),
        }
        print(f"\n[ROI] YOLO 후보: {held['class']} (score {score:.2f})")
    else:
        roi_image = crop_roi(frame, person["bbox"], padding=10)
        meta = {
            "mode": "person",
            "hint": None,
            "question": question,
            "score": 0.0,
        }
        print("\n[ROI] YOLO 후보 없음 -> 사람 전체를 Gemma에게 전달")

    if roi_image is None:
        print("[ROI] ROI를 만들 수 없습니다.")
        return

    save_roi(roi_image, meta)
    print(f"[ROI] 저장 완료: {ROI_IMAGE_PATH}")


# ============================ main ============================

os.makedirs("src/output", exist_ok=True)

yolo = YOLO(YOLO_MODEL_PATH)

cap = cv2.VideoCapture(PIPELINE, cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("카메라를 열 수 없습니다.")
    exit()

print("q : 종료")
print("l : 수동 캡처")
print("음성 프로세스의 요청 대기 중...\n")

# 음성 프로세스가 보낸 요청의 수정 시각
if os.path.exists(REQUEST_PATH):
    last_request = os.path.getmtime(REQUEST_PATH)
else:
    last_request = 0

while True:
    ret, frame = cap.read()
    key = cv2.waitKey(1) & 0xFF

    if not ret:
        break
    if key == ord("q"):
        break

    if FLIP_VERTICAL:
        frame = cv2.flip(frame, 0)

    # 1. 전체 클래스 탐지
    results = yolo.predict(
        source=frame,
        conf=CONF_TH,
        iou=IOU_TH,
        verbose=False,
    )

    result = results[0]

    # 2. Dictionary 변환
    objects = []

    for box in result.boxes:
        class_id = int(box.cls[0].item())
        x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()

        objects.append(
            {
                "class": result.names[class_id],
                "confidence": float(box.conf[0].item()),
                "bbox": [x1, y1, x2, y2],
            }
        )

    # 3. 기준 인물 + 들고 있는 물건
    persons = [obj for obj in objects if obj["class"] == "person"]
    persons.sort(key=lambda o: box_area(o["bbox"]), reverse=True)

    person = persons[0] if persons else None
    held, score = find_held_object(person, objects) if person else (None, 0.0)

    # 4. 화면 출력
    output_frame = result.plot()

    if held:
        hx1, hy1, hx2, hy2 = [int(v) for v in held["bbox"]]
        cv2.rectangle(output_frame, (hx1, hy1), (hx2, hy2), (0, 0, 255), 3)
        cv2.putText(
            output_frame,
            f"HELD? {held['class']} ({score:.2f})",
            (hx1, max(hy1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )

    cv2.imshow("Held Object", output_frame)

    # 5-a. 음성 요청 감지
    if os.path.exists(REQUEST_PATH):
        current_request = os.path.getmtime(REQUEST_PATH)

        if current_request != last_request:
            last_request = current_request

            with open(REQUEST_PATH, "r", encoding="utf-8") as file:
                request = json.load(file)

            question = request.get("question", "")
            print(f"\n[요청] {question}")

            capture_roi(frame, person, held, score, question)

    # 5-b. 수동 캡처 (마이크 없이 테스트)
    if key == ord("l"):
        capture_roi(frame, person, held, score, "이게 뭐야?")


cap.release()
cv2.destroyAllWindows()
print("\nVision 프로세스를 종료했습니다.")
