# -*- coding: utf-8 -*-
"""
plate_models (Model) — 번호판 감지기·글자 인식기를 등록하는 곳

인식 설정 창의 [감지기] / [글자 인식] 목록은 여기 등록된 것만 보여 준다. 아직 등록된 모델은 없다.

감지기 (DETECTORS)
    키: {"name": 표시 이름, "build": build(device) → detect}
    detect(frame_bgr) → (xyxy, conf)      xyxy: (N, 4) float 원본 좌표 [x1, y1, x2, y2], conf: (N,) float

글자 인식 (READERS)
    키: {"name": 표시 이름, "build": build(device) → read}
    read([번호판 잘라낸 BGR 이미지, ...]) → [(글자, 신뢰도), ...]    예: ("12가3456", 0.93). 못 읽으면 ("", 0.0)

등록 예 (모델을 만든 뒤 이 파일 아래에 추가):
    def _build_my_detector(device):
        from ultralytics import YOLO
        model = YOLO(os.path.join(MODELS_DIR, "plate_det", "best.pt"))
        def detect(frame):
            r = model.predict(frame, conf=0.25, device=device, verbose=False)[0]
            return r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()
        return detect
    DETECTORS["P1"] = {"name": "YOLO 번호판", "build": _build_my_detector}

가중치는 models\\<이름>\\ 에 둔다 (paths.MODELS_DIR, 저장소에는 올리지 않음).
"""
import os  # noqa: F401  (등록 예에서 사용)

from app.paths import MODELS_DIR  # noqa: F401

DETECTORS = {}
READERS = {}

PAD_RATIO = 0.08          # 번호판을 잘라낼 때 박스 둘레에 더하는 여백 (박스 크기 대비)


def crop_plate(frame, box, pad=PAD_RATIO):
    """번호판 박스 + 여백을 잘라냄 (화면 밖은 잘림)"""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = (float(v) for v in box)
    px, py = (x2 - x1) * pad, (y2 - y1) * pad
    x1, y1 = max(0, int(x1 - px)), max(0, int(y1 - py))
    x2, y2 = min(w, int(x2 + px)), min(h, int(y2 + py))
    return frame[y1:y2, x1:x2].copy()


def describe(registry, key):
    return f"{key} {registry[key]['name']}" if key in registry else ""
