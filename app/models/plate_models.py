# -*- coding: utf-8 -*-
"""
plate_models (Model) — 번호판 감지기·글자 인식기를 등록하는 곳

인식 설정 창의 [번호판 감지] / [글자 인식] 목록은 여기 등록된 것만 보여 준다.

감지기 (DETECTORS)
    키: {"name": 표시 이름, "build": build(device) → detect}
    detect(frame_bgr) → (xyxy, conf)      xyxy: (N, 4) float 원본 좌표 [x1, y1, x2, y2], conf: (N,) float

글자 인식 (READERS)
    키: {"name": 표시 이름, "build": build(device) → read}
    read([번호판 잘라낸 BGR 이미지, ...]) → [(글자, 신뢰도), ...]    예: ("12가3456", 0.93). 못 읽으면 ("", 0.0)

등록된 모델 (2026-11-20)
    감지: models\\plate_yolo\\*.pt 마다 하나씩 (YOLO, ultralytics)
          지금 있는 것: license-plate-finetune-v1s.pt = YOLO11s 번호판 감지
          (Hugging Face morsetechlab/yolov11-license-plate-detection, AGPL-3.0, 여러 나라 번호판으로 학습)
    글자: EasyOCR 한글+영어 (가중치 %USERPROFILE%\\.EasyOCR\\model 의 korean_g2.pth, craft_mlt_25k.pth)
          EOCR_KR  = 번호판 글자만 허용 + 한국 번호판 형식에 맞을 때만 읽은 것으로 침 (기본)
          EOCR_RAW = 읽은 글자를 그대로 (형식 검사 없음, 다른 나라 번호판·시험용)

가중치는 models\\<이름>\\ 에 둔다 (paths.MODELS_DIR, 저장소에는 올리지 않음).
"""
import glob
import os
import re

import cv2
import numpy as np

from app.paths import MODELS_DIR

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


# =====================================================================
# 감지기: YOLO (ultralytics) — models\plate_yolo\*.pt
# =====================================================================
YOLO_DIR = os.path.join(MODELS_DIR, "plate_yolo")
YOLO_CONF = 0.25          # 감지 신뢰도 기준
YOLO_IMGSZ = 640


def _yolo_builder(path):
    def build(device):
        from ultralytics import YOLO
        model = YOLO(path, task="detect")
        dev = 0 if str(device).startswith("cuda") else "cpu"
        precision = "fp16" if dev == 0 else "fp32"     # ultralytics 8.4: half 대신 quantize

        def detect(frame):
            r = model.predict(frame, conf=YOLO_CONF, imgsz=YOLO_IMGSZ, device=dev, quantize=precision, verbose=False)[0]
            b = r.boxes
            return b.xyxy.cpu().numpy().astype(float), b.conf.cpu().numpy().astype(float)
        return detect
    return build


def _yolo_key(path):
    """license-plate-finetune-v1s.pt → YOLO-v1s"""
    stem = os.path.splitext(os.path.basename(path))[0]
    return "YOLO-" + stem.replace("license-plate-finetune-", "")


for _p in sorted(glob.glob(os.path.join(YOLO_DIR, "*.pt"))):
    DETECTORS[_yolo_key(_p)] = {"name": f"YOLO 번호판 ({os.path.basename(_p)})", "build": _yolo_builder(_p)}


# =====================================================================
# 글자 인식: EasyOCR (한글 + 영어)
# =====================================================================
# 한국 번호판에 쓰이는 한글: 용도 글자 + 옛 지역 표시(서울12가3456) + 영업용·군용 등
PLATE_HANGUL = ("가나다라마거너더러머버서어저고노도로모보소오조구누두루무부수우주아바사자배하허호"
                "육해공국합외교영준기협임정"
                "서울부산대구인천광주전울세종경기강원충북남제")
PLATE_ALLOW = "0123456789" + "".join(dict.fromkeys(PLATE_HANGUL))
REGIONS = "서울|부산|대구|인천|광주|대전|울산|세종|경기|강원|충북|충남|전북|전남|경북|경남|제주"
PLATE_RE = re.compile(rf"^(?:{REGIONS})?\d{{2,3}}[가-힣]\d{{4}}$")     # 12가3456 / 123가4567 / 서울12가3456
OCR_MIN_HEIGHT = 96       # 잘라낸 번호판이 이보다 작으면 키워서 읽음 (작은 글자 인식률 개선)


def _prepare(crop):
    h, w = crop.shape[:2]
    if h < OCR_MIN_HEIGHT and h > 0:
        s = OCR_MIN_HEIGHT / h
        crop = cv2.resize(crop, (int(w * s), OCR_MIN_HEIGHT), interpolation=cv2.INTER_CUBIC)
    return crop


def join_lines(results):
    """EasyOCR 결과 [(박스, 글자, 신뢰도)] → 위 줄부터, 줄 안에서는 왼쪽부터 이어 붙인 글자와 평균 신뢰도
    (두 줄 번호판: 위 '서울12가', 아래 '3456')"""
    if not results:
        return "", 0.0
    items = []
    for box, text, conf in results:
        pts = np.asarray(box, float)
        items.append((pts[:, 1].mean(), pts[:, 0].min(), pts[:, 1].max() - pts[:, 1].min(), text, float(conf)))
    items.sort(key=lambda t: t[0])
    lines, cur = [], [items[0]]
    for it in items[1:]:
        if abs(it[0] - cur[-1][0]) < 0.5 * max(it[2], cur[-1][2]):   # 세로 중심이 가까우면 같은 줄
            cur.append(it)
        else:
            lines.append(cur)
            cur = [it]
    lines.append(cur)
    text = "".join(t[3] for line in lines for t in sorted(line, key=lambda t: t[1]))
    return text, float(np.mean([t[4] for t in items]))


def normalize_plate(text):
    return re.sub(r"[^0-9가-힣]", "", text)


def _easyocr_builder(strict):
    def build(device):
        import easyocr
        reader = easyocr.Reader(["ko", "en"], gpu=str(device).startswith("cuda"), verbose=False)

        def read(crops):
            out = []
            for crop in crops:
                if crop.size == 0:
                    out.append(("", 0.0))
                    continue
                res = reader.readtext(_prepare(crop), detail=1, paragraph=False,
                                      allowlist=PLATE_ALLOW if strict else None)
                text, conf = join_lines(res)
                if strict:
                    text = normalize_plate(text)
                    if not PLATE_RE.match(text):
                        text, conf = "", 0.0              # 번호판 형식이 아니면 '못 읽음' (다수결에 넣지 않음)
                else:
                    text = text.replace(" ", "")
                out.append((text, conf))
            return out
        return read
    return build


READERS["EOCR_KR"] = {"name": "EasyOCR 한국 번호판 (형식 검사)", "build": _easyocr_builder(strict=True)}
READERS["EOCR_RAW"] = {"name": "EasyOCR (형식 검사 없음)", "build": _easyocr_builder(strict=False)}
