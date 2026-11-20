# -*- coding: utf-8 -*-
"""
plate_models (Model) — 번호판 감지기·글자 인식기를 등록하는 곳

인식 설정 창의 [번호판 감지] / [글자 인식] 목록은 여기 등록된 것만 보여 준다.

감지기 (DETECTORS)
    키: {"name": 표시 이름, "build": build(device) → detect}
    detect(frame_bgr) → (xyxy, conf)      xyxy: (N, 4) float 원본 좌표 [x1, y1, x2, y2], conf: (N,) float
    detect(frame_bgr, conf=0.1)           (선택) 신뢰도 기준을 낮춰 감지 — 있으면 "낮은 감지로 추적 잇기" 를 씀

글자 인식 (READERS)
    키: {"name": 표시 이름, "build": build(device) → read}
    read([번호판 잘라낸 BGR 이미지, ...]) → [(글자, 신뢰도), ...]    예: ("12가3456", 0.93). 못 읽으면 ("", 0.0)

등록된 모델 (2026-11-20)
    감지: models\\plate_yolo\\*.pt 마다 하나씩 (YOLO, ultralytics, 입력 1280)
          지금 있는 것: license-plate-finetune-v1s.pt = YOLO-v1s (기본), license-plate-finetune-v1l.pt = YOLO-v1l
          (Hugging Face morsetechlab/yolov11-license-plate-detection, AGPL-3.0, 여러 나라 번호판으로 학습)
    글자: EasyOCR 한글+영어 (가중치 %USERPROFILE%\\.EasyOCR\\model 의 korean_g2.pth, craft_mlt_25k.pth)
          EOCR_KR  = 번호판 글자만 허용 + 한국 번호판 형식에 맞을 때만 읽은 것으로 침 (기본)
                     형식: 12가3456, 123가4567, 서울12가3456, 외교001001, 서울강남가1234 (PLATE_FORMATS)
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
LOW_CONF = 0.1            # 이미 추적 중인 번호판을 이어 갈 때만 쓰는 낮은 기준 (새 번호판은 YOLO_CONF 이상만)
# 감지 입력 크기. 1920x1080 을 640 으로 줄이면 번호판이 너무 작아져 한국 번호판을 거의 못 찾음
# (2026-11-20 합성 시험: 12종 x 4조건 감지 640 → 8/48, 1280 → 34/48 (v1s), 정면 3 m → 6 m, 시간 8.4 → 10.9 ms)
YOLO_IMGSZ = 1280
DEFAULT_DETECTOR = "YOLO-v1s"   # 기본 감지기 (v1l 등 다른 크기는 인식 설정 창에서 고름)


def _yolo_builder(path):
    def build(device):
        from ultralytics import YOLO
        model = YOLO(path, task="detect")
        dev = 0 if str(device).startswith("cuda") else "cpu"
        precision = "fp16" if dev == 0 else "fp32"     # ultralytics 8.4: half 대신 quantize

        def detect(frame, conf=None):
            r = model.predict(frame, conf=YOLO_CONF if conf is None else conf, imgsz=YOLO_IMGSZ, device=dev, quantize=precision, verbose=False)[0]
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
REGIONS = "서울|부산|대구|인천|광주|대전|울산|세종|경기|강원|충북|충남|전북|전남|경북|경남|제주"
# 외교·영사 등 (한글 2자 + 숫자 3-3, 예: 외교 001-001)
DIPLOMATIC = "외교|영사|준외|준영|국기|협정|대표"
# 구형 이륜차 위 줄의 시·군·구 (예: 서울강남 가 1234). 특별·광역시 구·군 + 시 (구·시·군 글자는 번호판에 쓰지 않음)
DISTRICTS = (
    "종로|중|용산|성동|광진|동대문|중랑|성북|강북|도봉|노원|은평|서대문|마포|양천|강서|구로|금천|영등포|동작|관악|서초|강남|송파|강동|"
    "서|동|영도|부산진|동래|남|북|해운대|사하|금정|연제|수영|사상|기장|수성|달서|달성|군위|"
    "미추홀|연수|남동|부평|계양|강화|옹진|광산|유성|대덕|울주|"
    "수원|성남|고양|용인|부천|안산|안양|남양주|화성|평택|의정부|시흥|파주|김포|광명|광주|군포|하남|오산|이천|안성|의왕|"
    "양주|포천|여주|동두천|과천|구리|가평|양평|연천|"
    "춘천|원주|강릉|동해|태백|속초|삼척|청주|충주|제천|천안|공주|보령|아산|서산|논산|계룡|당진|"
    "전주|군산|익산|정읍|남원|김제|목포|여수|순천|나주|광양|포항|경주|김천|안동|구미|영주|영천|상주|문경|경산|"
    "창원|진주|통영|사천|김해|밀양|거제|양산|제주|서귀포")
PLATE_FORMATS = [
    rf"\d{{2,3}}[가-힣]\d{{4}}",                     # 12가3456 / 123가4567
    rf"(?:{REGIONS})\d{{2}}[가-힣]\d{{4}}",           # 서울12가3456 (영업용·구형, 지역 뒤는 2자리뿐)
    rf"(?:{DIPLOMATIC})\d{{6}}",                     # 외교001001
    rf"(?:{REGIONS})(?:{DISTRICTS})[가-힣]\d{{4}}",   # 서울강남가1234 (구형 이륜차)
]
PLATE_RE = re.compile("^(?:" + "|".join(PLATE_FORMATS) + ")$")
_FORMAT_HANGUL = re.sub(r"[^가-힣]", "", REGIONS + DIPLOMATIC + DISTRICTS)
PLATE_ALLOW = "0123456789" + "".join(dict.fromkeys(PLATE_HANGUL + _FORMAT_HANGUL))
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


_REGION_LIST = REGIONS.split("|")


def fix_region(text):
    """앞 두 글자가 지역명과 한 글자만 다르고 그런 지역이 하나뿐이면 그 지역으로 고침 (예: 서문12가3456 → 서울12가3456).
    형식 검사에 맞는 글자는 그대로 둠"""
    if PLATE_RE.match(text) or len(text) < 3 or not re.match(r"^[가-힣]{2}", text):
        return text
    head = text[:2]
    near = [r for r in _REGION_LIST if sum(a != b for a, b in zip(r, head)) == 1]
    if len(near) == 1 and PLATE_RE.match(near[0] + text[2:]):
        return near[0] + text[2:]
    return text


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
                    text = fix_region(normalize_plate(text))
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
