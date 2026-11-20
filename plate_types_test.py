# -*- coding: utf-8 -*-
r"""
한국 번호판 종류별 인식 시험 (합성, 카메라 불필요)

번호판 종류(일반 7·8자리, 구형 녹색, 영업용 택시·버스·택배, 렌터카, 전기차, 법인 연두색, 외교, 건설기계, 이륜차)를
실제 크기·색·배치로 그려 차량 뒷면에 달고, 이 카메라 렌즈 값(fx, 1920x1080)으로 투영한 장면에서
    감지   : YOLO 가 번호판을 찾았는가 (정답 박스와 IoU >= 0.5)
    인식   : 찾은 번호판을 글자 인식기로 정확히 읽었는가 (형식 검사 EOCR_KR / 검사 없음 EOCR_RAW)
    OCR만  : 정답 위치를 잘라 주면 읽는가
를 감지기 설정별로 센다 (프레임 한 장 기준).

    python plate_types_test.py
    python plate_types_test.py --extra-yolo D:\path\license-plate-finetune-v1m.pt ...   # 다른 YOLO 도 비교
결과: results\plate_types_<시각>\ (summary.md, results.csv, plates.jpg = 그린 번호판 모음, 장면 예시)

주의: 합성 장면 + 맑은 고딕(실제 번호판 서체 아님). 외교·건설기계 번호 배열은 자료로 확인하지 못해 추정.
     2026-03-20 이륜차 전국번호판, 군용은 번호 배열 자료가 없어 뺌.
"""
import argparse
import csv
import math
import os
import re
import sys
from datetime import datetime

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import paths  # noqa: E402
from app.models import plate_models as PM  # noqa: E402

W, H, FX_DEFAULT = 1920, 1080, 1536.0
FONT = r"C:\Windows\Fonts\malgunbd.ttf"
PX_PER_MM = 4

WHITE, BLACK = (245, 245, 245), (20, 20, 20)
# 이름, 크기 mm, 바탕, 글자, 줄 목록, 왼쪽 띠, 차종, 형식 확인 여부
TYPES = [
    ("일반 8자리 (2019~, 태극 띠)", (520, 110), WHITE, BLACK, ["123가 4567"], "kor", "car", True),
    ("일반 7자리 (2006~)", (520, 110), WHITE, BLACK, ["12가 3456"], None, "car", True),
    ("구형 녹색 두 줄 (지역)", (335, 170), (40, 120, 70), WHITE, ["서울 12", "가 3456"], None, "car", True),
    ("영업용 택시 (노랑, 지역)", (335, 170), (240, 200, 30), BLACK, ["서울 32", "바 1234"], None, "car", True),
    ("영업용 버스 (노랑, 지역)", (335, 170), (240, 200, 30), BLACK, ["경기 70", "아 5678"], None, "bus", True),
    ("영업용 택배 (노랑, 지역)", (335, 170), (240, 200, 30), BLACK, ["인천 85", "배 2468"], None, "truck", True),
    ("렌터카 (흰색, 하·허·호)", (520, 110), WHITE, BLACK, ["123허 4567"], "kor", "car", True),
    ("전기차 (하늘색)", (520, 110), (120, 190, 235), BLACK, ["123가 4567"], "ev", "car", True),
    ("법인 업무용 (연두색, 2024~)", (520, 110), (170, 220, 90), BLACK, ["123가 4567"], "kor", "car", True),
    ("외교 (감청색) — 형식 추정", (520, 110), (40, 50, 110), WHITE, ["외교 001-001"], None, "car", False),
    ("건설기계 영업용 (주황) — 형식 추정", (335, 170), (235, 120, 30), WHITE, ["서울 04", "가 1234"], None, "truck", False),
    ("이륜차 구형 (흰 바탕 파란 글자)", (210, 115), WHITE, (40, 60, 170), ["서울강남", "가 1234"], None, "moto", True),
]
VEHICLE_MM = {"car": (1800, 1100), "bus": (2500, 2400), "truck": (2300, 1800), "moto": (700, 900)}
VEHICLE_COLOR = {"car": (60, 60, 65), "bus": (60, 120, 60), "truck": (150, 150, 155), "moto": (30, 30, 30)}
CONDITIONS = [("3m 정면", 3, 0), ("5m 정면", 5, 0), ("8m 정면", 8, 0), ("5m 30° 비스듬히", 5, 30)]


def norm(t):
    return re.sub(r"[^0-9가-힣]", "", t)


# ---------------------------------------------------------------- 번호판 그리기
def draw_plate(size_mm, bg, fg, lines, strip):
    w, h = size_mm[0] * PX_PER_MM, size_mm[1] * PX_PER_MM
    img = Image.new("RGB", (w, h), bg[::-1] if False else bg)
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((4, 4, w - 5, h - 5), radius=int(h * 0.08), outline=fg if bg != WHITE else BLACK, width=8)
    x0 = 0
    if strip:                                             # 왼쪽 띠 (태극·KOR / 전기차 표시)
        sw = int(w * 0.08)
        cx, cy, r = sw // 2 + 10, int(h * 0.38), int(sw * 0.3)
        if strip == "kor":
            d.pieslice((cx - r, cy - r, cx + r, cy + r), 180, 360, fill=(200, 40, 50))
            d.pieslice((cx - r, cy - r, cx + r, cy + r), 0, 180, fill=(30, 70, 160))
            f = ImageFont.truetype(FONT, int(h * 0.16))
            d.text((cx - d.textlength("KOR", font=f) / 2, cy + r + 6), "KOR", font=f, fill=(30, 70, 160))
        else:
            d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(30, 90, 180))
            f = ImageFont.truetype(FONT, int(h * 0.16))
            d.text((cx - d.textlength("EV", font=f) / 2, cy + r + 6), "EV", font=f, fill=(30, 90, 180))
        x0 = sw + 10
    area_w = w - x0
    if len(lines) == 1:
        sizes, ys = [h * 0.72], [0.5]
    else:
        sizes, ys = [h * 0.36, h * 0.48], [0.27, 0.68]
    for text, size, yc in zip(lines, sizes, ys):
        f = ImageFont.truetype(FONT, int(size))
        tw = d.textlength(text, font=f)
        if tw > area_w * 0.9:
            f = ImageFont.truetype(FONT, int(size * area_w * 0.9 / tw))
            tw = d.textlength(text, font=f)
        b = d.textbbox((0, 0), text, font=f)
        d.text((x0 + (area_w - tw) / 2, yc * h - (b[3] + b[1]) / 2), text, font=f, fill=fg)
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def vehicle_with_plate(kind, plate, plate_mm):
    vw, vh = VEHICLE_MM[kind]
    cw, ch = vw * PX_PER_MM // 2, vh * PX_PER_MM // 2           # 차체는 절반 해상도 (2 px/mm)
    car = np.zeros((ch, cw, 3), np.uint8)
    car[:] = VEHICLE_COLOR[kind]
    cv2.rectangle(car, (40, 60), (cw - 40, int(ch * 0.5)), (35, 35, 40), -1)                # 유리·어두운 부분
    for x in (30, cw - 30 - int(cw * 0.15)):
        cv2.rectangle(car, (x, int(ch * 0.62)), (x + int(cw * 0.15), int(ch * 0.72)), (30, 30, 170), -1)
    pw, ph = plate_mm[0] * 2, plate_mm[1] * 2
    p = cv2.resize(plate, (pw, ph), interpolation=cv2.INTER_AREA)
    x0, y0 = (cw - pw) // 2, int(ch * 0.78) - ph // 2
    car[y0:y0 + ph, x0:x0 + pw] = p
    return car, (x0, y0, pw, ph)


# ---------------------------------------------------------------- 투영
def rotation(yaw, pitch):
    y, p = math.radians(yaw), math.radians(pitch)
    Ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
    Rx = np.array([[1, 0, 0], [0, math.cos(p), -math.sin(p)], [0, math.sin(p), math.cos(p)]])
    return Rx @ Ry


def render(car, plate_rect, dist_m, yaw, fx, rng, pitch=10):
    x0, y0, pw, ph = plate_rect
    pcx, pcy = x0 + pw / 2, y0 + ph / 2
    K = np.array([[fx, 0, W / 2], [0, fx, H / 2], [0, 0, 1]])
    R = rotation(yaw, pitch)

    def project(pts):
        mm = (np.asarray(pts, float) - [pcx, pcy]) / 2.0                          # 차체 그림 2 px/mm
        X = np.c_[mm, np.zeros(len(mm))] @ R.T + [0, 0, dist_m * 1000]
        uv = X @ K.T
        return uv[:, :2] / uv[:, 2:3]

    ch, cw = car.shape[:2]
    src = np.float32([[0, 0], [cw, 0], [cw, ch], [0, ch]])
    dst = project(src).astype(np.float32)
    corners = project([[x0, y0], [x0 + pw, y0], [x0 + pw, y0 + ph], [x0, y0 + ph]])
    plate_px = float(np.linalg.norm(corners[1] - corners[0]))
    s = min(1.0, max(plate_px * 2, 8) / pw)
    img = car
    if s < 1.0:
        img = cv2.resize(car, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        src = src * s
    Hm = cv2.getPerspectiveTransform(src.astype(np.float32), dst)
    g = np.linspace(150, 70, H, dtype=np.float32)[:, None, None]
    scene = np.clip(np.repeat(np.repeat(g, W, 1), 3, 2) + rng.normal(0, 6, (H, W, 3)), 0, 255).astype(np.uint8)
    warped = cv2.warpPerspective(img, Hm, (W, H))
    mask = cv2.warpPerspective(np.full(img.shape[:2], 255, np.uint8), Hm, (W, H))
    scene[mask > 0] = warped[mask > 0]
    scene = np.clip(scene.astype(np.float32) + rng.normal(0, 3, scene.shape), 0, 255).astype(np.uint8)
    scene = cv2.imdecode(cv2.imencode(".jpg", scene, [cv2.IMWRITE_JPEG_QUALITY, 85])[1], cv2.IMREAD_COLOR)
    gt = [corners[:, 0].min(), corners[:, 1].min(), corners[:, 0].max(), corners[:, 1].max()]
    return scene, gt, plate_px


def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


# ---------------------------------------------------------------- 시험
def main():
    ap = argparse.ArgumentParser(description="한국 번호판 종류별 인식 시험 (합성)")
    ap.add_argument("--fx", type=float, default=FX_DEFAULT)
    ap.add_argument("--extra-yolo", nargs="*", default=[], help="함께 비교할 YOLO .pt (입력 1280)")
    a = ap.parse_args()
    out = os.path.join(paths.ROOT, "results", f"plate_types_{datetime.now():%Y%m%d_%H%M%S}")
    os.makedirs(out, exist_ok=True)

    from ultralytics import YOLO
    base = os.path.join(paths.MODELS_DIR, "plate_yolo", "license-plate-finetune-v1s.pt")
    detectors = [("v1s@640 (지금 앱)", base, 640), ("v1s@1280", base, 1280)]
    for p in a.extra_yolo:
        stem = os.path.splitext(os.path.basename(p))[0].replace("license-plate-finetune-", "")
        detectors.append((f"{stem}@1280", p, 1280))
    models = {p: YOLO(p, task="detect") for _, p, _ in detectors}
    readers = {k: PM.READERS[k]["build"]("cuda:0") for k in ("EOCR_KR", "EOCR_RAW")}

    rng = np.random.default_rng(0)
    plates = []
    rows = []
    for name, size, bg, fg, lines, strip, kind, confirmed in TYPES:
        plate = draw_plate(size, bg, fg, lines, strip)
        plates.append((name, plate))
        truth = norm("".join(lines))
        car, rect = vehicle_with_plate(kind, plate, size)
        for cond, dist, yaw in CONDITIONS:
            scene, gt, px = render(car, rect, dist, yaw, a.fx, rng)
            ocr_gt = {k: norm(r([PM.crop_plate(scene, gt)])[0][0]) for k, r in readers.items()}
            if cond == "5m 정면":
                cv2.imwrite(os.path.join(out, f"scene_{len(plates):02d}.jpg"), scene)
            for dname, p, imgsz in detectors:
                res = models[p].predict(scene, conf=0.25, imgsz=imgsz, device=0, verbose=False)[0]
                boxes, confs = res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy()
                best = max(range(len(confs)), key=lambda i: iou(boxes[i], gt), default=None)
                det = best is not None and iou(boxes[best], gt) >= 0.5
                got = {k: norm(r([PM.crop_plate(scene, boxes[best])])[0][0]) if det else "" for k, r in readers.items()}
                rows.append({"type": name, "format_confirmed": int(confirmed), "condition": cond, "plate_px": round(px),
                             "detector": dname, "detected": int(det),
                             "det_conf": round(float(confs[best]), 2) if det else "",
                             "truth": truth, "read_KR": got["EOCR_KR"], "ok_KR": int(got["EOCR_KR"] == truth),
                             "read_RAW": got["EOCR_RAW"], "ok_RAW": int(got["EOCR_RAW"] == truth),
                             "ocr_only_KR": ocr_gt["EOCR_KR"], "ocr_only_ok_KR": int(ocr_gt["EOCR_KR"] == truth),
                             "ocr_only_RAW": ocr_gt["EOCR_RAW"], "ocr_only_ok_RAW": int(ocr_gt["EOCR_RAW"] == truth)})
        print(f"  {name} 끝", flush=True)

    # 번호판 모음 그림
    tiles = []
    for name, p in plates:
        t = cv2.resize(p, (int(p.shape[1] * 160 / p.shape[0]), 160))
        canvas = np.full((200, 900, 3), 255, np.uint8)
        canvas[20:180, 10:10 + min(t.shape[1], 880)] = t[:, :880]
        tiles.append(canvas)
    cv2.imwrite(os.path.join(out, "plates.jpg"), np.vstack(tiles))

    with open(os.path.join(out, "results.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    dnames = [d[0] for d in detectors]
    conds = [c[0] for c in CONDITIONS]
    L = [f"# 한국 번호판 종류별 인식 시험 ({datetime.now():%Y-%m-%d %H:%M})", "",
         f"- 합성 장면 1920x1080, fx {a.fx:.0f}, 위에서 10° 내려다봄, 프레임 한 장 기준. 조건: {', '.join(conds)}",
         "- 감지 = YOLO conf 0.25, 정답 박스 IoU >= 0.5. 인식 = 감지 + 정확한 번호 (공백·기호 무시)", ""]
    L += ["## 1. 감지 (조건 4개 중 찾은 수)", "", "| 종류 | " + " | ".join(dnames) + " |",
          "| --- | " + " | ".join("---" for _ in dnames) + " |"]
    for name, *_ in TYPES:
        cells = []
        for dn in dnames:
            rs = [r for r in rows if r["type"] == name and r["detector"] == dn]
            cells.append(f"{sum(r['detected'] for r in rs)}/{len(rs)}")
        L.append(f"| {name} | " + " | ".join(cells) + " |")
    L += ["", "## 2. 인식 (감지 + 정확한 번호, 조건 4개 중) — 형식 검사 KR / 검사 없음 RAW", "",
          "| 종류 | " + " | ".join(dnames) + " | OCR만 KR / RAW |", "| --- | " + " | ".join("---" for _ in dnames) + " | --- |"]
    for name, *_ in TYPES:
        cells = []
        for dn in dnames:
            rs = [r for r in rows if r["type"] == name and r["detector"] == dn]
            cells.append(f"{sum(r['ok_KR'] for r in rs)} / {sum(r['ok_RAW'] for r in rs)}")
        rs = [r for r in rows if r["type"] == name and r["detector"] == dnames[0]]
        L.append(f"| {name} | " + " | ".join(cells) +
                 f" | {sum(r['ocr_only_ok_KR'] for r in rs)} / {sum(r['ocr_only_ok_RAW'] for r in rs)} |")
    L += ["", "## 3. OCR 이 읽은 글자 (정답 위치를 잘라 줌, 5m 정면)", "", "| 종류 | 정답 | KR | RAW |", "| --- | --- | --- | --- |"]
    for name, *_ in TYPES:
        r = next(r for r in rows if r["type"] == name and r["condition"] == "5m 정면")
        L.append(f"| {name} | {r['truth']} | {r['ocr_only_KR'] or '(못 읽음)'} | {r['ocr_only_RAW'] or '(못 읽음)'} |")
    with open(os.path.join(out, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\n결과: {out}")


if __name__ == "__main__":
    main()
