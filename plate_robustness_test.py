# -*- coding: utf-8 -*-
r"""
번호판 거리·왜곡 한계 시험 (합성, 카메라 불필요)

한국 긴 번호판(520 x 110 mm)을 단 차량 뒷모습을 3D 공간에 놓고, 이 카메라의 렌즈 값(1920x1080 기준 초점거리
FX px)으로 1920x1080 화면에 투영한다. 거리 / 좌우 각도(yaw) / 위아래 각도(pitch) / 기울기(roll) / 움직임 흐림을 바꿔 가며
    감지   : YOLO 가 번호판을 찾았는가 (정답 박스와 IoU >= 0.5)
    인식   : 찾은 번호판에서 번호를 정확히 읽었는가 (감지 + 글자, 실제 프로그램과 같음)
    OCR만  : 정답 위치를 직접 잘라 주면 OCR 이 읽는가 (감지기 탓인지 OCR 탓인지 가르기)
를 센다. 프레임 한 장 기준 (실제 프로그램은 5번 읽어 다수결이라 이보다 잘 됨).

    python plate_robustness_test.py                 # 전체 (약 2~4분)
    python plate_robustness_test.py --fx 1536       # 렌즈 초점거리(px, 1920 폭 기준)
결과: results\plate_robustness_<시각>\ (summary.md, results.csv, 예시 이미지)

주의: 합성 장면 + 맑은 고딕 글자(실제 번호판 서체 아님) + 렌즈 휨 없음 → 실제 성능과 차이가 있다.
     거리는 FX 가 맞을 때만 맞다 (보정 파일의 FX 1536 은 영상 너비 x 0.8 로 둔 값). 번호판 폭(px) 도 함께 적는다.
"""
import argparse
import csv
import math
import os
import sys
import time
from datetime import datetime

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import paths  # noqa: E402
from app.models import plate_models as PM  # noqa: E402

W, H = 1920, 1080
PLATE_MM = (520, 110)
CAR_MM = (1800, 1100)                 # 차량 뒷면 (폭, 높이)
PLATE_POS_MM = (640, 760)             # 차량 뒷면 안에서 번호판 왼쪽 위
PX_PER_MM = 2                         # 원본 그림 해상도
PLATES = ["12가3456", "34나5678", "123다4567", "567로8901"]
FONT = r"C:\Windows\Fonts\malgunbd.ttf"


# ---------------------------------------------------------------- 그리기
def plate_image(text):
    w, h = PLATE_MM[0] * PX_PER_MM, PLATE_MM[1] * PX_PER_MM
    img = Image.new("RGB", (w, h), (245, 245, 245))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((3, 3, w - 4, h - 4), radius=18, outline=(25, 25, 25), width=6)
    shown = text[:-4] + " " + text[-4:]                      # 12가 3456
    f = ImageFont.truetype(FONT, int(h * 0.74))
    tw = d.textlength(shown, font=f)
    if tw > w * 0.9:                                         # 칸에 맞게 줄임
        f = ImageFont.truetype(FONT, int(h * 0.74 * w * 0.9 / tw))
        tw = d.textlength(shown, font=f)
    box = d.textbbox((0, 0), shown, font=f)
    d.text(((w - tw) / 2, (h - (box[3] - box[1])) / 2 - box[1]), shown, font=f, fill=(15, 15, 15))
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def car_image(text, rng):
    cw, ch = CAR_MM[0] * PX_PER_MM, CAR_MM[1] * PX_PER_MM
    color = [int(v) for v in rng.choice([(40, 40, 45), (200, 200, 205), (120, 30, 30), (30, 60, 120), (90, 90, 95)])]
    car = np.zeros((ch, cw, 3), np.uint8)
    cv2.rectangle(car, (0, 0), (cw, ch), color, -1)
    cv2.rectangle(car, (60, 120), (cw - 60, 700), [max(0, c - 25) for c in color], -1)    # 뒷유리 자리
    cv2.rectangle(car, (60, 120), (cw - 60, 700), (30, 30, 30), 8)
    for x in (80, cw - 380):                                                                 # 후미등
        cv2.rectangle(car, (x, 820), (x + 300, 980), (30, 30, 170), -1)
    cv2.rectangle(car, (0, ch - 260), (cw, ch), (35, 35, 35), -1)                          # 범퍼
    p = plate_image(text)
    x0, y0 = PLATE_POS_MM[0] * PX_PER_MM, PLATE_POS_MM[1] * PX_PER_MM
    car[y0:y0 + p.shape[0], x0:x0 + p.shape[1]] = p
    return car


def background(rng):
    g = np.linspace(150, 70, H, dtype=np.float32)[:, None, None]
    bg = np.repeat(np.repeat(g, W, 1), 3, 2) + rng.normal(0, 6, (H, W, 3))
    return np.clip(bg, 0, 255).astype(np.uint8)


def rotation(yaw, pitch, roll):
    y, p, r = (math.radians(v) for v in (yaw, pitch, roll))
    Ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
    Rx = np.array([[1, 0, 0], [0, math.cos(p), -math.sin(p)], [0, math.sin(p), math.cos(p)]])
    Rz = np.array([[math.cos(r), -math.sin(r), 0], [math.sin(r), math.cos(r), 0], [0, 0, 1]])
    return Rz @ Rx @ Ry


def render(text, dist_m, yaw, pitch, roll, blur, fx, rng):
    """차량을 dist_m 거리에 두고 투영 → (장면, 정답 번호판 박스 xyxy, 번호판 폭 px)"""
    car = car_image(text, rng)
    K = np.array([[fx, 0, W / 2], [0, fx, H / 2], [0, 0, 1]])
    R = rotation(yaw, pitch, roll)
    # 번호판 중심이 화면 가운데 오도록 차량 좌표(mm, 번호판 중심 원점)를 잡는다
    pcx = (PLATE_POS_MM[0] + PLATE_MM[0] / 2) * PX_PER_MM
    pcy = (PLATE_POS_MM[1] + PLATE_MM[1] / 2) * PX_PER_MM

    def project(pts_px):
        mm = (np.asarray(pts_px, float) - [pcx, pcy]) / PX_PER_MM
        X = np.c_[mm, np.zeros(len(mm))] @ R.T + [0, 0, dist_m * 1000]
        uv = X @ K.T
        return uv[:, :2] / uv[:, 2:3]

    ch, cw = car.shape[:2]
    src = np.float32([[0, 0], [cw, 0], [cw, ch], [0, ch]])
    dst = project(src).astype(np.float32)
    # 멀리 있으면 먼저 줄여서 (축소 계단 현상 줄이기) 투영
    plate_w = np.linalg.norm(np.diff(project([[pcx - PLATE_MM[0], pcy], [pcx + PLATE_MM[0], pcy]]), axis=0))
    s = min(1.0, max(plate_w * 2, 8) / (PLATE_MM[0] * PX_PER_MM))
    if s < 1.0:
        car = cv2.resize(car, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        src = src * s
    Hm = cv2.getPerspectiveTransform(src.astype(np.float32), dst)
    scene = background(rng)
    warped = cv2.warpPerspective(car, Hm, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    mask = cv2.warpPerspective(np.full(car.shape[:2], 255, np.uint8), Hm, (W, H))
    scene[mask > 0] = warped[mask > 0]
    if blur > 0:                                             # 가로 움직임 흐림
        k = np.zeros((blur, blur), np.float32)
        k[blur // 2, :] = 1.0 / blur
        scene = cv2.filter2D(scene, -1, k)
    scene = np.clip(scene.astype(np.float32) + rng.normal(0, 3, scene.shape), 0, 255).astype(np.uint8)
    ok, enc = cv2.imencode(".jpg", scene, [cv2.IMWRITE_JPEG_QUALITY, 85])
    scene = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    x0, y0 = PLATE_POS_MM[0] * PX_PER_MM, PLATE_POS_MM[1] * PX_PER_MM
    corners = project([[x0, y0], [x0 + PLATE_MM[0] * PX_PER_MM, y0],
                       [x0 + PLATE_MM[0] * PX_PER_MM, y0 + PLATE_MM[1] * PX_PER_MM], [x0, y0 + PLATE_MM[1] * PX_PER_MM]])
    gt = [corners[:, 0].min(), corners[:, 1].min(), corners[:, 0].max(), corners[:, 1].max()]
    return scene, gt, float(np.linalg.norm(corners[1] - corners[0]))


def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


# ---------------------------------------------------------------- 시험
def main():
    ap = argparse.ArgumentParser(description="번호판 거리·왜곡 한계 시험 (합성)")
    ap.add_argument("--fx", type=float, default=1536.0, help="초점거리 px (1920 폭 기준, 보정 파일 값)")
    ap.add_argument("--detector", default="YOLO-v1s")
    ap.add_argument("--reader", default="EOCR_KR")
    ap.add_argument("--imgsz", type=int, default=None, help="YOLO 입력 크기 (생략하면 앱과 같은 plate_models.YOLO_IMGSZ)")
    a = ap.parse_args()
    if a.imgsz:
        PM.YOLO_IMGSZ = a.imgsz                              # 이 시험 안에서만 바꿈 (앱 설정은 그대로)
    out = os.path.join(paths.ROOT, "results",
                       f"plate_robustness_{datetime.now():%Y%m%d_%H%M%S}_imgsz{PM.YOLO_IMGSZ}")
    os.makedirs(out, exist_ok=True)
    print("모델 불러오는 중...", flush=True)
    detect = PM.DETECTORS[a.detector]["build"]("cuda:0")
    read = PM.READERS[a.reader]["build"]("cuda:0")
    rng = np.random.default_rng(0)
    rows = []

    def run(sweep, dist, yaw=0, pitch=10, roll=0, blur=0, save=False):
        for text in PLATES:
            scene, gt, pw = render(text, dist, yaw, pitch, roll, blur, a.fx, rng)
            xyxy, conf = detect(scene)
            best = max(range(len(conf)), key=lambda i: iou(xyxy[i], gt), default=None)
            det = best is not None and iou(xyxy[best], gt) >= 0.5
            got = read([PM.crop_plate(scene, xyxy[best])])[0][0] if det else ""
            ocr_gt = read([PM.crop_plate(scene, gt)])[0][0]
            rows.append({"sweep": sweep, "distance_m": dist, "yaw": yaw, "pitch": pitch, "roll": roll, "blur": blur,
                         "plate": text, "plate_px": round(pw, 1), "detected": int(det),
                         "det_conf": round(float(conf[best]), 2) if det else "", "read": got,
                         "correct": int(got == text), "ocr_only": ocr_gt, "ocr_only_correct": int(ocr_gt == text)})
            if save and text == PLATES[0]:
                vis = scene.copy()
                cv2.rectangle(vis, (int(gt[0]), int(gt[1])), (int(gt[2]), int(gt[3])), (0, 255, 0), 2)
                if det:
                    b = [int(v) for v in xyxy[best]]
                    cv2.rectangle(vis, (b[0], b[1]), (b[2], b[3]), (255, 120, 0), 2)
                cv2.imwrite(os.path.join(out, f"{sweep}_d{dist}_y{yaw}_p{pitch}_r{roll}_b{blur}.jpg"), vis)

    t0 = time.time()
    dists = [2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25]
    for d in dists:                                          # A. 거리 x 좌우 각도
        for y in (0, 15, 30, 45, 60):
            run("A_dist_yaw", d, yaw=y, save=y in (0, 45) and d in (5, 10, 15))
    for p in (0, 15, 30, 45, 60):                            # B. 위아래 각도 (5m)
        run("B_pitch", 5, pitch=p, save=p in (30, 60))
    for r in (0, 5, 10, 15, 20, 30):                         # C. 기울기 (5m)
        run("C_roll", 5, roll=r, save=r in (15, 30))
    for b in (0, 5, 10, 15, 25):                             # D. 움직임 흐림 (5m / 10m)
        run("D_blur", 5, blur=b, save=b in (15, 25))
        run("D_blur", 10, blur=b)
    print(f"시험 {len(rows)}건, {time.time() - t0:.0f}초", flush=True)

    with open(os.path.join(out, "results.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    def rate(rs, k):
        return f"{100 * sum(r[k] for r in rs) / len(rs):.0f}%" if rs else "-"

    lines = [f"# 번호판 거리·왜곡 한계 시험 ({datetime.now():%Y-%m-%d %H:%M})", "",
             f"- 감지 {a.detector} (입력 {PM.YOLO_IMGSZ}), 글자 {a.reader}, 초점거리 fx {a.fx:.0f} px (1920x1080), 번호판 520x110 mm, "
             f"번호판 {len(PLATES)}개 x 조건, 프레임 한 장 기준", "- 칸 = 감지 / 인식(감지+정확한 번호) / OCR만(정답 위치를 잘라 줌)", ""]
    A = [r for r in rows if r["sweep"] == "A_dist_yaw"]
    lines += ["## A. 거리 x 좌우 각도 (위에서 10도 내려다봄)", "",
              "| 거리 | 번호판 폭 | " + " | ".join(f"yaw {y}°" for y in (0, 15, 30, 45, 60)) + " |",
              "| --- | --- | " + " | ".join("---" for _ in range(5)) + " |"]
    for d in dists:
        rs = [r for r in A if r["distance_m"] == d]
        px = np.mean([r["plate_px"] for r in rs if r["yaw"] == 0])
        cells = [f"{rate(s, 'detected')} / {rate(s, 'correct')} / {rate(s, 'ocr_only_correct')}"
                 for s in ([r for r in rs if r["yaw"] == y] for y in (0, 15, 30, 45, 60))]
        lines.append(f"| {d} m | {px:.0f} px | " + " | ".join(cells) + " |")
    for key, title, col, unit in (("B_pitch", "B. 위아래 각도 (5 m)", "pitch", "°"), ("C_roll", "C. 기울기 (5 m)", "roll", "°"),
                                  ("D_blur", "D. 가로 움직임 흐림", "blur", " px")):
        rs = [r for r in rows if r["sweep"] == key]
        lines += ["", f"## {title}", "", "| 조건 | 거리 | 감지 | 인식 | OCR만 |", "| --- | --- | --- | --- | --- |"]
        for v in sorted({r[col] for r in rs}):
            for d in sorted({r["distance_m"] for r in rs}):
                s = [r for r in rs if r[col] == v and r["distance_m"] == d]
                if s:
                    lines.append(f"| {col} {v}{unit} | {d} m | {rate(s, 'detected')} | {rate(s, 'correct')} | {rate(s, 'ocr_only_correct')} |")
    wrong = [r for r in rows if r["detected"] and not r["correct"]][:15]
    if wrong:
        lines += ["", "## 잘못 읽은 예 (감지는 됨)", "", "| 조건 | 정답 | 읽은 것 |", "| --- | --- | --- |"]
        lines += [f"| {r['sweep']} d{r['distance_m']} y{r['yaw']} p{r['pitch']} r{r['roll']} b{r['blur']} | {r['plate']} | "
                  f"{r['read'] or '(못 읽음)'} |" for r in wrong]
    with open(os.path.join(out, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n결과: {out}")


if __name__ == "__main__":
    main()
