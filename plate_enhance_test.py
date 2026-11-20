# -*- coding: utf-8 -*-
"""
번호판 보정·선명한 프레임 고르기 시험 (합성, plate_robustness_test 의 장면을 씀)

A. 한 장 인식: 흐림·어두움·저대비·잡음 조건 x 보정(없음 / 대비 / 선명화 / 둘 다) → 감지+정확히 읽은 수
B. 흐림 단계별 선명도 (선명도 값이 흐림을 잘 구분하는지)
C. 번호판 하나를 여러 프레임 (흐림이 프레임마다 다름) → 고정된 번호가 맞는지
   모든 프레임 읽기 + 통째 다수결 / 자리별 다수결 / 흐린 프레임 건너뛰기 + 선명도 가중 다수결

    python plate_enhance_test.py [--detector YOLO-v1s] [--reader EOCR_KR]
결과: results\\plate_enhance_<시각>\\ (summary.md, single.csv, sequence.csv)
"""
import argparse
import collections
import csv
import os
import sys
import time
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import plate_robustness_test as R  # noqa: E402
from app import paths  # noqa: E402
from app.models import plate_enhance as PE  # noqa: E402
from app.models import plate_models as PM  # noqa: E402
from app.models import tracker as TR  # noqa: E402

VARIANTS = {"없음": (False, False), "대비": (True, False), "선명화": (False, True), "대비+선명화": (True, True)}
LIGHT = {
    "보통": lambda s, rng: s,
    "어두움": lambda s, rng: np.clip(s * 0.3 + rng.normal(0, 4, s.shape), 0, 255).astype(np.uint8),
    "저대비": lambda s, rng: np.clip(128 + (s.astype(np.float32) - 128) * 0.3, 0, 255).astype(np.uint8),
    "잡음": lambda s, rng: np.clip(s + rng.normal(0, 14, s.shape), 0, 255).astype(np.uint8),
}


def old_vote(preds):
    c = collections.Counter(t for t, *_ in preds)
    s = collections.defaultdict(float)
    for t, v, *_ in preds:
        s[t] += v
    return max(c, key=lambda t: (c[t], s[t]))


def main():
    ap = argparse.ArgumentParser(description="번호판 보정·선명한 프레임 고르기 시험 (합성)")
    ap.add_argument("--detector", default=PM.DEFAULT_DETECTOR)
    ap.add_argument("--reader", default="EOCR_KR")
    a = ap.parse_args()
    out = os.path.join(paths.ROOT, "results", f"plate_enhance_{datetime.now():%Y%m%d_%H%M%S}")
    os.makedirs(out, exist_ok=True)
    print("모델 불러오는 중...", flush=True)
    detect = PM.DETECTORS[a.detector]["build"]("cuda:0")
    read = PM.READERS[a.reader]["build"]("cuda:0")
    rng = np.random.default_rng(0)

    def scene_crop(text, dist, blur, light, yaw=0):
        scene, gt, _ = R.render(text, dist, yaw, 10, 0, blur, 1536.0, rng)
        scene = LIGHT[light](scene, rng)
        xyxy, conf = detect(scene)
        best = max(range(len(conf)), key=lambda i: R.iou(xyxy[i], gt), default=None)
        if best is None or R.iou(xyxy[best], gt) < 0.5:
            return None
        return PM.crop_plate(scene, xyxy[best])

    # ---------------- A. 한 장 인식 ----------------
    single = []
    times = {"sharp": [], "enh": []}
    t0 = time.time()
    for light in LIGHT:
        for dist in (5, 8, 10):
            for blur in (0, 5, 10, 15):
                for text in R.PLATES:
                    crop = scene_crop(text, dist, blur, light)
                    t = time.perf_counter()
                    sh = PE.sharpness(crop) if crop is not None else 0.0
                    times["sharp"].append(time.perf_counter() - t)
                    row = {"light": light, "distance_m": dist, "blur": blur, "plate": text,
                           "detected": int(crop is not None), "sharpness": round(sh, 1)}
                    for vn, (c, s) in VARIANTS.items():
                        if crop is None:
                            row[vn] = ""
                            continue
                        t = time.perf_counter()
                        img = PE.enhance(crop, c, s)
                        times["enh"].append(time.perf_counter() - t)
                        row[vn] = read([img])[0][0]
                    single.append(row)
        print(f"  A {light} 끝 ({time.time() - t0:.0f}초)", flush=True)
    with open(os.path.join(out, "single.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(single[0]))
        w.writeheader()
        w.writerows(single)

    # ---------------- C. 여러 프레임 ----------------
    seq_rows = []
    blur_pool = [0, 0, 5, 10, 15, 20, 25, 25]
    for rep in range(3):
        for dist in (5, 8, 10):
            for text in R.PLATES:
                blurs = list(rng.permutation(blur_pool))
                frames = []
                for b in blurs:
                    crop = scene_crop(text, dist, int(b), "보통")
                    if crop is None:
                        continue
                    frames.append((int(b), PE.sharpness(crop), read([crop])[0]))
                res = {"rep": rep, "distance_m": dist, "plate": text, "blurs": " ".join(map(str, blurs)),
                       "frames_detected": len(frames)}
                # 모든 프레임: 읽은 글자가 lock_frames 개 모이면 고정 (빈 글자는 세지 않음)
                preds = [(t, c) for _, _, (t, c) in frames if t][:TR.LOCK_FRAMES]
                res["all_whole"] = old_vote(preds) if preds else ""
                res["all_pos"] = TR.vote_text(preds)[0] if preds else ""
                # 흐린 프레임 건너뛰기 + 선명도 가중
                best, sel = 0.0, []
                for b, sh, (t, c) in frames:
                    best = max(best, sh)
                    if sh < best * PE.SKIP_RATIO or not t:
                        continue
                    sel.append((t, c, sh))
                    if len(sel) >= TR.LOCK_FRAMES:
                        break
                res["select_w"] = TR.vote_text(sel)[0] if sel else ""
                res["select_n"] = len(sel)
                seq_rows.append(res)
    with open(os.path.join(out, "sequence.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(seq_rows[0]))
        w.writeheader()
        w.writerows(seq_rows)

    # ---------------- 요약 ----------------
    L = [f"# 번호판 보정·선명한 프레임 고르기 시험 ({datetime.now():%Y-%m-%d %H:%M})", "",
         f"- 감지 {a.detector} (입력 {PM.YOLO_IMGSZ}), 글자 {a.reader}. 합성 번호판 4개 (`{'`, `'.join(R.PLATES)}`), 정면, 위에서 10°",
         f"- 처리 시간 (번호판 1장): 선명도 {np.mean(times['sharp']) * 1000:.2f} ms, "
         f"보정 {np.mean(times['enh']) * 1000:.2f} ms", "",
         "## A. 한 장 인식 (감지 + 정확히 읽은 수)", "",
         "| 조명 | 흐림 px | 감지 | " + " | ".join(VARIANTS) + " |", "| --- | --- | --- | " + " | ".join("---" for _ in VARIANTS) + " |"]
    for light in LIGHT:
        for blur in (0, 5, 10, 15):
            rs = [r for r in single if r["light"] == light and r["blur"] == blur]
            cells = [str(sum(r[v] == r["plate"] for r in rs)) for v in VARIANTS]
            L.append(f"| {light} | {blur} | {sum(r['detected'] for r in rs)}/{len(rs)} | " + " | ".join(cells) + " |")
    tot = {v: sum(r[v] == r["plate"] for r in single) for v in VARIANTS}
    wrong = {v: sum(r[v] not in ("", r["plate"]) for r in single) for v in VARIANTS}
    L.append(f"| **합계** | | {sum(r['detected'] for r in single)}/{len(single)} | "
             + " | ".join(f"**{tot[v]}** (오답 {wrong[v]})" for v in VARIANTS) + " |")
    L += ["", "## B. 흐림 단계별 선명도 (보통 조명, 감지된 것 평균)", "", "| 흐림 px | 5 m | 8 m | 10 m |", "| --- | --- | --- | --- |"]
    for blur in (0, 5, 10, 15):
        cells = []
        for dist in (5, 8, 10):
            v = [r["sharpness"] for r in single if r["light"] == "보통" and r["blur"] == blur
                 and r["distance_m"] == dist and r["detected"]]
            cells.append(f"{np.mean(v):.0f}" if v else "-")
        L.append(f"| {blur} | " + " | ".join(cells) + " |")
    L += ["", f"## C. 여러 프레임 (흐림 {blur_pool} 를 섞은 8프레임, {TR.LOCK_FRAMES}번 읽으면 고정, {len(seq_rows)}경우)", "",
          "| 방식 | 맞게 고정 | 틀린 번호로 고정 | 고정 못함 |", "| --- | --- | --- | --- |"]
    for k, name in (("all_whole", "모든 프레임 + 통째 다수결 (이전)"), ("all_pos", "모든 프레임 + 자리별 다수결"),
                    ("select_w", "흐린 프레임 건너뜀 + 선명도 가중 자리별 다수결")):
        ok = sum(r[k] == r["plate"] for r in seq_rows)
        bad = sum(r[k] not in ("", r["plate"]) for r in seq_rows)
        L.append(f"| {name} | {ok} | {bad} | {len(seq_rows) - ok - bad} |")
    with open(os.path.join(out, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\n결과: {out}")


if __name__ == "__main__":
    main()
