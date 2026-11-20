# -*- coding: utf-8 -*-
"""
고정 후 추적 유지 시험 (합성 장면, 앱과 같은 RecognitionPipeline.process 를 그대로 씀)

장면: 1920x1080, 차 2대가 반대 방향으로 지나감 (번호판 폭 약 200 px)
    차 A (위 줄, 오른쪽으로): 기둥 뒤를 지나감 → 짧게 가려짐
    차 B (아래 줄, 왼쪽으로): 큰 가림막 뒤를 지나감 → 길게 가려짐 (추적이 끝났다가 다시 나타남)
    감지기 결과를 일부러 흔듦: 프레임마다 25% 는 감지를 빼고, 25% 는 신뢰도를 반으로 (실제 카메라의 놓침·약한 감지 흉내)

설정 조합(이전 = 모두 끔 / 기다림·영상 따라가기 / + 낮은 감지 / + 번호로 다시 찾기)마다
    차마다 붙은 화면 ID 수 (1 이 이상적), 보이는 동안 박스가 표시된 비율, 글자 인식 횟수, 고정된 번호를 셈

    python tracking_hold_test.py [--frames 260] [--save-video]
결과: results\\tracking_hold_<시각>\\ (summary.md, frames.csv, 고르면 영상)
"""
import argparse
import csv
import os
import queue
import sys
import time
from datetime import datetime

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import plate_robustness_test as R  # noqa: E402
from app import paths  # noqa: E402
from app.models import plate_models as PM  # noqa: E402
from app.models import tracker as TR  # noqa: E402
from app.models.pipeline import RecognitionPipeline  # noqa: E402

W, H = 1920, 1080
SCALE = 0.2                                   # 차 그림 축소 (번호판 1040 px → 208 px)
CARS = [  # 이름, 번호, 시작 x, y, 속도 px/프레임
    ("A", R.PLATES[0], -760, 180, 11),
    ("B", R.PLATES[2], W + 40, 600, -11),
]
POLE = (900, 0, 1000, 560)                    # 차 A 를 짧게 가리는 기둥 (x1, y1, x2, y2)
WALL = (500, 560, 1250, 1080)                 # 차 B 를 길게 가리는 가림막
CONFIGS = {
    "이전 (모두 끔)": dict(hold_track=False, low_conf=False, reid=False),
    "기다림+영상 따라가기": dict(hold_track=True, low_conf=False, reid=False),
    "+ 낮은 감지": dict(hold_track=True, low_conf=True, reid=False),
    "+ 번호로 다시 찾기 (기본)": dict(hold_track=True, low_conf=True, reid=True),
}


def make_frames(n, rng):
    bg = R.background(rng)
    cars = []
    for name, text, x0, y0, v in CARS:
        car = cv2.resize(R.car_image(text, rng), None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)
        px, py = R.PLATE_POS_MM[0] * R.PX_PER_MM * SCALE, R.PLATE_POS_MM[1] * R.PX_PER_MM * SCALE
        pw, ph = R.PLATE_MM[0] * R.PX_PER_MM * SCALE, R.PLATE_MM[1] * R.PX_PER_MM * SCALE
        cars.append((name, text, x0, y0, v, car, (px, py, pw, ph)))
    frames, truth = [], []
    for fi in range(n):
        img = bg.copy()
        gts = {}
        for name, text, x0, y0, v, car, (px, py, pw, ph) in cars:
            x = int(x0 + v * fi)
            ch, cw = car.shape[:2]
            sx1, sx2 = max(0, -x), min(cw, W - x)
            if sx2 > sx1:
                img[y0:y0 + ch, x + sx1:x + sx2] = car[:, sx1:sx2]
            gts[name] = (x + px, y0 + py, x + px + pw, y0 + py + ph)
        cv2.rectangle(img, POLE[:2], POLE[2:], (90, 95, 100), -1)
        cv2.rectangle(img, WALL[:2], WALL[2:], (70, 75, 80), -1)
        img = np.clip(img + rng.normal(0, 3, img.shape), 0, 255).astype(np.uint8)
        frames.append(img)
        # 번호판이 화면 안에 다 있고 가려지지 않았을 때만 "보임"
        vis = {}
        for name, g in gts.items():
            inside = g[0] >= 0 and g[2] <= W
            hidden = any(not (g[2] <= o[0] or g[0] >= o[2] or g[3] <= o[1] or g[1] >= o[3]) for o in (POLE, WALL))
            vis[name] = (g, inside and not hidden)
        truth.append(vis)
    return frames, truth


def main():
    ap = argparse.ArgumentParser(description="고정 후 추적 유지 시험 (합성)")
    ap.add_argument("--frames", type=int, default=260)
    ap.add_argument("--detector", default=PM.DEFAULT_DETECTOR)
    ap.add_argument("--reader", default="EOCR_KR")
    ap.add_argument("--save-video", action="store_true")
    a = ap.parse_args()
    out = os.path.join(paths.ROOT, "results", f"tracking_hold_{datetime.now():%Y%m%d_%H%M%S}")
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(0)
    print("장면 만드는 중...", flush=True)
    frames, truth = make_frames(a.frames, rng)
    print("모델 불러오는 중...", flush=True)
    real_detect = PM.DETECTORS[a.detector]["build"]("cuda:0")
    read = PM.READERS[a.reader]["build"]("cuda:0")
    # 감지 결과를 한 번만 구해 두고 (낮은 기준), 같은 흔들기를 모든 조합에 똑같이 씀
    cache = [real_detect(f, conf=PM.LOW_CONF) for f in frames]
    shake = rng.random((a.frames, 8))

    def fake_detect_for(fi_ref):
        def detect(frame, conf=None):
            xyxy, cf = cache[fi_ref[0]]
            cf = cf.copy()
            keep = []
            for k in range(len(cf)):
                r = shake[fi_ref[0], k % 8]
                if r < 0.25:
                    continue                       # 감지 놓침
                if r < 0.5:
                    cf[k] *= 0.5                   # 약한 감지
                keep.append(k)
            thr = PM.YOLO_CONF if conf is None else conf
            keep = [k for k in keep if cf[k] >= thr]
            return xyxy[keep].reshape(-1, 4), cf[keep]
        return detect

    rows_out, summary = [], {}
    for cname, opts in CONFIGS.items():
        q = queue.Queue()
        pipe = RecognitionPipeline(q, lambda: (None, None))
        fi_ref = [0]
        pipe.detect, pipe.read = fake_detect_for(fi_ref), read
        pipe._det_has_conf = True
        pipe.det_key, pipe.read_key = a.detector, a.reader
        pipe.W, pipe.H = W, H
        pipe.reset_stats()
        for k, v in opts.items():
            setattr(pipe, k, v)
        TR.Track._next = 1
        reads = 0
        orig_read = pipe.read

        def counting_read(crops, _r=orig_read):
            nonlocal reads
            reads += len(crops)
            return _r(crops)
        pipe.read = counting_read
        ids = {n: [] for n, *_ in CARS}
        shown = {n: [0, 0] for n, *_ in CARS}       # 보이는 프레임 중 박스 표시 수, 보이는 프레임 수
        texts = {n: set() for n, *_ in CARS}
        writer = None
        if a.save_video and cname.endswith("(기본)"):
            writer = cv2.VideoWriter(os.path.join(out, "default.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30, (W // 2, H // 2))
        t0 = time.perf_counter()
        for fi, frame in enumerate(frames):
            fi_ref[0] = fi
            pipe.process(frame, fi, save=False)
            msg = None
            while not q.empty():
                k, d = q.get()
                if k == "frame":
                    msg = d
            rows = msg["rows"] if msg else []
            if writer is not None:
                writer.write(cv2.resize(msg["img"], (W // 2, H // 2)))
            for name, (g, visible) in truth[fi].items():
                best = max(rows, key=lambda r: TR.iou(r["box"], g), default=None)
                hit = best is not None and TR.iou(best["box"], g) >= 0.3 and best["state"] != "entering"
                if visible:
                    shown[name][1] += 1
                    shown[name][0] += int(hit)
                if hit:
                    if best["no"] is not None and (not ids[name] or ids[name][-1] != best["no"]):
                        ids[name].append(best["no"])
                    if best["text"]:
                        texts[name].add(best["text"])
                rows_out.append({"config": cname, "frame": fi, "car": name, "visible": int(visible), "shown": int(hit),
                                 "no": best["no"] if hit else "", "state": best["state"] if hit else "",
                                 "text": (best["text"] or "") if hit else "", "bridged": int(bool(hit and best.get("bridged")))})
        if writer is not None:
            writer.release()
        ms = (time.perf_counter() - t0) * 1000 / len(frames)
        summary[cname] = {"ids": ids, "shown": shown, "texts": texts, "reads": reads, "ms": ms}
        print(f"  {cname}: ID {ids}, 고정 {texts}, 읽기 {reads}회, {ms:.1f} ms/프레임", flush=True)

    with open(os.path.join(out, "frames.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0]))
        w.writeheader()
        w.writerows(rows_out)
    L = [f"# 고정 후 추적 유지 시험 ({datetime.now():%Y-%m-%d %H:%M})", "",
         f"- 합성 {a.frames} 프레임 1920x1080, 감지 {a.detector} (입력 {PM.YOLO_IMGSZ}), 글자 {a.reader}",
         f"- 차 A `{CARS[0][1]}` 기둥 뒤 짧게 가려짐, 차 B `{CARS[1][1]}` 가림막 뒤 길게 가려짐. "
         "감지 25% 빼고 25% 신뢰도 반 (모든 조합에 같게)",
         "- 화면 ID 들 = 그 차에 차례로 붙은 화면 ID (1개가 이상적). 표시 = 번호판이 보이는 프레임 중 박스가 그려진 비율", "",
         "| 설정 | 차 A 화면 ID | 차 A 표시 | 차 B 화면 ID | 차 B 표시 | 고정된 번호 | 글자 인식 횟수 | 처리 ms/프레임 |",
         "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for cname, s in summary.items():
        cells = []
        for n in ("A", "B"):
            sh = s["shown"][n]
            cells += [" → ".join(map(str, s["ids"][n])) or "-", f"{sh[0]}/{sh[1]} ({100 * sh[0] / max(sh[1], 1):.0f}%)"]
        tx = ", ".join(f"{n}: {'/'.join(sorted(s['texts'][n])) or '-'}" for n in ("A", "B"))
        L.append(f"| {cname} | " + " | ".join(cells) + f" | {tx} | {s['reads']} | {s['ms']:.1f} |")
    with open(os.path.join(out, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\n결과: {out}")


if __name__ == "__main__":
    main()
