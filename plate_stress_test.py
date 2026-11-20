# -*- coding: utf-8 -*-
"""
번호판 종합 시험 (합성 영상, 앱과 같은 RecognitionPipeline.process 로 감지 → 추적 → 글자 인식 → 고정)

    번호판 : 한국 번호판 12종 (plate_types_test 와 같은 그림)
    움직임 : 가로 지나감 / 반대 방향 / 다가옴 / 멀어짐 / 대각선 / 회전하며 들어와 멈춤 / 멈췄다 출발 + 카메라 흔들림 / 두 대 교차
    왜곡   : 원근 (좌우·위아래 각도, 기울기), 움직이는 방향으로 번지는 모션블러 (속도 x 노출), 초점 흐림,
             조명 (보통 / 어두움 / 역광 / 저대비), 잡음, JPEG 압축 — 장면마다 섞어서 정함 (시드 고정)
    설정   : 이전 (화질·추적 유지 모두 끔) / 기본 (지금 앱 기본) / 기본+보정 (대비·선명화도 켬)

번호판마다 잼: 맞게 고정했는지, 틀린 번호로 고정한 적이 있는지, 화면 ID 가 몇 개 붙었는지 (1 이 이상적),
보이는 동안 박스가 표시된 비율, 다 보인 뒤 고정까지 걸린 프레임 수

    python plate_stress_test.py [--frames 120] [--types 0,1,..] [--moves ..] [--configs ..]
결과: results\\plate_stress_<시각>\\ (summary.md, runs.csv, samples\\*.jpg)
"""
import argparse
import csv
import math
import os
import queue
import sys
import time
from collections import defaultdict
from datetime import datetime

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import plate_types_test as PT  # noqa: E402
from app import paths  # noqa: E402
from app.models import plate_models as PM  # noqa: E402
from app.models import tracker as TR  # noqa: E402
from app.models.pipeline import RecognitionPipeline  # noqa: E402

W, H, FX = 1920, 1080, 1536.0
CONFIGS = {
    "이전": dict(sharp_select=False, hold_track=False, low_conf=False, reid=False,
               enhance_contrast=False, enhance_sharpen=False),
    "기본": dict(sharp_select=True, hold_track=True, low_conf=True, reid=True,
               enhance_contrast=False, enhance_sharpen=False),
    "기본+보정": dict(sharp_select=True, hold_track=True, low_conf=True, reid=True,
                  enhance_contrast=True, enhance_sharpen=True),
}
LIGHTS = ["보통", "어두움", "역광", "저대비"]


# ---------------------------------------------------------------- 움직임 (t: 0~1 → 차마다 u, v, 거리 m, yaw, pitch, roll)
def _smooth(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def move_cross_lr(t):
    return [(-500 + t * (W + 1000), 600, 5.5, 0, 10, 0)]


def move_cross_rl(t):
    return [(W + 500 - t * (W + 1000), 520, 7.0, 12, 8, 3)]


def move_approach(t):
    return [(W / 2 + 220 * math.sin(t * 3), 420 + 220 * t, 15 - 11.5 * t, -15, 12, 0)]


def move_recede(t):
    return [(W / 2 - 300, 680 - 180 * t, 3.5 + 11.5 * t, 20, 10, 0)]


def move_diagonal(t):
    return [(-400 + t * (W + 800), 160 + t * 780, 5 + 2 * t, 25, 15, -8)]


def move_turn_in(t):
    s = _smooth(t / 0.7)
    return [(W * 0.15 + W * 0.5 * s, 560, 6.0, 55 * (1 - s), 10, 6 * (1 - s))]


def move_stop_go(t):
    if t < 0.3:
        s = 0.45 * _smooth(t / 0.3)
    elif t < 0.6:
        s = 0.45
    else:
        s = 0.45 + 0.55 * ((t - 0.6) / 0.4) ** 1.5
    return [(-500 + s * (W + 1000), 560, 5.0, 0, 10, 0)]


def move_two_cross(t):
    return [(-500 + t * (W + 1000), 470, 6.0, -8, 10, 0), (W + 500 - t * (W + 1000), 560, 5.5, 8, 10, 0)]


MOVES = {"가로 지나감": move_cross_lr, "반대 방향": move_cross_rl, "다가옴": move_approach, "멀어짐": move_recede,
         "대각선": move_diagonal, "회전하며 진입·멈춤": move_turn_in, "멈췄다 출발+흔들림": move_stop_go,
         "두 대 교차": move_two_cross}
SHAKE_MOVES = {"멈췄다 출발+흔들림"}


# ---------------------------------------------------------------- 그리기
def rot(yaw, pitch, roll):
    y, p, r = (math.radians(v) for v in (yaw, pitch, roll))
    Ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
    Rx = np.array([[1, 0, 0], [0, math.cos(p), -math.sin(p)], [0, math.sin(p), math.cos(p)]])
    Rz = np.array([[math.cos(r), -math.sin(r), 0], [math.sin(r), math.cos(r), 0], [0, 0, 1]])
    return Rz @ Rx @ Ry


def make_vehicle(type_idx):
    name, size, bg, fg, lines, strip, kind, _ = PT.TYPES[type_idx]
    plate = PT.draw_plate(size, bg, fg, lines, strip)
    car, (x0, y0, pw, ph) = PT.vehicle_with_plate(kind, plate, size)      # 2 px/mm
    car = cv2.resize(car, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)  # 1 px/mm (빠르게)
    return {"name": name, "truth": PT.norm("".join(lines)), "car": car,
            "rect": (x0 / 2, y0 / 2, pw / 2, ph / 2)}


def project_car(veh, u, v, dist, yaw, pitch, roll):
    x0, y0, pw, ph = veh["rect"]
    pc = np.array([x0 + pw / 2, y0 + ph / 2])
    R = rot(yaw, pitch, roll)

    def project(pts):
        mm = np.asarray(pts, float) - pc
        X = np.c_[mm, np.zeros(len(mm))] @ R.T + [0, 0, dist * 1000]
        return FX * X[:, :2] / X[:, 2:3] + [u, v]

    ch, cw = veh["car"].shape[:2]
    src = np.float32([[0, 0], [cw, 0], [cw, ch], [0, ch]])
    corners = project([[x0, y0], [x0 + pw, y0], [x0 + pw, y0 + ph], [x0, y0 + ph]])
    plate_px = float(np.linalg.norm(corners[1] - corners[0]))
    s = min(1.0, max(plate_px * 2, 8) / pw)
    img = veh["car"] if s >= 1 else cv2.resize(veh["car"], None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    Hm = cv2.getPerspectiveTransform((src * s).astype(np.float32), project(src).astype(np.float32))
    return img, Hm, corners, plate_px


def line_kernel(length, angle):
    L = max(1, int(round(length)))
    if L <= 1:
        return None
    k = np.zeros((L, L), np.float32)
    c = (L - 1) / 2
    dx, dy = math.cos(angle) * c, math.sin(angle) * c
    cv2.line(k, (int(round(c - dx)), int(round(c - dy))), (int(round(c + dx)), int(round(c + dy))), 1.0, 1)
    return k / max(k.sum(), 1e-6)


def make_background(rng):
    g = np.linspace(150, 80, H, dtype=np.float32)[:, None, None]
    bg = np.repeat(np.repeat(g, W, 1), 3, 2).astype(np.uint8)
    for _ in range(25):                                      # 배경 잡동사니 (건물·표지 흉내)
        x, y = int(rng.integers(0, W)), int(rng.integers(0, H))
        w, h = int(rng.integers(40, 400)), int(rng.integers(20, 300))
        cv2.rectangle(bg, (x, y), (x + w, y + h), [int(c) for c in rng.integers(40, 220, 3)], -1)
    for _ in range(8):                                       # 글자 비슷한 줄무늬 (헷갈리게)
        x, y = int(rng.integers(0, W - 200)), int(rng.integers(0, H - 40))
        cv2.putText(bg, str(int(rng.integers(1000, 9999))), (x, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                    (20, 20, 20), 2)
    return bg.astype(np.float32)


def render_frame(bg, cars, poses, prev_centers, cond, rng, shake):
    """한 프레임: 차마다 투영 → (움직임 블러) → 합성 → 조명·초점·잡음·흔들림·JPEG. 번호판 정답 박스 목록도 돌려줌"""
    out = bg.copy()
    if cond["light"] == "역광":
        out = np.clip(out * 1.25 + 70, 0, 255)
    gts, centers, pxs = [], [], []
    for veh, pose, prev in zip(cars, poses, prev_centers):
        img, Hm, corners, plate_px = project_car(veh, *pose)
        layer = cv2.warpPerspective(img, Hm, (W, H)).astype(np.float32)
        mask = cv2.warpPerspective(np.full(img.shape[:2], 1.0, np.float32), Hm, (W, H))
        if cond["light"] == "역광":
            layer *= 0.45
        c = corners.mean(0)
        if prev is not None:                                 # 움직인 거리 x 노출 비율 = 번짐 길이
            d = c - prev
            k = line_kernel(np.hypot(*d) * cond["exposure"], math.atan2(d[1], d[0]))
            if k is not None:
                layer = cv2.filter2D(layer * mask[..., None], -1, k)
                mask = cv2.filter2D(mask, -1, k)
                layer = layer / np.maximum(mask[..., None], 1e-3)
        m = mask[..., None]
        out = out * (1 - m) + layer * m
        gts.append([corners[:, 0].min(), corners[:, 1].min(), corners[:, 0].max(), corners[:, 1].max()])
        centers.append(c)
        pxs.append(plate_px)
    if shake is not None:
        A = shake
        out = cv2.warpAffine(out, A, (W, H), borderMode=cv2.BORDER_REFLECT)
        gts = [_affine_box(A, g) for g in gts]
    if cond["light"] == "어두움":
        out = out * 0.3
    elif cond["light"] == "저대비":
        out = 128 + (out - 128) * 0.3
    if cond["defocus"] > 0:
        out = cv2.GaussianBlur(out, (0, 0), cond["defocus"])
    out = np.clip(out + rng.normal(0, cond["noise"], out.shape), 0, 255).astype(np.uint8)
    out = cv2.imdecode(cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, cond["jpeg"]])[1], cv2.IMREAD_COLOR)
    return out, gts, centers, pxs


def _affine_box(A, g):
    pts = np.array([[g[0], g[1]], [g[2], g[1]], [g[2], g[3]], [g[0], g[3]]], float)
    p = pts @ A[:, :2].T + A[:, 2]
    return [p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()]


def visible(g, margin=4):
    return g[0] >= margin and g[1] >= margin and g[2] <= W - margin and g[3] <= H - margin


# ---------------------------------------------------------------- 시험
def main():
    ap = argparse.ArgumentParser(description="번호판 종합 시험 (합성 영상)")
    ap.add_argument("--frames", type=int, default=120, help="장면 하나의 프레임 수 (30 fps 기준 4 초)")
    ap.add_argument("--types", default=None, help="번호판 종류 번호 (쉼표, 생략하면 12종 모두)")
    ap.add_argument("--moves", default=None, help="움직임 이름 (쉼표, 생략하면 모두)")
    ap.add_argument("--configs", default=",".join(CONFIGS))
    ap.add_argument("--detector", default=PM.DEFAULT_DETECTOR)
    ap.add_argument("--reader", default="EOCR_KR")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    types = [int(x) for x in a.types.split(",")] if a.types else list(range(len(PT.TYPES)))
    moves = a.moves.split(",") if a.moves else list(MOVES)
    configs = a.configs.split(",")
    out = os.path.join(paths.ROOT, "results", f"plate_stress_{datetime.now():%Y%m%d_%H%M%S}")
    os.makedirs(os.path.join(out, "samples"), exist_ok=True)

    print("모델 불러오는 중...", flush=True)
    detect = PM.DETECTORS[a.detector]["build"]("cuda:0")
    read = PM.READERS[a.reader]["build"]("cuda:0")
    vehicles = {i: make_vehicle(i) for i in range(len(PT.TYPES))}
    rng_master = np.random.default_rng(a.seed)

    rows = []
    det_ms = []
    proc_ms = defaultdict(list)
    t_start = time.time()
    n_scen = len(types) * len(moves)
    k_scen = 0
    for ti in types:
        for mv in moves:
            k_scen += 1
            seed = int(rng_master.integers(1 << 30))
            rng = np.random.default_rng(seed)
            cond = {"light": LIGHTS[int(rng.integers(len(LIGHTS)))],
                    "exposure": float(rng.choice([0.1, 0.3, 0.6])),         # 프레임 사이 이동 거리 중 노출 동안 움직인 비율
                    "defocus": float(rng.choice([0.0, 0.0, 0.8, 1.5])),
                    "noise": float(rng.choice([3.0, 6.0, 12.0])),
                    "jpeg": int(rng.choice([70, 85, 95]))}
            cars = [vehicles[ti]]
            if mv == "두 대 교차":
                cars.append(vehicles[(ti + 5) % len(PT.TYPES)])
            bg = make_background(rng)
            pipes = {}
            for cn in configs:
                q = queue.Queue()
                p = RecognitionPipeline(q, lambda: (None, None))
                p.read, p._det_has_conf = read, True
                p.det_key, p.read_key = a.detector, a.reader
                p.W, p.H = W, H
                p.reset_stats()
                for k, v in CONFIGS[cn].items():
                    setattr(p, k, v)
                pipes[cn] = (p, q)
            track = {(cn, ci): {"ids": [], "shown": 0, "vis": 0, "locked": [], "first_vis": None, "lock_at": None}
                     for cn in configs for ci in range(len(cars))}
            prev = [None] * len(cars)
            for fi in range(a.frames):
                t = fi / max(a.frames - 1, 1)
                poses = MOVES[mv](t)
                shake = None
                if mv in SHAKE_MOVES:
                    ang = rng.normal(0, 1.5)
                    shake = cv2.getRotationMatrix2D((W / 2, H / 2), ang, 1.0)
                    shake[:, 2] += rng.normal(0, 6, 2)
                frame, gts, centers, pxs = render_frame(bg, cars, poses, prev, cond, rng, shake)
                prev = centers
                t0 = time.perf_counter()
                cached = detect(frame, conf=PM.LOW_CONF)
                det_ms.append((time.perf_counter() - t0) * 1000)

                def fake(fr, conf=None, _c=cached):
                    keep = _c[1] >= (PM.YOLO_CONF if conf is None else conf)
                    return _c[0][keep].reshape(-1, 4), _c[1][keep]
                if fi == a.frames // 2 and ti == types[0]:
                    cv2.imwrite(os.path.join(out, "samples", f"{mv}.jpg"), cv2.resize(frame, (W // 2, H // 2)))
                for cn in configs:
                    p, q = pipes[cn]
                    p.detect = fake
                    t0 = time.perf_counter()
                    p.process(frame, fi, save=False)
                    proc_ms[cn].append((time.perf_counter() - t0) * 1000)
                    msg = None
                    while not q.empty():
                        kk, d = q.get()
                        if kk == "frame":
                            msg = d
                    rws = msg["rows"] if msg else []
                    for ci, g in enumerate(gts):
                        st = track[(cn, ci)]
                        vis = visible(g)
                        best = max(rws, key=lambda r: TR.iou(r["box"], g), default=None)
                        hit = best is not None and TR.iou(best["box"], g) >= 0.3 and best["state"] != "entering"
                        if vis:
                            st["vis"] += 1
                            st["shown"] += int(hit)
                            if st["first_vis"] is None:
                                st["first_vis"] = fi
                        if hit:
                            if best["no"] is not None and (not st["ids"] or st["ids"][-1] != best["no"]):
                                st["ids"].append(best["no"])
                            if best["state"] == "locked" and best["text"]:
                                st["locked"].append(best["text"])
                                if st["lock_at"] is None:
                                    st["lock_at"] = fi
            for (cn, ci), st in track.items():
                veh = cars[ci]
                final = st["locked"][-1] if st["locked"] else ""
                ever_wrong = any(x != veh["truth"] for x in st["locked"])
                rows.append({"config": cn, "type": veh["name"], "move": mv, "car": ci, "truth": veh["truth"],
                             **{k: cond[k] for k in ("light", "exposure", "defocus", "noise", "jpeg")},
                             "final": final, "correct": int(final == veh["truth"]),
                             "wrong_lock": int(ever_wrong), "ids": len(set(st["ids"])),
                             "id_seq": " ".join(map(str, st["ids"])),
                             "shown_pct": round(100 * st["shown"] / st["vis"], 1) if st["vis"] else "",
                             "visible_frames": st["vis"],
                             "frames_to_lock": (st["lock_at"] - st["first_vis"]) if st["lock_at"] is not None
                             and st["first_vis"] is not None else ""})
            el = time.time() - t_start
            print(f"  [{k_scen}/{n_scen}] {vehicles[ti]['name'][:14]} / {mv} / {cond['light']} 노출{cond['exposure']} "
                  f"초점{cond['defocus']} — " + ", ".join(
                      f"{cn}: {'O' if r['correct'] else ('X' if r['wrong_lock'] else '-')}{r['ids']}"
                      for cn in configs for r in rows[-len(track):] if r['config'] == cn) +
                  f"  ({el / 60:.1f}분, 남은 약 {el / k_scen * (n_scen - k_scen) / 60:.0f}분)", flush=True)

    with open(os.path.join(out, "runs.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    write_summary(out, rows, configs, det_ms, proc_ms, a)
    print(f"\n결과: {out}")


def _agg(rs):
    n = len(rs)
    if not n:
        return "-"
    ok = sum(r["correct"] for r in rs)
    wrong = sum(r["wrong_lock"] for r in rs)
    one_id = sum(r["ids"] == 1 for r in rs if r["ids"] > 0)
    with_id = sum(r["ids"] > 0 for r in rs)
    shown = [r["shown_pct"] for r in rs if r["shown_pct"] != ""]
    sh = f"{np.mean(shown):.0f}%" if shown else "-"
    return f"**{ok}/{n}** ({100 * ok / n:.0f}%) · 틀림 {wrong} · ID1 {one_id}/{with_id} · 표시 {sh}"


def write_summary(out, rows, configs, det_ms, proc_ms, a):
    L = [f"# 번호판 종합 시험 ({datetime.now():%Y-%m-%d %H:%M})", "",
         f"- 합성 1920x1080, 장면 {a.frames} 프레임, 감지 {a.detector} (입력 {PM.YOLO_IMGSZ}), 글자 {a.reader}, "
         f"글자 고정까지 {TR.LOCK_FRAMES} 번",
         f"- 번호판 {len(set(r['type'] for r in rows))}종 x 움직임 {len(set(r['move'] for r in rows))}가지, 장면마다 조명·모션블러(노출)·초점 흐림·잡음·JPEG 를 섞음",
         f"- 처리 시간 (프레임당): 감지 {np.mean(det_ms):.1f} ms + " +
         ", ".join(f"{cn} 추적·인식 {np.mean(proc_ms[cn]):.1f} ms" for cn in configs),
         "- 칸 = **맞게 고정/번호판 수** · 틀림 = 틀린 번호로 고정한 적 있음 · ID1 = 화면 ID 1개로 끝까지 (ID 받은 번호판 중) · 표시 = 보이는 동안 박스 표시 평균", ""]

    def table(title, key, values):
        L.extend([f"## {title}", "", "| " + title + " | " + " | ".join(configs) + " |",
                  "| --- | " + " | ".join("---" for _ in configs) + " |"])
        for v in values:
            L.append(f"| {v} | " + " | ".join(_agg([r for r in rows if r['config'] == cn and r[key] == v])
                                              for cn in configs) + " |")
        L.append("")
    L.extend(["## 전체", "", "| 설정 | 결과 |", "| --- | --- |"])
    for cn in configs:
        L.append(f"| {cn} | " + _agg([r for r in rows if r["config"] == cn]) + " |")
    L.append("")
    table("움직임", "move", list(dict.fromkeys(r["move"] for r in rows)))
    table("번호판 종류", "type", list(dict.fromkeys(r["type"] for r in rows)))
    table("조명", "light", LIGHTS)
    table("노출 (모션블러)", "exposure", sorted(set(r["exposure"] for r in rows)))
    table("초점 흐림", "defocus", sorted(set(r["defocus"] for r in rows)))
    ftl = {cn: [r["frames_to_lock"] for r in rows if r["config"] == cn and r["frames_to_lock"] != ""] for cn in configs}
    L.append("## 다 보인 뒤 고정까지 프레임 (중앙값)")
    L.append("")
    L.append(" / ".join(f"{cn}: {np.median(v):.0f}" if v else f"{cn}: -" for cn, v in ftl.items()))
    wrong = [r for r in rows if r["wrong_lock"]]
    if wrong:
        L.extend(["", "## 틀린 번호로 고정한 경우", "", "| 설정 | 종류 | 움직임 | 정답 | 고정 |", "| --- | --- | --- | --- | --- |"])
        for r in wrong[:40]:
            L.append(f"| {r['config']} | {r['type']} | {r['move']} | {r['truth']} | {r['final']} |")
    with open(os.path.join(out, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
