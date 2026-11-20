"""번호판 감지기(YOLO) TensorRT FP16 엔진 만들기 + 속도·결과 비교.

    python export_trt.py                      # models\\plate_yolo\\*.pt 모두
    python export_trt.py --detectors YOLO-v1s --rebuild

엔진은 models\\plate_yolo\\<이름>_<입력 크기>.engine 에 저장되고, 앱은 엔진이 있으면 자동으로 씀
(없으면 지금처럼 PyTorch FP16). 엔진은 만든 GPU·TensorRT 버전에서만 동작.
비교: 1920x1080 프레임(합성 번호판 장면)에서 PyTorch FP16 vs TensorRT FP16 의 감지 시간(중앙값)과 박스 일치.
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from app.models import plate_models as PM  # noqa: E402


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def test_frames(n=40, seed=0):
    """비교용 1920x1080 프레임: plate_stress_test 의 합성 장면을 쓰고, 안 되면 무작위 영상"""
    try:
        import plate_stress_test as ST
        rng = np.random.default_rng(seed)
        vehicles = {i: ST.make_vehicle(i) for i in range(len(ST.PT.TYPES))}
        moves = list(ST.MOVES)
        frames = []
        for i in range(n):
            ti, mv = i % len(vehicles), moves[i % len(moves)]
            cond = {"light": ST.LIGHTS[i % len(ST.LIGHTS)], "exposure": 0.1, "defocus": 0.0, "noise": 3.0, "jpeg": 95}
            cars = [vehicles[ti]] + ([vehicles[(ti + 5) % len(vehicles)]] if mv == "두 대 교차" else [])
            poses = ST.MOVES[mv](float(rng.uniform(0.2, 0.8)))
            frame = ST.render_frame(ST.make_background(rng), cars, poses, [None] * len(cars), cond, rng, None)[0]
            frames.append(frame)
        return frames, "합성 번호판 장면 (plate_stress_test)"
    except Exception as e:
        print(f"합성 장면을 못 만듦 ({e}) → 무작위 영상")
    rng = np.random.default_rng(seed)
    return [rng.integers(0, 255, (1080, 1920, 3), dtype=np.uint8) for _ in range(n)], "무작위 영상 (박스 비교 의미 없음)"


def time_detect(detect, frames, conf):
    for f in frames[:5]:
        detect(f, conf=conf)                          # 예열
    ms, outs = [], []
    for f in frames:
        t0 = time.perf_counter()
        xyxy, cf = detect(f, conf=conf)
        ms.append((time.perf_counter() - t0) * 1000)
        outs.append((xyxy, cf))
    return float(np.median(ms)), outs


def compare(outs_a, outs_b, thr=0.9):
    """같은 프레임의 박스가 IoU thr 이상으로 짝지어지는 비율"""
    match = total = 0
    for (a, _), (b, _) in zip(outs_a, outs_b):
        total += max(len(a), len(b))
        used = set()
        for box in a:
            best = max(((iou(box, c), j) for j, c in enumerate(b) if j not in used), default=(0, -1))
            if best[0] >= thr:
                used.add(best[1])
                match += 1
    return match, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detectors", nargs="*", help="예: YOLO-v1s YOLO-v1l (기본: 모두)")
    ap.add_argument("--rebuild", action="store_true", help="엔진이 있어도 다시 만듦")
    ap.add_argument("--frames", type=int, default=40)
    args = ap.parse_args()

    import glob
    pts = sorted(glob.glob(os.path.join(PM.YOLO_DIR, "*.pt")))
    keys = {PM._yolo_key(p): p for p in pts}
    sel = args.detectors or list(keys)
    frames, kind = test_frames(args.frames)
    print(f"비교 프레임: {len(frames)}장, {kind}, 입력 {PM.YOLO_IMGSZ}")

    rows = []
    for k in sel:
        pt = keys[k]
        eng = PM.engine_path(pt)
        if args.rebuild or not os.path.isfile(eng):
            print(f"[{k}] TensorRT FP16 엔진 만드는 중 ... (몇 분)")
            t0 = time.time()
            PM.export_engine(pt)
            print(f"[{k}] 만듦 {time.time() - t0:.0f} s → {eng}")
        # PyTorch: 엔진을 잠시 숨기지 않고 직접 만듦
        from ultralytics import YOLO
        m_pt = YOLO(pt, task="detect")
        m_trt = YOLO(eng, task="detect")

        def mk(model, kw):
            def detect(frame, conf=None):
                r = model.predict(frame, conf=conf, imgsz=PM.YOLO_IMGSZ, device=0, verbose=False, **kw)[0]
                return r.boxes.xyxy.cpu().numpy().astype(float), r.boxes.conf.cpu().numpy().astype(float)
            return detect
        for conf in (PM.YOLO_CONF, PM.LOW_CONF):
            t_pt, o_pt = time_detect(mk(m_pt, {"quantize": "fp16"}), frames, conf)
            t_trt, o_trt = time_detect(mk(m_trt, {}), frames, conf)
            n_pt = sum(len(o[0]) for o in o_pt)
            n_trt = sum(len(o[0]) for o in o_trt)
            m, tot = compare(o_pt, o_trt)
            rows.append((k, conf, t_pt, t_trt, n_pt, n_trt, m, tot))
            print(f"[{k}] conf {conf}: PyTorch FP16 {t_pt:.1f} ms → TensorRT FP16 {t_trt:.1f} ms, "
                  f"박스 {n_pt} / {n_trt}, IoU≥0.9 일치 {m}/{tot}")
        size = os.path.getsize(eng) / 1e6
        print(f"[{k}] 엔진 {size:.0f} MB")

    print("\n| 감지기 | conf | PyTorch FP16 | TensorRT FP16 | 박스 수 (PT / TRT) | 박스 일치 |")
    print("| --- | --- | --- | --- | --- | --- |")
    for k, conf, a, b, na, nb, m, tot in rows:
        print(f"| {k} | {conf} | {a:.1f} ms | {b:.1f} ms | {na} / {nb} | {m}/{tot} |")


if __name__ == "__main__":
    main()
