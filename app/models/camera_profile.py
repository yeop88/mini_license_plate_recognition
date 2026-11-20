"""
Model — 카메라 스펙 (camera_profiles.json)

카메라는 이름(예: Global Shutter Camera)으로 구분한다. 번호는 연결 순서에 따라 바뀌기 때문.
파일 구조:
    {"last_camera": "이름",
     "cameras": {"이름": {"measured_at", "model", "color", "advertised": [...], "modes": [...],
                         "best": {"max_quality", "max_fps", "request", "request_size"},
                         "last_resolution", "last_used", "calibration_file"}}}

해상도 선택 키 (ViewModel·화면이 쓰는 값)
    "request"     : 요청 해상도(--width/--height)에서 가장 빠른 형식
    "max_fps"     : 최고 프레임   "max_quality" : 최고 화질
    "2048x1536"   : 그 해상도 (측정한 적 있으면 가장 빠른 형식, 없으면 MJPG)
"""

import json
import os
import time
from datetime import datetime

import cv2
import numpy as np

from app.models.camera import fourcc_text, open_camera, wake_frame_rate
from app.paths import PROFILE_PATH

PROBE_FORMATS = ["MJPG", "YUY2"]          # OpenCV 로 지정할 수 있는 형식
FALLBACK_SIZES = [(1920, 1080), (1280, 720), (640, 480)]   # 카메라가 지원 모드를 알려 주지 않을 때
PROBE_SECONDS = 2.0                       # 모드마다 FPS 를 재는 시간
WARMUP_SECONDS = 0.7                      # 모드를 바꾼 직후 버리는 시간 (처음 몇 장은 느리거나 검음)
MIN_FPS_RATIO = 0.25                      # 카메라가 알려 준 최대 FPS 의 25% 미만이면 '느림' → 쓰지 않음
MIN_FPS = 5.0                             # 또는 이 FPS 미만이면 '느림' (USB 2.0 에서 큰 모드가 1~3 fps 로 나오는 경우)
BLACK_LEVEL = 8                           # 프레임의 가장 밝은 값이 이보다 작으면 검은 프레임
BLACK_RATIO = 0.5                         # 검은 프레임이 이 비율 이상이면 '검은 화면' → 쓰지 않음
SAME_FPS = 1.0                            # 최고 프레임: FPS 차이가 이 이하면 해상도가 큰 쪽
PRESETS = ("request", "max_fps", "max_quality")


class ProfileRepository:
    """camera_profiles.json 읽기/쓰기"""

    def __init__(self, path=PROFILE_PATH):
        self.path = path
        try:
            with open(path, encoding="utf-8") as f:
                self.data = json.load(f)
        except (OSError, ValueError):
            self.data = {}
        self.data.setdefault("last_camera", None)
        self.data.setdefault("cameras", {})

    @property
    def last_camera(self):
        return self.data.get("last_camera")

    def get(self, name):
        return self.data["cameras"].get(name)

    def has(self, name):
        return name in self.data["cameras"]

    def put(self, name, profile):
        self.data["cameras"][name] = profile

    def ensure(self, name):
        return self.data["cameras"].setdefault(name, {})

    def mark_used(self, name, resolution_key):
        p = self.ensure(name)
        p["last_used"] = datetime.now().isoformat(timespec="seconds")
        p["last_resolution"] = resolution_key
        self.data["last_camera"] = name
        self.save()

    def save(self):
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)
        os.replace(tmp, self.path)


# ---------- 측정 ----------

def list_formats(index):
    """
    카메라가 DirectShow 로 알려 주는 지원 모드 [{"fourcc", "width", "height", "max_fps"}] (빠름, 0.1초 안팎).
    pygrabber 가 없거나 실패하면 [] (그때는 FALLBACK_SIZES 로 측정)
    """
    try:
        from pygrabber.dshow_graph import FilterGraph
        graph = FilterGraph()
        graph.add_video_input_device(index)
        out = []
        for f in graph.get_input_device().get_formats():
            # pygrabber 는 최소/최대 프레임을 뒤바꿔 돌려주므로 둘 중 큰 값이 최대 FPS
            fps = max(f.get("min_framerate") or 0, f.get("max_framerate") or 0)
            out.append({"fourcc": f["media_type_str"], "width": int(f["width"]), "height": int(f["height"]),
                        "max_fps": round(float(fps), 1)})
        del graph
        return out
    except Exception:
        return []


def measure_mode(cap, width, height, fourcc, seconds=PROBE_SECONDS, adv_fps=0.0):
    """
    이미 열린 카메라의 해상도·형식을 바꾸고 잰다 (카메라를 다시 열면 5~10초 걸리는 카메라가 있어서).
    실측 FPS, 실제 해상도·형식, 검은 프레임 비율, 밝기 → 쓸 수 있는지(ok)와 문제(problem)
    """
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    wake_frame_rate(cap)                    # 이 카메라는 모드를 바꾼 뒤 게인을 읽어야 60fps (camera.py)
    t = time.time()
    while time.time() - t < WARMUP_SECONDS:  # 바꾼 직후는 느리거나 검으므로 버린다
        cap.read()
    n, black, bright, frame, t0 = 0, 0, 0.0, None, time.time()
    while time.time() - t0 < seconds:
        ok, f = cap.read()
        if ok:
            n, frame = n + 1, f
            small = f[::8, ::8]
            black += int(small.max() < BLACK_LEVEL)
            bright += float(small.mean())
    fps = n / (time.time() - t0)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    problems = []
    if n == 0:
        problems.append("프레임 없음")
    else:
        if (w, h) != (width, height):
            problems.append(f"다른 해상도({w}x{h})로 열림")
        if black / n >= BLACK_RATIO:
            problems.append(f"검은 화면({black}/{n}장)")
        if fps < MIN_FPS:
            problems.append(f"느림({MIN_FPS:.0f}fps 미만)")
        elif adv_fps and fps < adv_fps * MIN_FPS_RATIO:
            problems.append(f"느림(알려 준 {adv_fps:.0f}fps 의 {MIN_FPS_RATIO * 100:.0f}% 미만)")
    return {"request": f"{fourcc} {width}x{height}", "request_fourcc": fourcc, "ok": not problems,
            "width": w, "height": h, "fourcc": fourcc_text(cap.get(cv2.CAP_PROP_FOURCC)),
            "fps": round(fps, 1), "advertised_fps": adv_fps, "frames": n,
            "black_ratio": round(black / n, 2) if n else None, "brightness": round(bright / n, 1) if n else None,
            "problem": ", ".join(problems)}, frame


def pick_best(modes, request_size=None):
    """쓸 수 있는(ok) 모드 중 최고 화질 / 최고 프레임 / 요청 해상도에서 가장 빠른 것"""
    ok = [m for m in modes if m.get("ok")]
    area = lambda m: m["width"] * m["height"]
    best_quality = max(ok, key=lambda m: (area(m), m["fps"]), default=None)
    top = max((m["fps"] for m in ok), default=0)
    best_fps = max((m for m in ok if m["fps"] >= top - SAME_FPS), key=lambda m: (area(m), m["fps"]), default=None)
    request = None
    if request_size:
        req = [m for m in ok if (m["width"], m["height"]) == tuple(request_size)]
        request = max(req, key=lambda m: m["fps"], default=None)
    return best_quality, best_fps, request


def probe_camera(index, width=1920, height=1080, stop=None, log=None):
    """
    카메라 호환성 시험 + 스펙 측정 (2026-11-20 바꿈: 지원 모드를 전부 재고, 검은 화면·느린 모드는 뺌).
      1. 카메라가 알려 주는 지원 모드 목록(해상도·형식·최대 FPS)을 받는다 (없으면 FALLBACK_SIZES)
      2. 모든 모드(해상도 x 형식)를 큰 해상도부터 하나씩 2초씩 잰다 (+ 요청 해상도 width x height)
         쓸 수 없음: 프레임 없음 / 다른 해상도로 열림 / 검은 화면 / 5fps 미만 또는 알려 준 최대 FPS 의 25% 미만
         쓸 수 없던 모드는 마지막에 한 번 더 잰다 (모드를 바꾼 직후 잠깐 멈추는 경우)
      3. 쓸 수 있는 모드 중 최고 화질(가장 큰 해상도) / 최고 프레임(가장 높은 FPS, 비슷하면 큰 해상도) /
         요청 해상도에서 가장 빠른 형식을 고른다
    카메라는 한 번만 열고 해상도·형식만 바꿔 가며 잰다. FPS 는 요청하지 않는다.
    처음 열 때는 요청 해상도(width x height, 60fps 모드)로 연다: 이 카메라(ELP 5MP)는 처음 연 모드의 프레임 간격을
    뒤 모드에도 쓰므로, 가장 큰 해상도(약 40fps)로 먼저 열면 모든 모드가 40fps 로 묶임 (2026-11-20 실측)
    stop: True 를 돌려주면 중간에 그만둠 (None 반환). log: 진행 메시지를 받을 함수
    """
    say = log or (lambda m: None)
    advertised = list_formats(index)
    adv = {}
    for a in advertised:
        if a["fourcc"] in PROBE_FORMATS:
            k = (a["width"], a["height"], a["fourcc"])
            adv[k] = max(adv.get(k, 0), a["max_fps"])
    if not adv:
        adv = {(w, h, f): 0.0 for w, h in FALLBACK_SIZES for f in PROBE_FORMATS}
    if (width, height, "MJPG") not in adv:
        adv[(width, height, "MJPG")] = 0.0
    order = sorted(adv, key=lambda k: (-k[0] * k[1], PROBE_FORMATS.index(k[2])))
    say(f"호환성 시험: 모드 {len(order)}개를 하나씩 {PROBE_SECONDS:.0f}초씩 잽니다 (약 {len(order) * 3 // 60 + 1}분)")

    cap = open_camera(index, width, height)
    if cap is None:
        return {"error": f"카메라 {index}번을 열 수 없습니다"}
    modes, color = [], None

    def run(i, w, h, fmt, total, retry=False):
        nonlocal color
        if stop is not None and stop():
            raise InterruptedError
        m, frame = measure_mode(cap, w, h, fmt, adv_fps=adv[(w, h, fmt)])
        say(f"  [{'재시험 ' if retry else ''}{i}/{total}] {fmt} {w}x{h} → {m['fps']:.1f}fps "
            + ("OK" if m["ok"] else f"사용 불가: {m['problem']}"))
        if frame is not None and color is None and m["ok"]:
            d = np.abs(frame[..., 0].astype(np.int16) - frame[..., 2].astype(np.int16)).mean()
            color = bool(d >= 1.0)          # 채널이 모두 같으면 흑백 카메라
        return m

    try:
        for i, (w, h, fmt) in enumerate(order, 1):
            modes.append(run(i, w, h, fmt, len(order)))
        bad = [k for k, m in enumerate(modes) if not m["ok"]]
        for n, k in enumerate(bad, 1):                     # 쓸 수 없던 모드는 한 번 더
            w, h, fmt = order[k]
            m = run(n, w, h, fmt, len(bad), retry=True)
            if m["ok"] or m["fps"] > modes[k]["fps"]:
                modes[k] = m
    except InterruptedError:
        cap.release()
        return None
    cap.release()
    best_quality, best_fps, request = pick_best(modes, (width, height))
    return {"measured_at": datetime.now().isoformat(timespec="seconds"), "model": "", "color": color,
            "advertised": advertised, "modes": modes,
            "best": {"max_quality": best_quality, "max_fps": best_fps, "request": request,
                     "request_size": [width, height]}}


# ---------- 해상도 선택 ----------

def parse_size(key):
    """'2048x1536' → (2048, 1536), 아니면 None"""
    try:
        w, h = (int(v) for v in str(key).lower().split("x"))
        return w, h
    except ValueError:
        return None


def _fastest_measured(profile, w, h):
    ok = [m for m in (profile or {}).get("modes", [])
          if m.get("ok") and m.get("fps", 0) > 0 and (m["width"], m["height"]) == (w, h)]
    return max(ok, key=lambda m: m["fps"]) if ok else None


def choose_mode(profile, key, request_size):
    """
    해상도 선택 키 → 열 설정 {"width", "height", "request_fourcc", "fps"(측정값 또는 None), "measured"}
    프리셋인데 스펙이 없으면 요청 해상도로 대신한다
    """
    best = (profile or {}).get("best") or {}
    if key in ("max_fps", "max_quality") and best.get(key):
        m = best[key]
        return {"width": m["width"], "height": m["height"], "request_fourcc": m["request_fourcc"],
                "fps": m["fps"], "measured": True}
    w, h = parse_size(key) or tuple(request_size)
    m = _fastest_measured(profile, w, h)
    if m:
        return {"width": w, "height": h, "request_fourcc": m["request_fourcc"], "fps": m["fps"], "measured": True}
    adv = [a for a in (profile or {}).get("advertised", []) if (a["width"], a["height"]) == (w, h)]
    fourcc = "MJPG" if not adv or any(a["fourcc"] == "MJPG" for a in adv) else adv[0]["fourcc"]
    return {"width": w, "height": h, "request_fourcc": fourcc, "fps": None, "measured": False}


def resolution_options(profile, request_size):
    """화면의 [해상도] 목록: [(키, 표시 이름), ...]. 프리셋 3개 + 카메라가 지원하는 해상도(큰 것부터)"""
    rw, rh = request_size
    fmt = lambda m: f"{m['width']}x{m['height']} {m['fps']:.0f}fps"
    best = (profile or {}).get("best") or {}
    req = _fastest_measured(profile, rw, rh)
    opts = [("request", f"기본 {rw}x{rh}" + (f" {req['fps']:.0f}fps" if req else ""))]
    if best.get("max_fps"):
        opts.append(("max_fps", f"최고 프레임 ({fmt(best['max_fps'])})"))
    if best.get("max_quality"):
        opts.append(("max_quality", f"최고 화질 ({fmt(best['max_quality'])})"))
    adv = (profile or {}).get("advertised", [])
    sizes = {(a["width"], a["height"]) for a in adv} | {(m["width"], m["height"])
                                                       for m in (profile or {}).get("modes", []) if m.get("ok")}
    for w, h in sorted(sizes, key=lambda s: -s[0] * s[1]):
        m = _fastest_measured(profile, w, h)
        if m:
            label = f"{w}x{h}  {m['request_fourcc']} {m['fps']:.0f}fps"
        else:
            mx = max((a["max_fps"] for a in adv if (a["width"], a["height"]) == (w, h)), default=0)
            label = f"{w}x{h}  (미측정" + (f", 최대 {mx:.0f}fps)" if mx else ")")
        opts.append((f"{w}x{h}", label))
    return opts


def record_mode(profile, request_fourcc, width, height, fourcc, fps):
    """실제로 열어 본 해상도의 FPS 를 스펙에 추가 (같은 요청이 있으면 갱신)"""
    entry = {"request": f"{request_fourcc} {width}x{height}", "request_fourcc": request_fourcc, "ok": fps > 0,
             "width": width, "height": height, "fourcc": fourcc, "fps": round(fps, 1), "source": "사용 중 측정"}
    modes = profile.setdefault("modes", [])
    for i, m in enumerate(modes):
        if m.get("request") == entry["request"]:
            modes[i] = entry
            return entry
    modes.append(entry)
    return entry
