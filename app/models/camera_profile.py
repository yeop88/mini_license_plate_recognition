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

from app.models.camera import fourcc_text, open_camera
from app.paths import PROFILE_PATH

PROBE_FORMATS = ["MJPG", "YUY2"]          # OpenCV 로 지정할 수 있는 형식
FALLBACK_SIZES = [(1920, 1080), (1280, 720), (640, 480)]   # 카메라가 지원 모드를 알려 주지 않을 때
PROBE_SECONDS = 1.0                       # 모드마다 FPS 를 재는 시간
MAX_FPS_RATIO = 0.97                      # 카메라가 알려 준 최대 FPS 의 97% 이상이면 '최고 프레임' 달성으로 봄
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


def measure_mode(cap, width, height, fourcc, seconds=PROBE_SECONDS):
    """이미 열린 카메라의 해상도·형식을 바꾸고 실측 FPS 를 잰다 (카메라를 다시 열면 5~10초 걸리는 카메라가 있어서)"""
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    for _ in range(3):                      # 바꾼 직후 몇 장은 느리게 오므로 버린다
        cap.read()
    n, frame, t0 = 0, None, time.time()
    while time.time() - t0 < seconds:
        ok, f = cap.read()
        if ok:
            n, frame = n + 1, f
    return {"request": f"{fourcc} {width}x{height}", "request_fourcc": fourcc, "ok": n > 0,
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fourcc": fourcc_text(cap.get(cv2.CAP_PROP_FOURCC)),
            "fps": round(n / (time.time() - t0), 1)}, frame


def probe_camera(index, width=1920, height=1080, stop=None, log=None):
    """
    카메라 스펙 측정: 최고 화질 / 최고 프레임 / 요청 해상도(width x height)에서 가장 빠른 형식.
      1. 카메라가 알려 주는 지원 모드 목록(해상도·형식·최대 FPS)을 받는다
      2. 최고 화질 = 가장 큰 해상도를 형식별로 재서 FPS 가 높은 쪽
      3. 최고 프레임 = 큰 해상도부터 재다가 '알려 준 최대 FPS 의 97%' 이상이 나온 첫 해상도
         (같은 FPS 면 해상도가 큰 쪽이 낫기 때문)
      4. 요청 해상도를 형식별로 재서 FPS 가 높은 쪽
    카메라는 한 번만 열고 해상도·형식만 바꿔 가며 잰다. FPS 는 요청하지 않는다.
    stop: True 를 돌려주면 중간에 그만둠 (None 반환). log: 진행 메시지를 받을 함수
    """
    say = log or (lambda m: None)
    advertised = list_formats(index)
    formats = [f for f in PROBE_FORMATS if not advertised or any(a["fourcc"] == f for a in advertised)] or PROBE_FORMATS
    sizes = sorted({(a["width"], a["height"]) for a in advertised}, key=lambda s: -s[0] * s[1]) or FALLBACK_SIZES
    adv_max = max((a["max_fps"] for a in advertised), default=0)
    say(f"지원 모드 {len(advertised)}개 (최대 {sizes[0][0]}x{sizes[0][1]}"
        + (f", 최대 {adv_max:.0f}fps" if adv_max else "") + ")")

    cap = open_camera(index, width, height)
    if cap is None:
        return {"error": f"카메라 {index}번을 열 수 없습니다"}
    measured, color = {}, None

    def meas(w, h, fmt):
        nonlocal color
        key = (w, h, fmt)
        if key not in measured:
            if stop is not None and stop():
                raise InterruptedError
            m, frame = measure_mode(cap, w, h, fmt)
            measured[key] = m
            say(f"  {fmt} {w}x{h} → {m['width']}x{m['height']} {m['fourcc']} {m['fps']:.1f}fps")
            if frame is not None and color is None:
                d = np.abs(frame[..., 0].astype(np.int16) - frame[..., 2].astype(np.int16)).mean()
                color = bool(d >= 1.0)      # 채널이 모두 같으면 흑백 카메라
        return measured[key]

    def fastest(w, h):
        ms = [meas(w, h, f) for f in formats]
        ms = [m for m in ms if m["ok"] and (m["width"], m["height"]) == (w, h)]
        return max(ms, key=lambda m: m["fps"]) if ms else None

    try:
        best_quality = fastest(*sizes[0])
        best_fps, target = None, adv_max * MAX_FPS_RATIO if adv_max else None
        for s in sizes:
            m = fastest(*s)
            if m and (best_fps is None or m["fps"] > best_fps["fps"] + 0.5):
                best_fps = m
            if target and m and m["fps"] >= target:
                break
            if s[0] * s[1] <= 640 * 480:
                break
        request = fastest(width, height)
    except InterruptedError:
        cap.release()
        return None
    cap.release()
    return {"measured_at": datetime.now().isoformat(timespec="seconds"), "model": "", "color": color,
            "advertised": advertised, "modes": list(measured.values()),
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
