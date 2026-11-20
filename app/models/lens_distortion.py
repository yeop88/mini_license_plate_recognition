"""
렌즈 왜곡 분석: 체커보드 검출/크기 자동 인식, 직선 휨 측정, 카메라 보정, 왜곡 종류 판정
"""

import json
import threading
import time
from datetime import datetime

import cv2
import numpy as np


def make_checkerboard(board, square_px=100, margin_px=100):
    """내부 코너가 board=(가로, 세로)개인 체커보드 이미지 (칸 수는 각각 +1)"""
    cols, rows = board[0] + 1, board[1] + 1
    img = np.full((rows * square_px + 2 * margin_px, cols * square_px + 2 * margin_px), 255, np.uint8)
    for r in range(rows):
        for c in range(cols):
            if (r + c) % 2 == 0:
                y, x = margin_px + r * square_px, margin_px + c * square_px
                img[y:y + square_px, x:x + square_px] = 0
    return img


def line_deviation(points, center):
    """
    점들이 직선에서 얼마나 벗어났는지 계산.
    반환: (최대 오차 px, 휨 방향 값) - 휨 방향 값 > 0 이면 가운데가 영상 중심 바깥쪽으로 볼록(술통형)
    """
    vx, vy, x0, y0 = cv2.fitLine(points.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01).ravel()
    d = (points[:, 0] - x0) * vy - (points[:, 1] - y0) * vx
    max_err = float(np.abs(d).max())

    # 양 끝점을 잇는 현(chord) 대비 가운데 점이 어느 쪽으로 휘었는지
    a, b, m = points[0], points[-1], points[len(points) // 2]
    ab = b - a
    t = np.dot(m - a, ab) / max(np.dot(ab, ab), 1e-9)
    foot = a + t * ab
    bulge = m - foot
    to_center = center - foot
    norm = np.linalg.norm(to_center)
    if norm < 1e-6:
        return max_err, 0.0
    # 중심 반대 방향으로 볼록하면 술통형(+)
    return max_err, float(-np.dot(bulge, to_center / norm))


class DistortionAnalyzer:
    """체커보드로 렌즈 왜곡을 실시간 측정하고, 샘플이 모이면 보정 계수를 계산한다"""

    MIN_SAMPLES = 8
    MAX_SAMPLES = 25
    SAMPLE_INTERVAL = 1.0  # 샘플 수집 최소 간격 (초)

    def __init__(self, board):
        self.set_board(board)

    def set_board(self, board):
        """체커보드 크기 설정 (None = 아직 모름, 자동 인식 대기). 크기가 바뀌면 샘플/보정값 초기화"""
        self.board = board  # 내부 코너 (가로, 세로)
        if board is not None:
            objp = np.zeros((board[0] * board[1], 3), np.float32)
            objp[:, :2] = np.mgrid[0:board[0], 0:board[1]].T.reshape(-1, 2)
            self.objp = objp
        self.reset()

    def reset(self):
        # 세대 번호: 초기화 전에 시작된 백그라운드 보정 결과는 버린다
        self.generation = getattr(self, "generation", 0) + 1
        self.new_calib = None
        self.samples = []
        self.image_size = None
        self.last_sample_time = 0.0
        self.calib = None          # 보정 결과 dict
        self.maps = None           # 보정 화면용 remap 테이블
        self.calibrating = False
        self.pending = None        # 백그라운드 보정 결과 (메인 스레드에서 반영)
        self.samples_at_calib = 0
        self.last = None           # 마지막 프레임 분석 결과

    def detect(self, frame):
        """체커보드 코너 검출. 빠르게 하려고 축소 영상에서 찾고 원본에서 정밀화한다"""
        if self.board is None:
            return None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        scale = min(1.0, 640 / w)
        small = cv2.resize(gray, None, fx=scale, fy=scale) if scale < 1.0 else gray
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK
        found, corners = cv2.findChessboardCorners(small, self.board, flags)
        if not found:
            return None
        corners = (corners / scale).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
        cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        return corners

    def measure_lines(self, pts, center):
        """체커보드 행/열의 직선 오차와 휨 방향"""
        cols, rows = self.board
        grid = pts.reshape(rows, cols, 2)
        lines = [grid[r, :, :] for r in range(rows)] + [grid[:, c, :] for c in range(cols)]
        errs, bulges = [], []
        for line in lines:
            e, b = line_deviation(line, center)
            errs.append(e)
            bulges.append(b)
        return float(np.mean(errs)), float(np.max(errs)), float(np.mean(bulges))

    def analyze(self, frame):
        h, w = frame.shape[:2]
        if self.image_size != (w, h):
            if self.image_size is not None:
                self.reset()
            self.image_size = (w, h)

        corners = self.detect(frame)
        result = {"found": corners is not None, "corners": corners, "new_sample": False}
        if corners is not None:
            pts = corners.reshape(-1, 2)
            center = np.array([w / 2, h / 2])
            diag = float(np.hypot(w, h))
            mean_e, max_e, bulge = self.measure_lines(pts, center)
            result.update(mean_err=mean_e, max_err=max_e, bulge=bulge, diag=diag)
            # 보정값이 있으면 코너를 왜곡 제거한 뒤 다시 측정 (보정 효과 확인)
            if self.calib is not None:
                und = cv2.undistortPoints(corners, self.calib["K"], self.calib["dist"],
                                          P=self.calib["K"]).reshape(-1, 2)
                result["corrected_err"] = self.measure_lines(und, center)[0]
                result["corners_und"] = und  # 보정 화면 좌표계의 코너 (가이드선 표시용)
            result["new_sample"] = self.try_add_sample(corners)

        self.collect_pending()
        self.maybe_calibrate()
        self.last = result
        return result

    def try_add_sample(self, corners):
        now = time.time()
        if now - self.last_sample_time < self.SAMPLE_INTERVAL or len(self.samples) >= self.MAX_SAMPLES:
            return False
        # 이전 샘플과 위치가 거의 같으면 버린다 (다양한 위치/각도가 있어야 보정이 정확)
        c = corners.reshape(-1, 2)
        for s in self.samples:
            if np.abs(s.reshape(-1, 2) - c).mean() < 20:
                return False
        self.samples.append(corners.copy())
        self.last_sample_time = now
        return True

    def maybe_calibrate(self):
        n = len(self.samples)
        if self.calibrating or n < self.MIN_SAMPLES:
            return
        # 처음 8장에서 한 번, 이후 3장 추가될 때마다 다시 계산
        if self.calib is not None and n - self.samples_at_calib < 3:
            return
        self.calibrating = True
        self.samples_at_calib = n
        samples = list(self.samples)
        threading.Thread(target=self._calibrate, args=(samples, self.image_size, self.generation),
                         daemon=True).start()

    def _calibrate(self, samples, size, generation):
        try:
            objpoints = [self.objp] * len(samples)
            # 초기값: 초점거리 ≈ 영상 너비, 주점 = 영상 중심. 화소는 정사각형이라고 가정
            w, h = size
            K0 = np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], np.float64)
            flags = cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_ASPECT_RATIO
            # 샘플이 적을 때 k3/주점까지 풀면 값이 튀므로 고정
            if len(samples) < 15:
                flags |= cv2.CALIB_FIX_K3 | cv2.CALIB_FIX_PRINCIPAL_POINT
            rms, K, dist, _, _ = cv2.calibrateCamera(objpoints, samples, size, K0, None, flags=flags)
            res = {"rms": rms, "K": K, "dist": dist.ravel()[:5], "n": len(samples), "size": size}
        except cv2.error as e:
            res = {"error": str(e)}
        if generation == self.generation:
            self.pending = res

    def collect_pending(self):
        """백그라운드 보정 결과를 반영. 새 결과가 있으면 dict 반환"""
        if self.pending is None:
            return None
        res, self.pending = self.pending, None
        self.calibrating = False
        if "error" in res or res["size"] != self.image_size:
            self.new_calib = res
            return res
        w, h = res["size"]
        res["dist"] = np.pad(res["dist"], (0, 5 - len(res["dist"])))
        self.calib = res
        self.maps = cv2.initUndistortRectifyMap(res["K"], res["dist"], None, res["K"], (w, h), cv2.CV_16SC2)
        self.new_calib = res
        return res

    def undistort(self, frame):
        if self.maps is None or frame.shape[1::-1] != tuple(self.image_size):
            return frame
        return cv2.remap(frame, self.maps[0], self.maps[1], cv2.INTER_LINEAR)

    def save(self, path):
        c = self.calib
        data = {
            "image_size": list(c["size"]),
            "board_inner_corners": list(self.board),
            "samples": c["n"],
            "rms_reprojection_error_px": c["rms"],
            "camera_matrix": c["K"].tolist(),
            "dist_coeffs_k1_k2_p1_p2_k3": c["dist"].tolist(),
            "saved_at": datetime.now().isoformat(timespec="seconds"),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def load(self, path):
        """save() 로 저장한 보정값을 불러와 바로 쓸 수 있게 한다 (체커보드 크기도 저장값으로). 저장 dict 반환"""
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        K = np.array(d["camera_matrix"], np.float64)
        dist = np.pad(np.array(d["dist_coeffs_k1_k2_p1_p2_k3"], np.float64), (0, 5))[:5]
        self.set_calibration({"rms": d["rms_reprojection_error_px"], "K": K, "dist": dist, "n": d["samples"],
                              "size": tuple(d["image_size"])}, tuple(d["board_inner_corners"]))
        return d

    def set_calibration(self, calib, board):
        """밖에서 구한 보정값(calib dict: rms, K, dist, n, size)을 바로 쓰게 한다. 샘플은 비움"""
        if self.board != board:
            self.set_board(board)
        else:
            self.reset()
        w, h = calib["size"]
        calib = dict(calib, dist=np.pad(np.asarray(calib["dist"], np.float64).ravel(), (0, 5))[:5])
        self.image_size = (w, h)    # 해상도가 다른 프레임이 오면 analyze() 가 초기화한다
        self.calib = calib
        self.maps = cv2.initUndistortRectifyMap(calib["K"], calib["dist"], None, calib["K"], (w, h), cv2.CV_16SC2)


# ---------------------------------------------------------------
# 캡처 이미지로 보정 (화면보다 큰 체커보드도 가능)
# ---------------------------------------------------------------
FOCAL_GUESS = 0.8    # 초점거리 = 영상 너비 x 이 값으로 고정. 정면에 가까운 사진은 초점거리를 정할 수 없지만,
                     # 화면을 펴는 데는 휨 모양(k1, k2)만 맞으면 된다 (값을 바꿔도 보정 결과가 같음, 2026-11-13 시험)


def detect_board_any(img_bgr):
    """
    체커보드 코너 찾기. 보드가 화면보다 커서 일부만 보여도 보이는 격자 전체를 쓴다.
    반환: (코너 Nx1x2 float32, 가로 코너 수, 세로 코너 수) 또는 None
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    flags = cv2.CALIB_CB_LARGER | cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
    try:
        ok, corners, meta = cv2.findChessboardCornersSBWithMeta(gray, (5, 4), flags)
    except cv2.error:
        ok = False
    if ok and meta is not None:
        rows, cols = meta.shape
        if len(corners) == rows * cols:
            return corners.reshape(-1, 1, 2).astype(np.float32), cols, rows
    size = detect_board_size(img_bgr)           # 보드 전체가 보이는 일반 경우
    if size is None:
        return None
    ok, corners = cv2.findChessboardCorners(gray, size, cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
    if not ok:
        return None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
    cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return corners.astype(np.float32), size[0], size[1]


def _grid_line_errors(pts, cols, rows):
    """격자 각 줄의 양 끝을 잇는 직선에서 코너가 벗어난 거리 (최대, 평균) px"""
    g = pts.reshape(rows, cols, 2)
    errs = []
    for line in [g[r] for r in range(rows)] + [g[:, c] for c in range(cols)]:
        a, b = line[0], line[-1]
        d = b - a
        n = np.array([-d[1], d[0]]) / max(np.linalg.norm(d), 1e-9)
        errs.append(np.abs((line - a) @ n).max())
    return float(np.max(errs)), float(np.mean(errs))


def calibrate_from_images(images):
    """
    캡처 이미지들(같은 해상도)로 렌즈 보정. images: [(이름, BGR 이미지)]
    반환: {"calib": {...}, "board", "used": [이름], "skipped": [이름], "err_before", "err_after"} 또는 {"error"}
    """
    objpts, imgpts, grids, used, skipped, size = [], [], [], [], [], None
    for name, img in images:
        h, w = img.shape[:2]
        if size is None:
            size = (w, h)
        if (w, h) != size:
            skipped.append(name)
            continue
        found = detect_board_any(img)
        if found is None:
            skipped.append(name)
            continue
        corners, cols, rows = found
        obj = np.zeros((rows * cols, 3), np.float32)
        obj[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
        objpts.append(obj)
        imgpts.append(corners)
        grids.append((cols, rows))
        used.append(name)
    if not objpts:
        return {"error": "체커보드가 보이는 캡처가 없습니다", "skipped": skipped}
    w, h = size
    K0 = np.array([[w * FOCAL_GUESS, 0, w / 2], [0, w * FOCAL_GUESS, h / 2], [0, 0, 1]], np.float64)
    flags = (cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_FOCAL_LENGTH | cv2.CALIB_FIX_PRINCIPAL_POINT |
             cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K3)
    try:
        rms, K, dist, _, _ = cv2.calibrateCamera(objpts, imgpts, size, K0, None, flags=flags)
    except cv2.error as e:
        return {"error": f"보정 계산 실패: {e}", "skipped": skipped}
    before = [_grid_line_errors(p.reshape(-1, 2), *g) for p, g in zip(imgpts, grids)]
    after = [_grid_line_errors(cv2.undistortPoints(p, K, dist, P=K).reshape(-1, 2), *g) for p, g in zip(imgpts, grids)]
    return {"calib": {"rms": float(rms), "K": K, "dist": dist.ravel(), "n": len(used), "size": size},
            "board": grids[0], "used": used, "skipped": skipped,
            "err_before": (max(e[0] for e in before), float(np.mean([e[1] for e in before]))),
            "err_after": (max(e[0] for e in after), float(np.mean([e[1] for e in after])))}


def detect_board_size(frame):
    """
    화면에 보이는 체커보드의 내부 코너 수 (가로, 세로)를 자동으로 알아낸다. 못 찾으면 None.
    findChessboardCornersSBWithMeta + CALIB_CB_LARGER: 최소 3x3 이상인 보드를 찾고 실제 크기를 돌려준다.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    scale = min(1.0, 960 / gray.shape[1])
    if scale < 1.0:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    try:
        ok, _, meta = cv2.findChessboardCornersSBWithMeta(gray, (3, 3), cv2.CALIB_CB_LARGER)
    except cv2.error:
        return None
    if not ok or meta is None:
        return None
    a, b = meta.shape[:2]
    return (max(a, b), min(a, b))  # 가로가 긴 방향으로 통일 (9x6 형태)


def classify_lens(calib):
    """
    보정 계수로 렌즈 왜곡 종류/강도를 판정한다.
    방사 왜곡: 중심에서 반지름 r(정규화 좌표)인 점은 r * (1 + k1 r^2 + k2 r^4 + k3 r^6) 로 이동한다.
      - 이동량이 음수(안쪽) -> 술통형(Barrel), 양수(바깥) -> 실패형(Pincushion),
        중간에 부호가 바뀌면 -> 콧수염형(Mustache, 술통+실패 혼합)
    접선 왜곡: p1, p2 가 크면 렌즈와 센서가 평행하지 않음(기울어짐)
    """
    k1, k2, p1, p2, k3 = calib["dist"]
    K = calib["K"]
    w, h = calib["size"]
    fx, fy = K[0, 0], K[1, 1]
    r_corner = float(np.hypot((w / 2) / fx, (h / 2) / fy))  # 영상 모서리까지의 정규화 반지름
    r = np.linspace(0, r_corner, 50)[1:]
    radial = k1 * r ** 2 + k2 * r ** 4 + k3 * r ** 6  # 반지름 대비 이동 비율
    corner_pct = float(radial[-1] * 100)
    corner_px = float(abs(radial[-1]) * r_corner * fx)

    big = np.abs(radial) > 0.002  # 0.2% 미만 구간은 판정에서 제외
    signs = set(np.sign(radial[big]).astype(int))
    if not signs:
        kind, kind_en = "왜곡 거의 없음", "None"
    elif signs == {-1}:
        kind, kind_en = "술통형(Barrel)", "Barrel"
    elif signs == {1}:
        kind, kind_en = "실패형(Pincushion)", "Pincushion"
    else:
        kind, kind_en = "콧수염형(Mustache)", "Mustache"

    mag = abs(corner_pct)
    if mag < 0.5:
        level, level_en = "매우 약함", "very weak"
    elif mag < 2:
        level, level_en = "약함", "weak"
    elif mag < 5:
        level, level_en = "보통", "moderate"
    else:
        level, level_en = "강함", "strong"

    # 접선 왜곡에 의한 모서리 이동량 (px) 대략치
    tangential_px = float(2 * np.hypot(p1, p2) * r_corner ** 2 * fx)
    tilted = tangential_px > 1.0
    return {
        "kind": kind, "kind_en": kind_en, "level": level, "level_en": level_en,
        "corner_pct": corner_pct, "corner_px": corner_px,
        "tangential_px": tangential_px, "tilted": tilted,
    }


def calib_is_stable(calib):
    w = calib["size"][0]
    return calib["rms"] < 1.0 and 0.2 * w < calib["K"][0, 0] < 5 * w


def distortion_type(bulge, diag):
    # 휨 정도가 대각선의 0.015% (1280x720 기준 약 0.2px) 미만이면 왜곡이 거의 없다고 본다
    if abs(bulge) < diag * 0.00015:
        return "None (straight)"
    return "Barrel" if bulge > 0 else "Pincushion"

