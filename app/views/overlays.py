"""
View — 영상 위에 그리는 정보 (왼쪽 위 카메라 정보, 오른쪽 위 왜곡 분석, 가이드선)
ViewModel.render_state() 가 준 값(st)만 보고 그린다
"""

from datetime import datetime

import cv2

from app.models.focus import peaking_mask, roi_rect
from app.models.lens_distortion import distortion_type
from app.views import guides

FOCUS_COLORS = {"IN FOCUS": (0, 220, 0), "ALMOST": (0, 200, 255), "BLURRY": (0, 0, 255),
                "SEARCHING": (230, 230, 230)}   # BGR
PEAKING_COLOR = (0, 0, 255)


def draw_peaking(view, threshold):
    """포커스 피킹: 초점이 맞은 윤곽을 빨간색으로 (화면용 이미지에서 계산)"""
    gray = cv2.cvtColor(view, cv2.COLOR_BGR2GRAY)
    view[peaking_mask(gray, threshold) > 0] = PEAKING_COLOR


def draw_focus(view, st):
    """초점 도우미: 측정 영역 상자 + 선명도 막대(흰 선 = 최고값) + 판정"""
    h, w = view.shape[:2]
    x0, y0, x1, y1 = roi_rect(w, h)
    color = FOCUS_COLORS.get(st.focus_verdict, (255, 255, 255)) if st.focus_peak > 0 else (200, 200, 200)
    cv2.rectangle(view, (x0, y0), (x1, y1), color, 2, cv2.LINE_AA)

    # 막대: 최고값의 110% 를 막대 끝으로. 흰 세로선 = 최고값 위치
    bx0, bx1, by = x0, x1, y1 + 8
    bw, bh = bx1 - bx0, 14
    view[by:by + bh, bx0:bx1] = (view[by:by + bh, bx0:bx1] * 0.35).astype(view.dtype)
    full = max(st.focus_peak * 1.1, 1e-6)
    cur = int(bw * min(1.0, st.focus_score / full))
    cv2.rectangle(view, (bx0, by), (bx0 + cur, by + bh), color, -1)
    px = bx0 + int(bw * st.focus_peak / full)
    cv2.line(view, (px, by - 3), (px, by + bh + 3), (255, 255, 255), 2)

    arrow = "^" if st.focus_trend > st.focus_score * 0.002 else "v" if st.focus_trend < -st.focus_score * 0.002 else "-"
    text = (f"Focus {st.focus_score:.0f} / peak {st.focus_peak:.0f} ({st.focus_ratio * 100:.0f}%) {arrow} "
            f"{st.focus_verdict}")
    font, s, t = cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
    (tw, th), _ = cv2.getTextSize(text, font, s, t)
    tx, ty = max(4, (w - tw) // 2), max(th + 6, y0 - 10)
    view[ty - th - 5:ty + 5, tx - 4:tx + tw + 4] = (view[ty - th - 5:ty + 5, tx - 4:tx + tw + 4] * 0.35).astype(view.dtype)
    cv2.putText(view, text, (tx, ty), font, s, color, t, cv2.LINE_AA)


def draw_box(view, lines, right=False, colors=None):
    font, scale, thick, gap = cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1, 20
    box_w = max(cv2.getTextSize(t, font, scale, thick)[0][0] for t in lines) + 16
    box_h = gap * len(lines) + 10
    x0 = view.shape[1] - box_w if right else 0
    # 반투명 검은 배경 박스
    roi = view[0:box_h, x0:x0 + box_w]
    view[0:box_h, x0:x0 + box_w] = (roi * 0.4).astype(view.dtype)
    for n, text in enumerate(lines):
        color = colors[n] if colors else (255, 255, 255)
        cv2.putText(view, text, (x0 + 8, gap * (n + 1)), font, scale, color, thick, cv2.LINE_AA)


def draw_info(view, w, h, st, view_fps):
    cam_fps = f"{st.setting_fps:.0f}" if st.setting_fps > 0 else "N/A"
    an = f"{st.analysis_fps:4.1f}" if st.analyze_on else "off"
    lines = [
        f"FPS camera {st.camera_fps:4.1f}  view {view_fps:4.1f}  analysis {an}",
        f"Camera setting FPS {cam_fps}",
        f"Resolution {w}x{h}  {st.fourcc}",
        f"Camera {st.camera_name}" + (f"  ({st.model})" if st.model else ""),
        f"Backend {st.backend}",
        f"Frame #{st.frame_no}  Time {st.elapsed // 60:02d}:{st.elapsed % 60:02d}",
        f"Brightness {st.brightness:5.1f}  Captured {st.captured}",
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    ]
    colors = [(0, 255, 0)] + [(255, 255, 255)] * (len(lines) - 1)
    draw_box(view, lines, colors=colors)


def draw_distortion(view, st):
    res = st.res
    green, white, yellow, gray = (0, 255, 0), (255, 255, 255), (0, 255, 255), (170, 170, 170)
    lines, colors = ["[ Lens Distortion ]"], [yellow]
    b = st.board
    mode = "auto" if st.auto_board_on else "fixed"
    if b is None:
        lines.append("Board: searching (auto)..." if st.auto_board_on else "Board: not set")
        colors.append(gray)
    elif res["found"]:
        diag = res["diag"]
        lines += [f"Board {b[0]}x{b[1]} ({mode}): DETECTED",
                  f"Line error mean {res['mean_err']:.2f}px  max {res['max_err']:.2f}px",
                  f"  ({res['max_err'] / diag * 100:.3f}% of diagonal)",
                  f"Bend {distortion_type(res['bulge'], diag)}  ({res['bulge']:+.2f}px)"]
        colors += [green, white, white, white]
        if "corrected_err" in res:
            lines.append(f"After correction mean {res['corrected_err']:.2f}px")
            colors.append(green)
    else:
        lines.append(f"Board {b[0]}x{b[1]} ({mode}): not found")
        colors.append(gray)

    status = " (calibrating...)" if st.calibrating else ""
    lines.append(f"Samples {st.samples}/{st.max_samples} (min {st.min_samples}){status}")
    colors.append(white)
    if st.calib is not None:
        k1, k2, p1, p2, k3 = st.calib["dist"]
        K = st.calib["K"]
        lines += [f"k1 {k1:+.4f}  k2 {k2:+.4f}  k3 {k3:+.4f}",
                  f"p1 {p1:+.5f}  p2 {p2:+.5f}",
                  f"fx {K[0, 0]:.0f}  fy {K[1, 1]:.0f}  cx {K[0, 2]:.0f}  cy {K[1, 2]:.0f}",
                  f"RMS {st.calib['rms']:.3f}px  ({st.calib['n']} samples)"]
        colors += [white, white, white, green if st.calib["rms"] < 0.5 else yellow]
        if st.lens is not None:
            L = st.lens
            lines += [f"Lens: {L['kind_en']}, {L['level_en']}",
                      f"  corner shift {L['corner_pct']:+.2f}% ({L['corner_px']:.1f}px)",
                      f"  tilt: {'YES' if L['tilted'] else 'no'} ({L['tangential_px']:.1f}px)"]
            colors += [yellow, white, white]
        else:
            lines.append("Lens: unstable - collect more samples")
            colors.append(gray)
    else:
        lines.append("Calibration: not yet")
        colors.append(gray)
    lines.append(f"View: {'UNDISTORTED' if st.undistort_on else 'RAW'}"
                 f"  (auto correct {'ON' if st.auto_correct_on else 'OFF'})")
    colors.append(yellow if st.undistort_on else white)
    draw_box(view, lines, right=True, colors=colors)


GRID_ROWS = {"ALL": 6, "GRID": 6, "FINE": 12}     # 정사각 격자: 짧은 변을 몇 칸으로 나눌지
LINE_COLORS = {"THIRDS": (0, 255, 255), "CENTER": (255, 0, 255), "DIAG": (0, 165, 255)}  # BGR


def line_guides(mode, w, h):
    """가이드 종류 → (선 목록, 설명 문구)"""
    if mode in GRID_ROWS:
        lines, cell = guides.square_grid_lines(w, h, GRID_ROWS[mode])
        return lines, f"Square grid: cell {cell:.0f}px ({min(w, h) / cell:.0f} cells on short side)"
    if mode == "THIRDS":
        return guides.thirds_lines(w, h), "Rule of thirds"
    if mode == "CENTER":
        return guides.center_lines(w, h), f"Center + circles (r step {min(w, h) / 8:.0f}px)"
    if mode == "DIAG":
        return guides.diagonal_lines(w, h), "Diagonals + center"
    return [], ""


def draw_guides(view, scale, w, h, st):
    res, calib = st.res, st.calib
    lines, label = line_guides(st.guide_mode, w, h)
    if lines:
        # 원본 화면 + 보정값 있음 -> 렌즈가 직선을 휘게 보이는 모양(곡선), 그 외 -> 반듯한 선
        curved = calib if (calib is not None and not st.undistort_on) else None
        color = LINE_COLORS.get(st.guide_mode, guides.GRID_COLOR)
        guides.draw_polylines(view, scale, guides.distort_polylines(lines, curved), color)
        guides.draw_center_mark(view)
        guides.draw_label(view, label + ("  [lens-curved]" if curved is not None else ""), color)

    if st.guide_mode not in ("ALL", "BOARD"):
        return
    if res is None or not res["found"]:
        guides.draw_banner(view, "Guide: show the checkerboard to check line straightness", (200, 200, 200))
        return
    if st.undistort_on and "corners_und" in res:
        max_e, mean_e = guides.draw_board_guides(view, scale, res["corners_und"], st.board)
        text, color = guides.verdict(max_e)
        guides.draw_banner(view, f"Calibration check: line error max {max_e:.2f}px, "
                                 f"mean {mean_e:.2f}px -> {text}", color)
    else:
        max_e, mean_e = guides.draw_board_guides(view, scale, res["corners"].reshape(-1, 2), st.board)
        hint = "press U to check correction" if calib is not None else "collect samples to calibrate"
        guides.draw_banner(view, f"RAW lens bend: max {max_e:.2f}px, mean {mean_e:.2f}px  ({hint})",
                           (255, 255, 255))


def compose(frame, st, max_width, view_fps):
    """화면에 보낼 이미지: 축소 + 체커보드 코너 + 가이드 + 정보 박스 (캡처 원본은 건드리지 않음)"""
    h, w = frame.shape[:2]
    scale = min(1.0, max_width / w)
    if scale < 1.0:
        view = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        view = frame.copy()
    if st.peaking_on:
        draw_peaking(view, st.peaking_threshold)      # 다른 그림보다 먼저 (글자·선에는 칠하지 않게)
    res = st.res
    board_guide = st.guide_on and st.guide_mode in ("ALL", "BOARD")
    if res is not None and res["found"] and not st.undistort_on and not board_guide:
        cv2.drawChessboardCorners(view, st.board, res["corners"] * scale, True)
    if st.guide_on:
        draw_guides(view, scale, w, h, st)
    if st.focus_on:
        draw_focus(view, st)
    draw_info(view, w, h, st, view_fps)
    if res is not None:
        draw_distortion(view, st)
    return view
