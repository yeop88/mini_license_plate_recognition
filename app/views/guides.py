"""
캘리브레이션·구도 확인용 가이드선

1. 선 가이드 (GRID 정사각 격자 / FINE 촘촘한 정사각 격자 / THIRDS 3분할 / CENTER 중심·동심원 / DIAG 대각선)
   - 정사각 격자: 짧은 변을 N칸으로 나눈 크기의 정사각 칸, 선 하나가 영상 중심을 지남
   - 보정 화면: 반듯한 선. 화면 속 직선(문틀, 책상 모서리, 체커보드)이 선과 나란하면 보정이 잘 된 것
   - 원본 화면(보정값 있음): 보정값으로 계산한 '렌즈가 직선을 휘게 보이는 모양'의 곡선.
                실제 휜 직선이 이 곡선을 따라가면 보정값(왜곡 모델)이 렌즈와 맞는 것
2. 체커보드 가이드 (BOARD)
   - 체커보드 각 행/열의 양 끝을 잇는 직선과, 코너가 그 직선에서 벗어난 정도(색/화살표)를 표시
   - 보정 화면에서 모든 코너가 직선 위에 있으면(초록) 보정이 잘 된 것
"""

import cv2
import numpy as np

GRID_COLOR = (255, 200, 0)       # 하늘색 (BGR)
CENTER_COLOR = (0, 0, 255)       # 빨강
LINE_COLOR = (0, 255, 255)       # 노랑
GOOD, OK, BAD = (0, 220, 0), (0, 200, 255), (0, 0, 255)  # 초록 / 주황 / 빨강

GOOD_PX = 1.0   # 보정 화면에서 직선 최대 오차가 이 값 미만이면 GOOD
OK_PX = 2.5     # 이 값 미만이면 OK, 이상이면 BAD


def deviation_color(err):
    return GOOD if err < GOOD_PX else OK if err < OK_PX else BAD


def verdict(max_err):
    if max_err < GOOD_PX:
        return "GOOD", GOOD
    if max_err < OK_PX:
        return "OK", OK
    return "BAD", BAD


def grid_polylines(w, h, calib=None, nx=8, ny=6, steps=40):
    """
    (예전 방식) 화면을 가로 nx, 세로 ny 로 나눈 격자. 영상 비율에 따라 칸이 직사각형이 된다.
    지금 화면은 square_grid_lines(정사각 칸)를 쓴다. calib 가 있으면 렌즈 왜곡을 적용한 곡선
    """
    lines = []
    for i in range(nx + 1):
        x = w * i / nx
        lines.append(np.stack([np.full(steps, x), np.linspace(0, h, steps)], 1))
    for j in range(ny + 1):
        y = h * j / ny
        lines.append(np.stack([np.linspace(0, w, steps), np.full(steps, y)], 1))
    return distort_polylines(lines, calib)


# ---------------------------------------------------------------
# 여러 가지 가이드선 (모두 원본 해상도 좌표의 꺾은선 목록을 돌려준다)
# ---------------------------------------------------------------
def _hline(y, x0, x1, steps):
    return np.stack([np.linspace(x0, x1, steps), np.full(steps, float(y))], 1)


def _vline(x, y0, y1, steps):
    return np.stack([np.full(steps, float(x)), np.linspace(y0, y1, steps)], 1)


def square_grid_lines(w, h, rows=6, steps=40):
    """
    정사각 칸 격자: 짧은 변을 rows 칸으로 나눈 크기를 한 칸으로 하고, 선 하나가 영상 중심을 지나게 놓는다.
    반환: (선 목록, 칸 크기 px)
    """
    cell = min(w, h) / rows
    cx, cy = w / 2, h / 2
    lines = []
    k = -int(np.ceil(cx / cell))
    while cx + k * cell <= w + 1e-6:
        x = cx + k * cell
        if x >= -1e-6:
            lines.append(_vline(x, 0, h, steps))
        k += 1
    k = -int(np.ceil(cy / cell))
    while cy + k * cell <= h + 1e-6:
        y = cy + k * cell
        if y >= -1e-6:
            lines.append(_hline(y, 0, w, steps))
        k += 1
    return lines, cell


def thirds_lines(w, h, steps=40):
    """3분할선: 가로·세로를 3등분하는 선 4개"""
    return [_vline(w * i / 3, 0, h, steps) for i in (1, 2)] + [_hline(h * j / 3, 0, w, steps) for j in (1, 2)]


def center_lines(w, h, rings=4, steps=90):
    """중심 십자선 + 동심원 (반지름 = 짧은 변 절반을 rings 등분)"""
    cx, cy = w / 2, h / 2
    lines = [_vline(cx, 0, h, steps), _hline(cy, 0, w, steps)]
    r_max = min(w, h) / 2
    t = np.linspace(0, 2 * np.pi, steps)
    for i in range(1, rings + 1):
        r = r_max * i / rings
        lines.append(np.stack([cx + r * np.cos(t), cy + r * np.sin(t)], 1))
    return lines


def diagonal_lines(w, h, steps=60):
    """대각선 2개 + 중심 십자"""
    d1 = np.stack([np.linspace(0, w, steps), np.linspace(0, h, steps)], 1)
    d2 = np.stack([np.linspace(0, w, steps), np.linspace(h, 0, steps)], 1)
    return [d1, d2, _vline(w / 2, 0, h, steps), _hline(h / 2, 0, w, steps)]


def distort_polylines(lines, calib):
    """
    반듯한 선(보정 화면 좌표) -> 정규화 좌표 -> 왜곡 모델 적용 -> 원본 화면 좌표.
    calib 가 없으면 그대로. 원본 화면에서 '렌즈가 직선을 휘게 보이는 모양'을 그릴 때 쓴다
    """
    if calib is None:
        return lines
    K, dist = calib["K"], calib["dist"]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    curved = []
    for pts in lines:
        obj = np.stack([(pts[:, 0] - cx) / fx, (pts[:, 1] - cy) / fy, np.ones(len(pts))], 1)
        img, _ = cv2.projectPoints(obj, np.zeros(3), np.zeros(3), K, dist)
        curved.append(img.reshape(-1, 2))
    return curved


def draw_polylines(view, scale, polylines, color=GRID_COLOR, thickness=1):
    for pts in polylines:
        p = np.round(pts * scale).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(view, [p], False, color, thickness, cv2.LINE_AA)


def draw_center_mark(view):
    h, w = view.shape[:2]
    cx, cy = w // 2, h // 2
    cv2.line(view, (cx - 15, cy), (cx + 15, cy), CENTER_COLOR, 1, cv2.LINE_AA)
    cv2.line(view, (cx, cy - 15), (cx, cy + 15), CENTER_COLOR, 1, cv2.LINE_AA)


def draw_label(view, text, color=GRID_COLOR):
    """영상 왼쪽 아래 작은 설명 (예: 격자 칸 크기)"""
    font, s, t = cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
    (tw, th), _ = cv2.getTextSize(text, font, s, t)
    h = view.shape[0]
    x, y = 8, h - 45
    roi = view[y - th - 6:y + 6, x - 4:x + tw + 4]
    view[y - th - 6:y + 6, x - 4:x + tw + 4] = (roi * 0.35).astype(view.dtype)
    cv2.putText(view, text, (x, y), font, s, color, t, cv2.LINE_AA)


def draw_grid(view, scale, polylines):
    h, w = view.shape[:2]
    for pts in polylines:
        p = np.round(pts * scale).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(view, [p], False, GRID_COLOR, 1, cv2.LINE_AA)
    # 영상 중심 십자선
    cx, cy = w // 2, h // 2
    cv2.line(view, (cx - 15, cy), (cx + 15, cy), CENTER_COLOR, 1, cv2.LINE_AA)
    cv2.line(view, (cx, cy - 15), (cx, cy + 15), CENTER_COLOR, 1, cv2.LINE_AA)


def board_line_errors(pts, board):
    """
    체커보드 행/열마다 양 끝 코너를 잇는 직선과, 각 코너가 그 직선에서 벗어난 거리.
    반환: [(시작점, 끝점, 코너들, 오차들), ...], 최대 오차, 평균 오차
    """
    cols, rows = board
    grid = pts.reshape(rows, cols, 2)
    lines = [grid[r] for r in range(rows)] + [grid[:, c] for c in range(cols)]
    out, all_err = [], []
    for line in lines:
        a, b = line[0], line[-1]
        d = b - a
        n = np.array([-d[1], d[0]]) / max(np.linalg.norm(d), 1e-9)  # 직선의 법선
        err = (line - a) @ n                                       # 부호 있는 거리
        out.append((a, b, line, err))
        all_err.extend(np.abs(err))
    all_err = np.array(all_err)
    return out, float(all_err.max()), float(all_err.mean())


def draw_banner(view, text, color):
    """영상 아래쪽 가운데에 판정 문구 표시"""
    font, s, t = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
    (tw, th), _ = cv2.getTextSize(text, font, s, t)
    h, w = view.shape[:2]
    x, y = (w - tw) // 2, h - 15
    roi = view[y - th - 8:y + 8, x - 8:x + tw + 8]
    view[y - th - 8:y + 8, x - 8:x + tw + 8] = (roi * 0.35).astype(view.dtype)
    cv2.putText(view, text, (x, y), font, s, color, t, cv2.LINE_AA)


def draw_board_guides(view, scale, pts, board, exaggerate=10):
    """
    체커보드 가이드 그리기. pts 는 원본 해상도 좌표.
    직선(노랑) + 코너 점(오차에 따라 초록/주황/빨강) + 벗어난 방향 화살표(오차를 exaggerate 배 확대)
    반환: (최대 오차, 평균 오차) px
    """
    lines, max_err, mean_err = board_line_errors(pts, board)
    for a, b, line, err in lines:
        d = (b - a) / max(np.linalg.norm(b - a), 1e-9)
        n = np.array([-d[1], d[0]])
        # 양 끝을 조금 늘려서 그린다
        p0 = tuple(np.round((a - d * 30) * scale).astype(int))
        p1 = tuple(np.round((b + d * 30) * scale).astype(int))
        cv2.line(view, p0, p1, LINE_COLOR, 1, cv2.LINE_AA)
        for p, e in zip(line, err):
            q = np.round(p * scale).astype(int)
            # 직선 위의 원래 자리에서 실제 코너 쪽으로 확대한 화살표
            base = np.round((p - n * e) * scale).astype(int)
            tip = np.round((p - n * e + n * e * exaggerate) * scale).astype(int)
            color = deviation_color(abs(e))
            if abs(e) >= GOOD_PX:
                cv2.arrowedLine(view, tuple(base), tuple(tip), color, 1, cv2.LINE_AA, tipLength=0.3)
            cv2.circle(view, tuple(q), 3, color, -1, cv2.LINE_AA)
    return max_err, mean_err
