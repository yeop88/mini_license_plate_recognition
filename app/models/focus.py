"""
Model — 초점 맞추기 도우미 (수동 초점 렌즈용)

이 카메라(ELP 5MP 글로벌 셔터)는 렌즈를 움직이는 모터가 없어 오토포커스가 불가능하다
(드라이버는 초점 값을 받아들이지만 0~1023 을 바꿔도 선명도가 변하지 않음, 2026-11-13 실측).
그래서 사람이 렌즈 링을 돌릴 때 '지금 얼마나 선명한지'를 숫자·막대·색으로 알려 준다.

    FocusMeter.update(frame) : 측정 영역(가운데)의 선명도(라플라시안 분산)를 계산해 현재값·최고값·비율·판정 갱신.
                               최고점을 한 번 지나가기 전에는 '최고점 찾는 중'(지금 값이 곧 최고값이라 판정 불가)
    peaking_mask(gray)       : 초점이 맞은 윤곽(라플라시안이 큰 곳)만 골라낸 마스크 → 화면에서 색으로 칠함
"""

import time

import cv2
import numpy as np

ROI_FRACTION = 0.4        # 측정 영역: 가운데 가로·세로 40%
MEASURE_WIDTH = 480       # 측정 영역을 이 너비로 줄여서 계산 (해상도와 상관없이 빠르게)
SMOOTH = 0.6              # 현재값 평활 (노이즈·조명 깜빡임에 덜 흔들리게)
GOOD_RATIO, NEAR_RATIO = 0.95, 0.80   # 최고값 대비 비율: 이상이면 '초점 맞음' / '거의'
PASSED_RATIO = 0.90       # 최고값을 찍은 뒤 이 비율 아래로 내려가야 '최고점을 지나갔다'(판정 시작)
# 포커스 피킹 감도: 라플라시안 크기(2차 미분, 0~255 영상 기준) 이 값 이상만 칠함.
# 1차 미분(소벨)은 살짝 흐려져 넓어진 윤곽도 잡아서, 흐림에 더 민감한 2차 미분을 쓴다 (체커보드 시험)
PEAKING_LEVELS = {"높음": 40, "보통": 70, "낮음": 110}
PEAKING_THRESHOLD = PEAKING_LEVELS["보통"]


def roi_rect(w, h, fraction=ROI_FRACTION):
    rw, rh = int(w * fraction), int(h * fraction)
    x0, y0 = (w - rw) // 2, (h - rh) // 2
    return x0, y0, x0 + rw, y0 + rh


def sharpness(gray):
    """선명도 = 라플라시안 분산. 초점이 맞을수록 커진다"""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


class FocusMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.score = 0.0          # 평활한 현재 선명도
        self.raw = 0.0            # 이번 프레임 선명도
        self.peak = 0.0           # 지금까지의 최고값 (렌즈를 끝에서 끝까지 한 번 돌리면 최고점이 잡힘)
        self.trend = 0.0          # +면 좋아지는 중, -면 나빠지는 중
        self.passed = False       # 최고점을 한 번 지나갔는지 (그 전에는 지금 값이 곧 최고값이라 판정할 수 없음)
        self.updated = 0.0

    def update(self, frame):
        h, w = frame.shape[:2]
        x0, y0, x1, y1 = roi_rect(w, h)
        roi = frame[y0:y1, x0:x1]
        scale = min(1.0, MEASURE_WIDTH / roi.shape[1])
        if scale < 1.0:
            roi = cv2.resize(roi, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        self.raw = sharpness(gray)
        prev = self.score
        self.score = self.raw if prev == 0 else SMOOTH * prev + (1 - SMOOTH) * self.raw
        self.trend = 0.0 if prev == 0 else 0.7 * self.trend + 0.3 * (self.score - prev)
        self.peak = max(self.peak, self.score)
        if self.score < self.peak * PASSED_RATIO:
            self.passed = True
        self.updated = time.time()

    @property
    def ratio(self):
        return self.score / self.peak if self.peak > 0 else 0.0

    def verdict(self):
        """(영문 판정, 한글 판정)"""
        if not self.passed:
            return "SEARCHING", "최고점 찾는 중"
        r = self.ratio
        if r >= GOOD_RATIO:
            return "IN FOCUS", "초점 맞음"
        if r >= NEAR_RATIO:
            return "ALMOST", "거의 맞음"
        return "BLURRY", "흐림"


def peaking_mask(gray, threshold=PEAKING_THRESHOLD):
    """초점이 맞은 윤곽 마스크 (uint8 0/255). 흐리면 거의 비고, 초점이 맞을수록 윤곽이 많이 잡힌다"""
    lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
    mask = (lap >= threshold).astype(np.uint8) * 255
    return cv2.dilate(mask, np.ones((2, 2), np.uint8))
