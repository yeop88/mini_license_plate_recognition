# -*- coding: utf-8 -*-
"""
plate_enhance (Model) — 잘라낸 번호판의 선명도 재기·가벼운 보정 (학습 없음, 번호판 1장에 1 ms 안팎)

선명도 (sharpness)
    번호판을 높이 SHARP_H 로 맞춘 뒤 라플라시안 분산 (값이 클수록 선명).
    크기를 맞추므로 거리가 달라도 비교할 수 있고, 줄일 때 평균이 되어 잡음의 영향이 줄어듦.
    추적 중인 같은 번호판끼리 비교하는 데 씀 (절대 기준값은 장면마다 다름):
        지금 프레임 선명도 < 그 번호판의 가장 선명했던 값 x SKIP_RATIO → 읽지 않음 (흐린 프레임 건너뜀)
        다수결 가중치 = 선명도 / 그 번호판의 가장 큰 선명도

보정 (enhance) — 글자 인식 직전, 잘라낸 번호판에만
    1) 키우기 : 높이 OCR_H 보다 작으면 키움 (글자 인식기가 작은 글자를 잘 못 읽음)
    2) 대비   : CLAHE (밝기 채널만, 어둡거나 역광인 번호판)
    3) 선명화 : 언샤프 마스크 (약하게, 너무 세면 글자 둘레가 깨져 오히려 못 읽음)
    없는 정보를 만들어 내지 않는 고전 영상 처리만 씀 (학습형 초해상도는 글자를 지어낼 수 있어 쓰지 않음)
"""
import cv2

SHARP_H = 32              # 선명도를 잴 때 맞추는 높이 (px)
SKIP_RATIO = 0.5          # 가장 선명했던 값의 이 비율보다 흐리면 읽지 않음
OCR_H = 96                # 글자 인식 전에 이 높이까지 키움 (plate_models.OCR_MIN_HEIGHT 와 같게)
CLAHE_CLIP = 2.0
UNSHARP_SIGMA = 1.5       # OCR_H 높이 기준
UNSHARP_AMOUNT = 0.8

_clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=(2, 6))


def sharpness(crop):
    """잘라낸 번호판 BGR → 선명도 (라플라시안 분산, 높이 SHARP_H 기준). 빈 영상이면 0"""
    if crop is None or crop.size == 0:
        return 0.0
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    h, w = g.shape[:2]
    s = SHARP_H / max(h, 1)
    g = cv2.resize(g, (max(1, int(round(w * s))), SHARP_H),
                   interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    return float(cv2.Laplacian(g, cv2.CV_32F).var())


def enhance(crop, contrast=True, sharpen=True):
    """잘라낸 번호판 BGR → 보정한 BGR (키우기 + 대비 + 선명화). 둘 다 끄면 키우기만"""
    if crop is None or crop.size == 0:
        return crop
    h, w = crop.shape[:2]
    if h < OCR_H:
        s = OCR_H / h
        crop = cv2.resize(crop, (int(w * s), OCR_H), interpolation=cv2.INTER_CUBIC)
    if not (contrast or sharpen):
        return crop
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0]
    if contrast:
        L = _clahe.apply(L)
    if sharpen:
        sigma = UNSHARP_SIGMA * crop.shape[0] / OCR_H
        blur = cv2.GaussianBlur(L, (0, 0), sigma)
        L = cv2.addWeighted(L, 1 + UNSHARP_AMOUNT, blur, -UNSHARP_AMOUNT, 0)
    lab[:, :, 0] = L
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def weights(sharps):
    """선명도 목록 → 다수결 가중치 (가장 선명한 것 = 1). 모두 0 이면 모두 1"""
    m = max(sharps) if sharps else 0.0
    return [1.0 if m <= 0 else max(s / m, 1e-3) for s in sharps]

