# -*- coding: utf-8 -*-
"""저장 위치. 모두 프로젝트 폴더(mini_license_plate_recognition) 기준"""
import glob
import os

ROOT = PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROFILE_PATH = os.path.join(ROOT, "camera_profiles.json")            # 카메라별 스펙 (자동 생성·갱신)
CAPTURE_DIR = os.path.join(ROOT, "captures", "camera")               # 카메라 캡처 PNG (Space). '캡처로 보정' 이 여기서 읽음
REC_CAPTURE_DIR = os.path.join(ROOT, "captures", "recognition")      # 인식 결과 프레임 캡처
MODELS_DIR = os.path.join(ROOT, "models")                            # 번호판 감지·글자 인식 가중치 (plate_models.py 에 등록)
VIDEOS_DIR = os.path.join(ROOT, "videos")                            # 시험용 영상 파일 (인식 설정 창의 '영상 파일' 입력)
VIDEO_EXTS = ("*.mp4", "*.avi", "*.mkv", "*.mov")


def list_videos():
    out = []
    for ext in VIDEO_EXTS:
        out += glob.glob(os.path.join(VIDEOS_DIR, ext))
    return sorted(out)


def default_video():
    vids = list_videos()
    return vids[-1] if vids else None
