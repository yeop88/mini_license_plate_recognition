# -*- coding: utf-8 -*-
r"""
카메라 번호판 인식 GUI — 실행 진입점 (MVVM)

    Model     : app\models      카메라(camera, camera_profile, focus, lens_distortion)
                                 인식(plate_models, tracker, render, frame_saver, pipeline)
    ViewModel : app\viewmodels  camera_viewmodel (카메라 상태·명령), recognition_viewmodel (인식 상태·명령)
    View      : app\views       main_window (영상 + 도구 줄 + 메시지), camera_panel·recognition_panel (별도 설정 창)

실행 (콘솔 창 없이):
    pythonw main.py                          # 카메라로 시작 (지난번 카메라·해상도). 인식은 [▶ 인식 시작] 또는 I
    pythonw main.py --camera 1 --size 640x480
    pythonw main.py --video videos\xxx.mp4   # 영상 파일로 시작 (감지기가 준비되면 바로 재생)
    pythonw main.py --detector P1 --reader R1   # plate_models.py 에 등록한 모델 키 (생략하면 첫 번째 등록 모델)
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tkinter as tk  # noqa: E402

from app.models import plate_models as PM  # noqa: E402
from app.models.camera_profile import PRESETS, ProfileRepository, parse_size  # noqa: E402
from app.viewmodels.camera_viewmodel import CameraViewModel  # noqa: E402
from app.viewmodels.recognition_viewmodel import RecognitionViewModel  # noqa: E402
from app.views.main_window import MainWindow  # noqa: E402


def parse_board(text):
    if text.lower() == "auto":
        return None
    try:
        cols, rows = (int(v) for v in text.lower().split("x"))
        return cols, rows
    except ValueError:
        raise argparse.ArgumentTypeError("체커보드 크기는 9x6 처럼 입력하세요 (내부 코너 수) 또는 auto")


def parse_size_arg(text):
    if parse_size(text) is None:
        raise argparse.ArgumentTypeError("해상도는 2048x1536 처럼 입력하세요")
    return text.lower()


def main():
    ap = argparse.ArgumentParser(description="카메라 번호판 인식 GUI: 카메라 영상 + 렌즈 보정 + 번호판 감지·추적·글자 인식 (MVVM)")
    g = ap.add_argument_group("카메라")
    g.add_argument("--camera", type=int, default=None, help="시작할 카메라 번호 (생략하면 지난번 카메라)")
    g.add_argument("--width", type=int, default=1920, help="해상도 목록의 '기본' 너비 (기본 1920)")
    g.add_argument("--height", type=int, default=1080, help="해상도 목록의 '기본' 높이 (기본 1080)")
    g.add_argument("--size", type=parse_size_arg, default=None, help="이 해상도로 열기 (예: 2048x1536)")
    g.add_argument("--mode", choices=PRESETS, default=None,
                   help="request = 기본 해상도, max_fps = 최고 프레임, max_quality = 최고 화질")
    g.add_argument("--board", type=parse_board, default=None, help="체커보드 내부 코너 수 (예: 9x6, 기본 자동 인식)")
    g = ap.add_argument_group("인식")
    g.add_argument("--video", default=None, help="영상 파일로 시작 (생략하면 카메라 입력)")
    g.add_argument("--detector", default=None, choices=list(PM.DETECTORS) or None,
                   help="번호판 감지기 키 (plate_models.py 에 등록, 생략하면 첫 번째)")
    g.add_argument("--reader", default=None, choices=list(PM.READERS) or None,
                   help="글자 인식 키 (plate_models.py 에 등록, 생략하면 첫 번째)")
    a = ap.parse_args()

    cam_vm = CameraViewModel(ProfileRepository(), a.camera, a.width, a.height, a.board, resolution=a.size or a.mode)
    rec_vm = RecognitionViewModel(cam_vm, a.detector, a.reader, source="video" if a.video else "camera",
                                  video=os.path.abspath(a.video) if a.video else None, auto_play=bool(a.video))
    root = tk.Tk()
    MainWindow(root, cam_vm, rec_vm)
    root.mainloop()


if __name__ == "__main__":
    main()
