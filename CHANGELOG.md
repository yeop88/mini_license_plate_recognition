# Changelog

이 프로젝트의 바뀐 점을 적습니다. 형식은 [Keep a Changelog](https://keepachangelog.com/ko/1.1.0/) 를 따르고, 버전은 [Semantic Versioning](https://semver.org/lang/ko/) 을 씁니다.

## [Unreleased]

## [0.1.0] - 2026-11-20

카메라 앱과 인식 파이프라인을 하나로 합쳐 번호판 인식 프로그램의 첫 판을 만들었습니다.

### Added
- 카메라 번호판 인식 GUI (`main.py`, `run.bat`), MVVM 구조 (`app\models`, `app\viewmodels`, `app\views`).
- 카메라: 카메라 검색·전환, 해상도 선택(기본 / 최고 프레임 / 최고 화질 / 지원 해상도), 카메라별 스펙 측정·저장(`camera_profiles.json`), 캡처.
- 렌즈 왜곡: 체커보드 자동 인식, 왜곡 분석, 자동 보정, 보정 화면, 캡처로 보정, 카메라별 보정값(`calibration_<이름>.json`).
- 가이드선 7종, 초점 도우미(선명도 막대·포커스 피킹).
- 메인 창 도구 줄의 **[카메라 설정] / [인식 설정]** 버튼으로 여는 별도 설정 창 (스크롤, 두 창 함께 열기 가능, 같은 단축키).
- 인식 파이프라인: 카메라(보정 화면이면 왜곡을 편 영상) 또는 영상 파일 입력 → 번호판 감지 → 추적(화면 ID) → 글자 인식 → N번 읽어 다수결로 번호 고정.
- 번호판 모델 등록 자리 `app\models\plate_models.py` (감지기 `DETECTORS`, 글자 인식 `READERS`). 글자 인식 없이 감지·추적만도 가능.
- 방향에 상관없는 추적기 (IoU + 등속 예측, 가장자리 번호판은 다 보일 때까지 읽지 않음, 중복 감지 박스 제거).
- 한글 번호 표시 (맑은 고딕), 인식 결과 프레임 캡처 (`captures\recognition\`, 결과 jpg + `predictions.csv`).
- 영상 파일 입력의 재생바 (클릭 점프·드래그).

### Known issues
- 번호판 감지·글자 인식 모델이 아직 없습니다 (인식 시작 시 안내 메시지).
- 이 PC 의 카메라(ELP 5MP)가 1920x1080 에서 약 1 fps·검은 화면 (OpenCV 단독도 같음, USB 연결 문제로 보임). 640x480 은 60 fps 정상.

[Unreleased]: https://github.com/yeop88/mini_license_plate_recognition/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/yeop88/mini_license_plate_recognition/releases/tag/v0.1.0
