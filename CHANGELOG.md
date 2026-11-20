# Changelog

이 프로젝트의 바뀐 점을 적습니다. 형식은 [Keep a Changelog](https://keepachangelog.com/ko/1.1.0/) 를 따르고, 버전은 [Semantic Versioning](https://semver.org/lang/ko/) 을 씁니다.

## [Unreleased]

### Added
- 번호판 감지기 `YOLO-v1s`: YOLO11s 번호판 감지 (Hugging Face morsetechlab/yolov11-license-plate-detection, AGPL-3.0). `models\plate_yolo\*.pt` 를 넣으면 크기별로 자동 등록.
- 글자 인식 `EOCR_KR`: EasyOCR 한글+영어, 번호판 글자만 허용하고 한국 번호판 형식(`12가3456`, `123가4567`, `서울12가3456`)에 맞을 때만 읽은 것으로 침. 두 줄 번호판 지원.
- 글자 인식 `EOCR_RAW`: EasyOCR 결과 그대로 (형식 검사 없음).
- 기본 모델: `YOLO-v1s` + `EOCR_KR`.

- 카메라 호환성 시험: 지원 모드를 전부 재고 검은 화면·느린 모드는 빼서 최고 화질·최고 프레임 결정 (`camera_compat_test.py`, [카메라 설정] → [호환성 시험 (스펙 다시 측정)]).
- 번호판 시험 도구: `plate_robustness_test.py` (거리·각도·흐림), `plate_types_test.py` (한국 번호판 12종).
- 문서 `docs/한국_번호판_데이터셋.md`: 무료 한국 번호판 학습 데이터 (AI Hub, Roboflow, GitHub 합성 생성기, RLPR).

- 번호판 감지기 `YOLO-v1l` (같은 저장소의 큰 모델, 먼 번호판을 더 잘 찾음). 기본은 `YOLO-v1s` 유지.

- 흐린 프레임 건너뛰기: 번호판 선명도를 재서 가장 선명했을 때의 절반보다 흐리면 읽지 않고, 다수결을 선명도로 가중 (`plate_enhance.py`).
- 글자 인식 전 보정 옵션: 대비(CLAHE)·선명화(언샤프). 합성 시험에서 효과가 없어 기본은 끔.
- 고정 후 추적 유지: 감지가 끊겨도 30 프레임 유지, 번호판 영상으로 따라가기(템플릿 매칭), 낮은 신뢰도 감지로 잇기, 가림에서 나온 번호판을 중심 거리로 잇기, 같은 번호로 이전 화면 ID·고정값 되살리기.
- 인식 설정 창에 [화질·추적 유지] 묶음 (위 기능을 하나씩 끄고 켬).
- 시험 도구 `plate_enhance_test.py`, `tracking_hold_test.py`.

### Changed
- 번호 고정을 **글자 자리별 다수결**로 바꿈: 같은 모양(길이·한글 위치)의 읽기끼리 묶고 자리마다 가장 많은 글자 → 매번 다른 자리 한 글자씩 틀려도 맞는 번호로 고정 (`tracker.vote_text`).
- `EOCR_KR` 형식 검사 확장: 외교(`외교001001` 등 7종), 구형 이륜차(`서울강남가1234`, 시·군·구 이름), 지역명 한 글자 오류 고침(`서문` → `서울`). 지역명 뒤에는 숫자 2자리만 허용.
- **번호판 감지 입력 크기 640 → 1280**: 640 에서는 한국 번호판을 거의 못 찾음 (합성 시험 12종 x 4조건 감지 8/48 → 34/48, 정면 3 m → 6 m, 감지 시간 +2.5 ms).
- `requirements.txt` 에 torch, ultralytics, easyocr, huggingface_hub 추가.
- 스펙 측정(`probe_camera`)이 일부 해상도만 재던 것을 모든 지원 모드 시험으로 바꿈. 처음 여는 모드는 기본 해상도.

### Fixed
- 카메라가 약 40 fps 로 묶이던 문제: 열 때·모드를 바꿀 때 게인 값을 한 번 읽음 (`wake_frame_rate`, 값은 바꾸지 않음) → 1920x1080 60 fps.

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
- 번호판 감지·글자 인식 모델이 아직 없습니다 (인식 시작 시 안내 메시지). → [Unreleased] 에서 추가.
- 이 PC 의 카메라(ELP 5MP)가 1920x1080 에서 약 1 fps·검은 화면 (OpenCV 단독도 같음, USB 연결 문제로 보임). 640x480 은 60 fps 정상.

[Unreleased]: https://github.com/yeop88/mini_license_plate_recognition/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/yeop88/mini_license_plate_recognition/releases/tag/v0.1.0
