# 외부 구성요소 라이선스·출처

이 저장소의 코드는 [GNU AGPL-3.0](LICENSE) 입니다. 아래 구성요소는 저장소에 들어 있지 않고(패키지·가중치는 각자 설치·다운로드), 각자의 라이선스를 따릅니다.

## 왜 AGPL-3.0 인가

번호판 감지는 [Ultralytics](https://github.com/ultralytics/ultralytics)(AGPL-3.0) 를 불러 쓰고, 기본 감지 가중치(morsetechlab, AGPL-3.0)도 AGPL-3.0 입니다. 이 프로그램은 둘과 함께 동작하도록 만들어졌으므로 같은 AGPL-3.0 으로 공개합니다.
이 프로그램을 고쳐서 배포하거나 네트워크 서비스로 제공하면 고친 소스도 AGPL-3.0 으로 공개해야 합니다. 상업적으로 소스 공개 없이 쓰려면 Ultralytics 기업 라이선스와 가중치 저작자의 허락이 따로 필요합니다.

## Python 패키지 (`requirements.txt`, 직접 설치)

| 패키지 | 확인한 버전 | 라이선스 | 쓰는 곳 |
| --- | --- | --- | --- |
| [ultralytics](https://github.com/ultralytics/ultralytics) | 8.4.193 | AGPL-3.0 | 번호판 감지 (YOLO), TensorRT 엔진 만들기 |
| [EasyOCR](https://github.com/JaidedAI/EasyOCR) | 1.7.2 | Apache-2.0 | 번호판 글자 인식 |
| [PyTorch](https://github.com/pytorch/pytorch) | 2.14.1 | BSD-3-Clause 계열 | 모델 실행 |
| [OpenCV (opencv-python)](https://github.com/opencv/opencv-python) | 5.0.0 | Apache-2.0 | 카메라·영상 처리 |
| [NumPy](https://numpy.org/) | 2.4.6 | BSD-3-Clause | 배열 계산 |
| [Pillow](https://python-pillow.org/) | 12.3.0 | MIT-CMU (HPND) | 한글 글자 그리기 |
| [pygrabber](https://github.com/andreaschiavinato/python_grabber) | 0.2 | MIT | (선택) Windows 카메라 이름 |
| [huggingface_hub](https://github.com/huggingface/huggingface_hub) | 2.4.0 | Apache-2.0 | (선택) 가중치 받기 |
| [TensorRT](https://developer.nvidia.com/tensorrt) | 10.16 | NVIDIA 소프트웨어 라이선스 | (선택) 감지기 FP16 엔진. 엔진 파일은 저장소에 넣지 않음 |

## 모델 가중치 (저장소에 넣지 않음, `models\` 에 직접 받음)

| 가중치 | 출처 | 라이선스 |
| --- | --- | --- |
| `license-plate-finetune-v1s.pt`, `-v1l.pt` (YOLO11 번호판 감지) | Hugging Face [morsetechlab/yolov11-license-plate-detection](https://huggingface.co/morsetechlab/yolov11-license-plate-detection) | AGPL-3.0 |
| `korean_g2.pth`, `craft_mlt_25k.pth` (EasyOCR 한글 인식·글자 위치) | EasyOCR 가 처음 실행할 때 자동으로 받음 (`%USERPROFILE%\.EasyOCR\model\`) | Apache-2.0 (EasyOCR 배포 조건) |

`export_trt.py` 로 만든 `*.engine` 은 위 YOLO 가중치에서 나온 것이므로 같은 AGPL-3.0 조건을 따릅니다.

## 글꼴

화면의 한글 표시와 시험용 번호판 그림에 Windows 의 맑은 고딕(`C:\Windows\Fonts\malgun*.ttf`)을 씁니다. 글꼴 파일은 저장소에 넣지 않으며 Microsoft 의 Windows 라이선스를 따릅니다.

## 데이터

학습·시험 데이터(`datasets\`, AI Hub·RLPR 등)는 저장소에 넣지 않습니다. 출처와 라이선스는 `docs\한국_번호판_데이터셋.md` 에 정리했습니다. 시험 도구의 번호판 영상은 프로그램이 직접 그린 합성 이미지이며 실제 번호와 관계없습니다.
