# Ubuntu · RTX 5070 · Isaac Sim 5.1

사용자가 지정한 실행 환경이다. 시뮬레이터 Python과 학습용 CUDA 환경을 분리한다.
이 저장소는 Isaac Sim을 자동으로 설치하거나 GPU 드라이버를 변경하지 않는다.

## 1. 실행 환경과 패키지

Isaac Sim 5.1의 내장 Python은 3.11이다. 아래 `ISAAC_PY`를 실제 설치 위치로 바꾼다.

```bash
cd track-digital-twin
export ISAAC_PY="$HOME/isaacsim/python.sh"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
"$ISAAC_PY" --version
"$ISAAC_PY" -m pip install -r requirements-sim.txt
"$ISAAC_PY" run_sim.py --check
```

`--check`는 이미지 해상도와 JSON 치수가 맞는지 검사하고 차량·카메라 설정과 GPU/VRAM을 출력한다.
현재 트랙은 **2.355 × 2.4m**, 이미지 **2355 × 2400px**다.
RTX 5070이라는 이름만으로 노트북 VRAM을 가정하지 않는다. 렌더 메모리가 부족하면 `configs/rtx.json`에서
카메라를 320×240, 뷰포트를 640×360으로 줄인 사본을 만든 뒤 `--config`로 지정한다.
기존 데이터와 해상도 설정을 섞기 전에 모델 입력·검출 라벨을 다시 확인한다.

실행 API 기준은 5.1이며, 6.x 분기는 호환을 위한 구현만 포함하고 실제 호환성을 검증하지 않았다.
설치/드라이버 요구사항은 [NVIDIA 5.1 문서](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html),
Python 환경은 [공식 설명](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_python.html)을 따른다.

## 2. 먼저 수동 주행과 카메라 확인

```bash
"$ISAAC_PY" run_sim.py --mode manual --seed 1
```

OpenCV 카메라 창에서 W/S 전진·후진, A/D 제자리 회전, 스페이스 정지, Q 종료.
키 입력이 끊기면 0.3초 후 명령이 정지로 돌아간다. 실제 지연 시간은 화면 프레임 처리 속도의 영향을 받는다.
바퀴가 뜨거나 회전하지 않으면 학습 데이터를 수집하기 전에 바퀴 접촉·조인트·질량·토크를 확인한다.

외부 Nucleus/Jetbot 에셋을 다운로드하지 않고 USD 도형으로 4륜 차량, 표지판, 신호등을 생성한다.
차량 +X 전방, +Y 왼쪽, +Z 위쪽. USD 조인트 속도는 deg/s이므로 rad/s 명령을 변환한다.
카메라의 RGB는 학습 전처리에서 YUV로 변환하며, OpenCV 표시에서만 BGR로 바꾼다.

## 3. 기하학 교사로 장면 검증

```bash
"$ISAAC_PY" run_sim.py --mode expert --duration 90 --seed 1000 --record
"$ISAAC_PY" run_sim.py --mode expert --route inner --duration 90 --seed 1001 --record
```

`expert`는 위치 정답과 중심선을 사용하는 **데이터 수집용 교사**다. 카메라 자율주행 모델이 아니다.
`outer`는 바깥 순환, `inner`는 가운데 도로와 교차로를 거치는 예제 경로다. 교차로 회전 가능성은 차량 치수와
실제 마찰을 보정한 뒤 확인한다. 교사도 이탈하면 정책 학습보다 경로·물리·카메라 보정이 먼저다.

결과는 `outputs/expert-outer-seed1000/` 등에 저장한다. 같은 폴더는 덮어쓰지 않는다.

| 파일 | 내용 |
|---|---|
| `episode.json` | 설정, 난수 시드, 조명·마찰·모터·카메라 무작위화 값 |
| `frames.jsonl` | 카메라 프레임 뒤에 적용할 속도/각속도, 위치, 도로 여유, 검출 정답 |
| `images/` | `--record`로 저장한 카메라 JPEG |
| `summary.json` | 완주·이탈·최소 여유·경로 오차 또는 오류 |
| `scene.usda`, `assets/` | 디버깅용 장면과 절차적으로 만든 표지판. 경로는 해당 기기의 절대 경로 |

차체와 바퀴를 감싸는 보수적인 사각형의 가장자리를 1mm 해상도 도로 마스크와 대조한다.
`min_clearance_m=0`은 도로 경계 밖 샘플이 있음을 뜻한다. 충돌 센서나 정확한 타이어 외곽 측정을 대신하지 않는다.

## 4. 강건성 학습용 데이터 수집

```bash
python3 run_batch.py --python "$ISAAC_PY" --mode collect --count 30 --duration 90 --output outputs/train
python3 run_batch.py --python "$ISAAC_PY" --mode collect --route inner --seed-start 2000 --count 10 --duration 90 --output outputs/train
```

각 에피소드는 새 프로세스에서 실행한다. 시드로 조명 밝기·색온도, 바닥 색·반사/마찰,
좌우 모터 편차, 카메라 높이/각도/화각, 출발 오차, 영상 밝기/잡음/지연, 소품 위치·방향을 바꾼다.
영상 지연이 생겨도 객체 라벨은 **그 영상이 촬영된 프레임**과 함께 이동한다.
기본 무작위화 범위는 가정이며, 실제 310호/대강당 영상 분포로 교체해야 한다.

## 5. 별도 CUDA 환경에서 학습

Python 3.11 이상의 별도 환경을 만들고, [PyTorch 공식 설치 선택기](https://pytorch.org/get-started/locally/)에서
Linux/Pip/CUDA 조합을 골라 RTX 5070을 지원하는 빌드를 설치한다. Isaac의 번들 torch를 학습 환경에 덮어쓰지 않는다.

```bash
python3.11 -m venv .venv-train
source .venv-train/bin/activate
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-train.txt
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print(torch.cuda.get_device_name(0)); print(torch.ones(1, device="cuda") * 2)'
python train_policy.py outputs/train --output models/lane --epochs 30 --batch-size 64 --device cuda
```

CUDA 12.8 빌드는 예시다. 설치 시점 공식 빌드·드라이버 조합과 위 GPU 텐서 연산의 성공을 먼저 확인한다.
메모리가 부족하면 배치를 줄이고, 시뮬레이터와 학습을 동시에 실행하지 않는다.

학습 코드는 RGB 하단 절반 → YUV → 3×3 블러 → 200×66 → CHW/255 전처리를 공유한다.
작은 CNN이 선속도와 각속도를 예측하며, 곡선 프레임 가중치와 밝기·채널·잡음·좌우 반전을 사용한다.
데이터는 **장면 시드 단위**로 train/validation/test를 분리한다. 인접 프레임이나 같은 시드의 다른 주행이 분할을 넘지 않는다.
가장 좋은 validation 모델을 ONNX로 내보내고 test 오차를 기록한다. 이 오차만으로 강건하다고 판정하지 않는다.

## 6. 보지 않은 조건에서 카메라 정책 평가

```bash
python3 run_batch.py --python "$ISAAC_PY" --mode evaluate --policy models/lane/policy.onnx --count 20 --duration 90 --output outputs/eval
python3 run_batch.py --python "$ISAAC_PY" --mode evaluate --policy models/lane/policy.onnx --seed-start 10000 --count 20 --duration 90 --stress --output outputs/stress
python3 evaluate_runs.py outputs/eval
python3 evaluate_runs.py outputs/stress
```

학습에 사용한 모든 시드와 평가 시드가 겹치면 실행을 거부한다. ONNX 정책은 카메라 영상만 사용하고,
위치 정답은 평가 지표 계산에만 사용한다. 오류·이탈 에피소드도 분모에 포함한다.
성공은 최소 1바퀴 완주하고 도로 밖으로 나간 프레임이 없는 경우다. 다른 경로와 실제 영상·실차에서도 별도 평가한다.
정상 조건과 스트레스 조건을 나눠 완주율, 이탈 프레임, 최소 도로 여유를 비교한다.

## 7. 표지판·신호등 검출기

```bash
python make_detector_dataset.py outputs/train --output datasets/signs
python train_detector.py datasets/signs/data.yaml --batch-size 8 --device 0
```

클래스는 경적, STOP, 감속20, 빨강/노랑/초록 신호, 좌/우/직진의 9종이다. 렌더러의 보이는 표면 bbox로 YOLO 라벨을 생성한다.
학습 데이터에 클래스가 빠져 있으면 소품 배치·경로·카메라 각도를 바꾸어 더 수집한다.
절차적 표지판과 사진의 종이 주름·반사·배경 차이가 있으므로, 실차 카메라에서 촬영하고 라벨링한 영상도 섞어야 한다.
임의 사진 한 장으로 검출 성능을 보장할 수 없다. 학습 API는 [Ultralytics 공식 문서](https://docs.ultralytics.com/modes/train/)를 따른다.

검출·차선 모델과 미션 상태기계는 `--missions --detector ...`로 연결한다.
[미션 제어 통합 안내](autonomy.md)에 실행 명령, 추적·거리·교차로 제어, 카메라/바퀴 보정과 성공 집계를 정리했다.

## 8. 기존 8_4.py 연결

교재 파일과 `.h5` 모델은 노트북에 별도로 준비한다. 공개 저장소에는 포함하지 않는다.
TensorFlow는 이 모드에서만 필요하며 Isaac Python 3.11과 호환되는 환경에서 먼저 H5 로드를 확인한다.

```bash
"$ISAAC_PY" run_sim.py --legacy-script /absolute/path/8_4.py --legacy-model /absolute/path/lane_navigation_final.h5 --seed 500 --duration 90
```

이 브리지는 카메라와 GPIO를 교체하고 H5 경로를 전달한다. 카메라 원본의 180도 뒤집힘도 맞춘다.
카메라 창에서 W/위쪽 화살표 시작, S/아래쪽 정지, Q 종료. TensorFlow 추론은 CPU를 사용한다.
실차 PWM 0.4와 가상 바퀴 속도의 관계는 아직 실측 보정 전이다.

## 9. 검증 범위

현재 개발 환경에서 치수·기하·카메라 좌표·전처리·데이터 분리·USD 구조와 검출→추적→거리→미션 주행 통합을 검사한다.
**Ubuntu RTX 5070에서 Isaac Sim 5.1 실제 렌더링, PhysX 접촉/주행, 완주율, GPU 메모리 및 학습 품질은 아직 검증하지 않았다.**
RTX에서 처음 실행할 때 `--check`, 수동 주행, 교사 주행의 결과와 `summary.json`부터 확인한다.
