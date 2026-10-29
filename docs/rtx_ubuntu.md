# Ubuntu · RTX 5070 · Isaac Sim 5.1

사용자가 지정한 실행 환경이다. 시뮬레이터 Python과 학습용 CUDA 환경을 분리한다.
아직 저장소를 받지 않았어도 아래 0번부터 진행할 수 있다. 에이전트에게 맡기려면
[시작 프롬프트](../START_HERE_PROMPT.md) 전체를 대화 입력창에 붙여 넣는다.
저장소의 실행 스크립트 자체에는 Isaac Sim 설치·GPU 드라이버 변경 기능이 없으므로,
에이전트가 설치 상태를 확인하고 아래 절차로 준비한다.

## 0. Git 확인과 저장소 클론

Ubuntu에서 터미널을 열고 다음을 실행한다. 작업 에이전트가 있다면 에이전트가 직접 실행한다.

```bash
pwd
git --version
```

`git: command not found`가 나오면 Git과 다운로드/압축 해제에 필요한 도구를 설치한다.
관리자 비밀번호가 필요하면 Ubuntu 로그인 비밀번호를 해당 터미널에 입력한다. 비밀번호를 대화에 보내지 않는다.

```bash
sudo apt-get update
sudo apt-get install -y git ca-certificates unzip
```

`$HOME/projects/track-digital-twin`이 **없는 경우** 다음 명령을 실행한다.
`$HOME`은 Ubuntu 사용자 홈 경로이므로 사용자 이름을 따로 바꿀 필요가 없다.

```bash
mkdir -p "$HOME/projects"
cd "$HOME/projects"
git clone --branch feat/isaac-sim-rtx5070 --single-branch https://github.com/gawbi/track-digital-twin.git
cd track-digital-twin
git status --short --branch
```

마지막 출력에 `feat/isaac-sim-rtx5070`이 보이면 맞는 브랜치다. 현재 실행·학습 코드는 아직 `main`에 병합되지 않았다.
같은 폴더가 이미 있으면 먼저 그 폴더의 `git status`와 `git remote -v`를 확인한다.
동일 저장소이며 변경이 없을 때만 fetch 후 대상 브랜치를 `--ff-only`로 갱신한다.
다른 프로젝트나 수정 중인 폴더이면 파일을 보존하고 새 이름의 폴더에 clone한다.
기존 폴더 삭제, hard reset, 자동 stash로 충돌을 피하지 않는다.

클론 뒤에는 [상세 인계](../RTX_HANDOFF_PROMPT.md), [이슈 #2](https://github.com/gawbi/track-digital-twin/issues/2),
[후속 구현 요구사항](real_vehicle_plan.md)을 읽는다. GitHub 이슈에 접근할 수 없어도 로컬 요구사항 문서로 진행할 수 있다.

## 1. 실행 환경과 패키지

이후 명령은 클론한 `track-digital-twin` 폴더에서 실행한다. 먼저 설치 전제와 여유 공간을 확인한다.

```bash
lsb_release -a
uname -m
df -h .
free -h
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
```

`nvidia-smi`가 없거나 GPU를 읽지 못하면 그 오류를 먼저 진단한다. 기존 드라이버를 무조건 재설치하지 않는다.
기기와 드라이버는 [NVIDIA 5.1 요구사항](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html)과 대조한다.
RTX 5070 이름만으로 지원·VRAM·실행 성공을 단정하지 않고 작은 장면부터 확인한다.

### 이미 Isaac Sim 5.1이 설치된 경우

설치 폴더의 `python.sh` 위치를 찾는다. 기존 안내의 `$HOME/isaacsim/python.sh`는 예시 경로다.

```bash
find "$HOME" -maxdepth 5 -type f -name python.sh -path '*isaac*' -print 2>/dev/null
```

발견한 실제 경로를 `ISAAC_PY`로 지정한다. 예를 들어 `$HOME/isaacsim/python.sh`가 존재한다면:

```bash
export ISAAC_PY="$HOME/isaacsim/python.sh"
"$ISAAC_PY" --version
```

Isaac Sim 5.1의 Python은 3.11이다. Python 버전만으로 Isaac 버전을 확정하지 말고 설치 정보/런처도 확인한다.
pip로 설치한 경우 `python.sh`가 없으므로 그 가상환경의 `bin/python`을 `ISAAC_PY`로 쓴다.
2026-10-29 실행 기기에서는 `ISAAC_PY=$HOME/isaacsim_env/bin/python`(isaacsim 5.1.0.0, Python 3.11.16)이었다.
검색에 나오지 않으면 다른 디스크의 설치 위치와 기존 가상환경도 확인한다.

### Isaac Sim이 아직 없는 경우

공식 5.1 설치 방식을 선택해 **설치까지 진행**한다. 아래 두 방식을 중복 설치할 필요는 없다.

- **Standalone:** [공식 설치 안내](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_workstation.html)에서
  5.1.0의 기기 아키텍처에 맞는 패키지를 받아 새 폴더에 압축 해제하고 `post_install.sh`를 실행한다.
  기존 설치 폴더 위에 압축을 풀지 않는다. 그 폴더의 `python.sh`를 `ISAAC_PY`로 지정한다.
- **전용 Python 환경:** [공식 Python 설치 안내](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_python.html)에
  따라 Python 3.11 환경에 정확히 5.1.0을 설치한다. 아래는 `python3.11`이 이미 준비된 경우의 명령이다.

```bash
python3.11 --version
python3.11 -m venv .venv-isaac
.venv-isaac/bin/python -m pip install --upgrade pip
.venv-isaac/bin/python -m pip install 'isaacsim[all,extscache]==5.1.0' --extra-index-url https://pypi.nvidia.com
export ISAAC_PY="$PWD/.venv-isaac/bin/python"
```

Python 3.11이 없으면 에이전트가 Ubuntu 버전과 기존 환경 관리 도구를 확인해 전용 3.11 환경을 준비한다.
시스템 기본 Python의 링크를 바꾸거나 기존 `.venv-isaac`을 덮어쓰지 않는다.
NVIDIA 로그인·라이선스 동의·관리자 권한이 실제로 필요하면 필요한 조치만 사용자에게 요청하고,
사용자의 동의를 대신 입력하지 않는다. 독립적으로 가능한 저장소 코드 작업은 계속 진행한다.
설치 로그에 나타난 버전·Python 실행 파일·설치 경로를 남긴다.

### 프로젝트 패키지와 첫 검사

위에서 정한 실행 파일로 프로젝트 의존성을 설치한다. 매 터미널/에이전트 실행에서 `ISAAC_PY`를 다시 지정하거나
기록한 실제 경로를 사용한다. 학습용 torch는 별도 환경인 5번에서 설치한다.

```bash
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

짧은 실제 렌더링도 확인한다. 이 명령은 화면 없이 실행하고 RGB와 결과를 저장한다.
같은 출력 폴더가 이미 있으면 다른 경로를 사용한다.

```bash
"$ISAAC_PY" run_sim.py --mode expert --headless --duration 10 --seed 1000 --record --output outputs/first-smoke
```

`outputs/first-smoke/expert-outer-seed1000/summary.json`과 `images/`를 확인한다.
`--check` 통과만으로 렌더링·물리 주행이 성공한 것은 아니다. 처음 셰이더를 준비하는 동안은 로그와 GPU 사용량을 확인한다.
렌더링·카메라·바퀴 접촉 오류는 원인을 수정하고 짧게 재실행한 뒤 다음 단계로 간다.

## 2. 먼저 수동 주행과 카메라 확인

```bash
"$ISAAC_PY" run_sim.py --mode manual --duration 15 --seed 1 --output outputs/first-manual
```

OpenCV 카메라 창에서 W/S 전진·후진, A/D 제자리 회전, 스페이스 정지, Q 종료.
위 명령은 시뮬레이션 시간 15초 뒤 종료한다. 사용자가 더 오래 조작하려면 `--duration` 값을 늘리거나 제거한다.
GUI를 사용할 수 없는 에이전트 세션에서는 이 단계를 미실행으로 기록하고 앞의 headless 결과로 다음 작업을 진행한다.
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

### 차량 구동 모델 (`simulation.drive_model`)

2026-10-29 RTX 실행에서 `tools/physics_probe.py`로 PhysX 강체 바퀴를 측정한 결과, 직진·부호·단위·접지는 정상이지만
**전진하면서 도는 스키드 회전은 명령의 3–9%만 재현**되었다(실린더 바퀴는 회전 중 차체가 넘어짐, 구 바퀴·마찰·토크·솔버
반복·축간을 바꿔도 동일). 쿨롱 마찰 계산으로는 이 차량 형상의 회전 효율이 약 0.46이어야 한다.
그래서 수집·평가 기본값은 `kinematic_skid`다. 좌우 바퀴 목표 → 모터 편차·1차 지연·포화 → 회전 효율 η로 차체를 움직이고,
엔코더 값은 목표가 아닌 모델 바퀴 회전율이다. η·시간상수는 [추정]이며 무작위화한다.
`physx`는 진단용으로 남겼다. 실차 측정(`tools/motion_calibration.py`)으로 η와 duty당 속도를 보정한다.

```bash
"$ISAAC_PY" tools/physics_probe.py --output outputs/probe-kinematic
"$ISAAC_PY" tools/physics_probe.py --output outputs/probe-physx --set 'simulation.drive_model="physx"'
```

교사는 급회전 꼭짓점(교차로)까지 직진한 뒤 제자리 회전하고, 실현된 요 변화로 미끄러짐 비율과 좌우 모터 편차를 보상한다.
inner 경로의 T자 회전점은 바깥 도로 중심선에서 2.5cm 안쪽이다(차량 회전 반경 0.148m > 바깥 도로 반폭 0.13m).

### 카메라 (`camera`)

카메라 높이·기울기·화각·왜곡은 2026-10-14 차량 카메라 사진에서 추정한 값이다(높이 약 7cm, 8.5° 위, 중심 fx≈330px,
k1≈-0.3). 렌즈 왜곡이 있으면 더 넓은 핀홀 영상을 2배 해상도로 렌더하고 `dtwin/lens.py`로 왜곡 영상과 라벨을 함께 변환한다.
실제 사진과 같은 위치에서 비교하려면:

```bash
"$ISAAC_PY" tools/render_view.py --output outputs/views --pose 0 0.45 -90 --pose -0.4 0.8725 0
```

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
python train_policy.py outputs/train --output models/lane --epochs 30 --batch-size 64 --device cuda --frame-stride 2
python evaluate_lane_offline.py models/lane/policy.onnx --synthetic outputs/train --split validation
```

CUDA 12.8 빌드는 예시다. 설치 시점 공식 빌드·드라이버 조합과 위 GPU 텐서 연산의 성공을 먼저 확인한다.
메모리가 부족하면 배치를 줄이고, 시뮬레이터와 학습을 동시에 실행하지 않는다.

학습 코드는 RGB 하단 절반 → YUV → 3×3 블러 → 200×66 → CHW/255 전처리를 공유한다.
작은 CNN이 선속도와 각속도를 예측하며, 곡선 프레임 가중치와 밝기·채널·잡음·좌우 반전을 사용한다.
데이터는 **장면 시드 단위**로 train/validation/test를 분리한다. 인접 프레임이나 같은 시드의 다른 주행이 분할을 넘지 않는다.
가장 좋은 validation 모델을 ONNX로 내보낸다. **학습 스크립트는 test를 평가하지 않는다.** 설정을 고정한 뒤
`evaluate_lane_offline.py --split test --final`로 한 번 평가하며, 접근할 때마다 `test_access.jsonl`에 기록된다.
오프라인 오차만으로 강건하다고 판정하지 않는다. 실제 데이터 미세조정은 [후속 구현](real_vehicle_plan.md#구현-현황-2026-10-29)을 따른다.

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
python make_detector_dataset.py outputs/train --output datasets/signs --stride 6
python train_detector.py datasets/signs/data.yaml --weights models/yolo11n.pt --optimizer AdamW --lr0 0.001 \
  --epochs 60 --batch-size 32 --name synthetic --experiment A
python evaluate_detector.py models/detector/synthetic/weights/best.onnx datasets/signs --split validation --select
```

`--weights`는 해당 가중치에서 **새 학습**을 시작하고, `--resume RUN/weights/last.pt`는 중단된 학습을 이어간다.
`optimizer=auto`는 `--lr0`를 무시하므로 학습률을 비교할 때는 optimizer를 명시한다. 실제 적용값은 `best.json`에 남는다.
신뢰도 임계값은 validation에서 고르고(`--select`가 `best.json`에 저장), test는 `--split test --final`로 마지막에 한 번 본다.

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

2026-10-29 RTX 5070 Ti Laptop(12GB)·Isaac Sim 5.1에서 실제 렌더링·교사 주행·수집·학습·평가를 실행했다.
결과와 남은 항목은 [검증 기록](validation.md)에 있다.
