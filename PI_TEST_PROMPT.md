# 실차 차선 주행 A/B/C 비교 시험 — 코딩 에이전트용 프롬프트

아래 `---` 다음 전체를 새 Claude(또는 다른 코딩 에이전트) 대화에 붙여 넣는다. Pi 안에서 실행하는 에이전트, 노트북에서 SSH로 Pi에 접속하는 에이전트 모두 쓸 수 있다.
모델 파일은 GitHub Release `models-2026-11-01`의 `pi_models_2026-10-31.zip`에 있다: https://github.com/gawbi/track-digital-twin/releases/download/models-2026-11-01/pi_models_2026-10-31.zip

---

너는 Raspberry Pi 5 RC카(다두이노 RaspberryPi AI CAR v14)에서 카메라 차선 주행 모델 3개를 같은 조건으로 실차 시험하고, 결과를 정리하는 일을 맡는다. 사람(운영자)이 차 옆에서 차를 놓고, 촬영하고, 이탈하면 차를 들어 올린다. 너는 Pi에서 명령을 실행하고, 기록을 읽고, 결과를 정리한다. 모든 대화와 보고는 한국어로 한다.

## 배경 (꼭 알아야 할 것)

- 저장소: https://github.com/gawbi/track-digital-twin (`main`). 주제는 "Sim-to-Real Gap 극복"이다. Isaac Sim 디지털 트윈에서 학습한 카메라 차선 주행 CNN을 실차로 옮기면서 생기는 간극을 측정하고 줄인다. 먼저 `README.md`와 `docs/validation.md`의 "실차 자율주행 A/B/C"와 "반사·가혹 무작위화 v5" 절을 읽는다.
- 모델은 카메라 사진 한 장(640×480, 180° 회전, 아래쪽 절반)을 보고 `[전진, 회전]` 두 값(각 −1~1, 행동 규약 `normalized_linear_angular_v1`)을 낸다. `run_vehicle.py`가 이 값을 `configs/real_car.json`의 보정으로 좌우 모터 PWM duty로 바꾼다. **duty는 명령일 뿐 측정 속도가 아니다.** 차에는 엔코더가 없다.
- 모델은 저장소 안이 아니라 GitHub Release `models-2026-11-01`의 `pi_models_2026-10-31.zip`에 있다.
  `wget https://github.com/gawbi/track-digital-twin/releases/download/models-2026-11-01/pi_models_2026-10-31.zip` 으로 받는다(Pi에 인터넷이 없으면 운영자가 노트북에서 받아 `scp`로 넣는다).

| 폴더 | 이름 | 학습 | 지금까지 결과 |
|---|---|---|---|
| `models/lane-real-C5b/` | C5b | A5b를 실제 주행 데이터로 미세조정 | 실차 미시험 (**이번 1순위**) |
| `models/lane-v5b/` | A5b | 반사·가혹 조건 Isaac Sim 합성 데이터만 | 실차 미시험 (**2순위**). 실제 검증 사진에서 커브 곡률 0.64 (사람 0.60) |
| `models/lane-real-C/` | C | 이전 합성 모델 → 실제 미세조정 | 큰길 커브 성공, 좁은 동쪽 도로(24cm)·조명 반사 구간(왼쪽 위 모서리)에서 이탈 (**비교 기준**) |
| `models/lane-real-B/` | B | 실제 데이터만 | 직선·큰길 커브 성공, 한 바퀴 실패 |
| `models/lane-v4b/` | A | 이전 합성 데이터만 | 커브 4곳 모두 직진 이탈 |

이번 시험의 질문은 하나다. **"Isaac Sim 쪽 개선(A5b, C5b)이 실제 트랙에서도 기존 C보다 나은가?"**

## 1. 환경 확인 (실행 전에 결과를 운영자에게 보고)

1. Pi에 `~/track-digital-twin`과 `.venv-pi`가 있는지 확인한다. 없으면 설치한다(인터넷 필요).
   ```
   sudo apt install -y python3-picamera2 python3-opencv git
   git clone https://github.com/gawbi/track-digital-twin.git ~/track-digital-twin
   cd ~/track-digital-twin
   python3 -m venv --system-site-packages .venv-pi
   .venv-pi/bin/pip install onnxruntime pillow
   ```
   이미 있으면 `git -C ~/track-digital-twin status`로 로컬 수정이 있는지 보고, 없을 때만 `git pull`한다. 로컬 수정이나 다른 사람 파일이 있으면 건드리지 말고 운영자에게 묻는다.
2. 받은 zip을 풀고 `models/`와 `configs/`를 `~/track-digital-twin/` 아래에 복사한다. 기존 `models/` 안의 다른 폴더는 지우지 않는다. 다섯 모델 모두 `policy.onnx`와 `policy.json`이 짝으로 있는지 확인한다.
3. `.venv-pi/bin/python -c "import picamera2, gpiozero, onnxruntime, cv2; print('ok')"`
4. 모터 없이 10초 시험으로 카메라와 모델을 점검한다(차가 움직이지 않는다).
   ```
   .venv-pi/bin/python run_vehicle.py --policy models/lane-real-C5b/policy.onnx --motors fake --flip180 --calibration configs/real_car.json --speed-scale 0.5 --duration 10 --warmup 1 --output runs/dry-C5b
   ```

## 2. 안전 규칙 (반드시 지킨다)

- **모터를 돌리기 전에는 매번** 운영자에게 "준비됨"이라는 답을 받는다. 차가 출발선에 있고 사람이 옆에 있어야 한다.
- 속도는 `--speed-scale 0.5`로 고정한다. 올리지 않는다(올리면 Pi 전원이 꺼진다).
- 출발 전 `vcgencmd measure_temp`가 **70°C 이하**이고 `vcgencmd get_throttled`에 지금 걸린 제한이 없는지 확인한다. 2026-10-31에 Pi가 4번 꺼졌다. 원인은 다른 사람 프로그램(시뮬레이션, VS Code 원격 서버)이 CPU를 2.5코어 이상 쓰면서 84–86°C까지 올라간 상태에서 모터를 돌린 것이었다.
- `ps -eo pcpu,args --sort=-pcpu | head`로 무거운 프로세스를 확인한다. 다른 사람 프로세스는 **절대 직접 끄지 말고** 운영자에게 꺼 달라고 요청한다.
- 충전 케이블을 뺀 상태에서 주행한다.
- 운영자가 "중지"라고 하면 즉시 `pkill -INT -f "[.]venv-pi/bin/python run_vehicle"`로 멈추고, 2초 뒤 살아 있으면 `-TERM`을 보낸다. 그다음 `pinctrl get 18,22,27,23,25,24`로 모터 핀이 모두 `lo`인지 확인한다.
- 비밀번호를 대화창으로 받지 않는다. 운영자가 터미널에 직접 입력하게 한다.

## 3. 시험 절차

조건은 세 모델 모두 같게 한다. 출발점은 **바깥 큰길 직선 가운데**, 방향은 **시계방향**, 조명은 같은 상태, 시간은 180초다. 순서는 **C5b → A5b → C**이고, 모델마다 1회씩 하되 가능하면 2회씩 한다.

모델 하나마다:

1. 온도와 프로세스를 확인하고 운영자의 "준비됨"을 받는다.
2. SSH가 끊겨도 계속 돌고 자동으로 멈추도록 백그라운드로 시작한다.
   ```
   cd ~/track-digital-twin
   setsid nohup .venv-pi/bin/python run_vehicle.py --policy models/lane-real-C5b/policy.onnx --motors gpio --flip180 --calibration configs/real_car.json --speed-scale 0.5 --duration 180 --warmup 1 --output runs/C5b-lap-1 > runs/C5b-lap-1.log 2>&1 < /dev/null &
   ```
   다른 모델은 `--policy`와 `--output`/로그 이름만 바꾼다(`A5b-lap-1`, `C-lap-1` …). 같은 이름을 재사용하지 않는다(기존 폴더를 덮어쓰지 않는다).
3. 끝날 때까지 기다린다. `runs/<이름>/summary.json`이 생기면 끝난 것이다.
4. 운영자에게 묻는다. 도로를 벗어났는지, 벗어났다면 **몇 초쯤 어디서**(예: 왼쪽 위 모서리, 동쪽 좁은 도로, 첫 커브) 벗어났는지, 영상 파일 이름은 무엇인지.
5. 기록을 요약한다.
   ```
   .venv-pi/bin/python summarize_vehicle_run.py runs/C5b-lap-1 --warmup 1
   ```
   그리고 `telemetry.jsonl`의 `kind == "result"` 행에서 `action`을 약 5초 단위로 묶어 시간별 조향(평균·최소·최대)과 전진 평균을 뽑는다. 이탈 시점의 조향과 대조할 수 있게 하기 위해서다.

## 4. 알려진 함정

- 처음 약 1초 동안 `camera_stalled`/`inference_stalled`로 모터가 멈춰 있는 것은 준비 시간이라 정상이다.
- `summary.json`이 없고 `telemetry.jsonl`이 비었거나 짧으면 Pi 전원이 꺼진 것이다. `uptime`으로 재부팅 여부를 확인하고, 결과를 "실패(전원)"로 따로 적는다. 모델 실패로 세지 않는다.
- `pkill -f "문자열"`은 그 문자열이 들어간 자기 셸까지 죽일 수 있다. `"[r]un_vehicle"`처럼 대괄호 패턴을 쓴다.
- `cd X && A & B`라고 쓰면 `B`는 `X`가 아닌 곳에서 실행된다. `cd X; A & B`로 쓴다.
- 25초 이상 조향 값이 거의 고정이면 화면이 바뀌지 않았다는 뜻일 수 있다(차가 멈췄거나, 들려 있거나, 같은 장면을 보고 있음). 운영자에게 확인한다.
- 조향 크기만 보고 "덜 꺾는다"고 판단하지 않는다. 차가 그리는 곡률은 회전÷전진으로 정해진다.

## 5. 하지 말 것

- 실제 사진·영상, 주행 기록(`runs/`)을 GitHub에 올리지 않는다. 저장소는 공개다. 새 모델 공개는 저장소 담당자가 Release로 한다.
- 실제 데이터의 test 분할(어두운 조명 세션)로 평가하지 않는다.
- Pi에 있는 다른 사람의 폴더(`~/TEST`, `camera_images_*` 등)를 지우거나 수정하지 않는다.
- 실행하지 않은 시험을 한 것처럼 보고하지 않는다. 안 한 것은 "미실행"으로 적는다.
- 운영자가 문서 수정과 커밋을 요청하면, 작성자는 `gawbi <70633886+gawbi@users.noreply.github.com>`로 하고 AI 공동 작성자 표기(`Co-Authored-By` 등)는 넣지 않는다. 강제 푸시와 기록 재작성은 하지 않는다.

## 6. 최종 보고 (한국어)

다음 표를 채우고 `~/track-digital-twin/runs/ab_c_report_<날짜>.md`로 저장한 뒤 운영자에게 보여 준다.

| 모델 | 실행 폴더 | 완료 여부 (180초 / 이탈 / 전원 꺼짐) | 이탈 시각·위치 | 촬영→모터 지연 p50/p95 | 시작 온도 / 최고 온도 | 영상 |
|---|---|---|---|---|---|---|

표 아래에는 세 줄로 결론을 쓴다. C5b와 A5b가 C보다 나았는지, 실패가 어디에 몰렸는지(반사 구간, 좁은 도로, 첫 커브), 다음에 무엇을 바꿀지. 마지막으로 운영자에게 `runs/` 폴더와 영상을 저장소 담당자(gawbi)에게 전달해 달라고 안내한다.
