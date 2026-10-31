# 실습실 현장 체크리스트 — 실측·실제 데이터·Pi 측정

대상: 2026-10-30(금)·10-31(토) 13:00–15:00 실습실 대여. 2시간 안에 **시뮬레이터 보정값, 독립 촬영 세션,
엔코더 여부, Pi 지연**을 확보하는 순서다. 사람이 나온 영상·사진은 노트북 로컬에만 두고 공개 저장소에 올리지 않는다.

## 0. 출발 전 Pi 준비 (집에서 10분)

Raspberry Pi OS(Bookworm 64-bit) 기준. 시스템의 picamera2/gpiozero를 쓰기 위해 `--system-site-packages`로 만든다.

```bash
sudo apt install -y python3-picamera2 python3-opencv git
git clone --branch feat/isaac-sim-rtx5070 https://github.com/gawbi/track-digital-twin.git
cd track-digital-twin
python3 -m venv --system-site-packages .venv-pi
.venv-pi/bin/pip install onnxruntime pillow
.venv-pi/bin/python -c "import picamera2, gpiozero, onnxruntime, cv2; print('ok')"
```

노트북에서 학습한 `policy.onnx`/`policy.json`(필요하면 검출기 `best.onnx`/`best.json`)을 Pi의 `models/`로 복사한다.

## 1. 측정 (15분) — `configs/rtx.json`의 [추정] 값을 바꿀 근거

| 항목 | 방법 | 현재 시뮬레이터 값 |
|---|---|---|
| 카메라 렌즈 높이 | 매트 표면→렌즈 중심, 주행 때 서보 각도 그대로 | 7cm [사진 추정] |
| 카메라 기울기 | 휴대폰 수평계를 카메라 윗면에. 위/아래 구분 | 8.5° 위 [사진 추정] |
| 렌즈 앞뒤 위치 | 차체 앞끝 기준 | 차 중심에서 13cm 앞 |
| 카메라 모듈 | 모델명·화각(광각/어안) 표기 | 핀홀 등가 88°, k1≈-0.3 |
| 바퀴 지름, 폭 | 줄자 | 6.5cm, 3cm |
| 앞뒤 축간, 좌우 바퀴 중심 간격 | 바퀴 중심끼리 | 14cm [추정], 13cm |
| 무게 | 저울 | 0.9kg [추정] |
| **엔코더 유무** | 모터 뒤 원판·홀센서·추가 선 확인, 사진 | 미확인 — 미션 거리/정지 판정에 필요 |

카메라 서보 각도는 주행·수집·평가 내내 **고정**한다. 각도가 바뀌면 새 보정값과 새 세션으로 기록한다.

## 2. 카메라 화각·왜곡 사진 (10분)

A4에 체커보드(9×6 내부 모서리)를 출력해 칸 한 변을 자로 잰다. 차 카메라로 거리·각도·화면 위치를 바꿔 15–20장 찍는다
(`collect_real.py --driver none --motors fake`로 차를 움직이지 않고 녹화하면 된다). 이후 OpenCV `calibrateCamera`로
`intrinsics`(fx, fy, cx, cy)와 `distortion`을 구해 `--calibration` 카메라 JSON에 넣는다.
추가로 차를 안쪽 횡단보도 정면 20cm·40cm 지점에 놓고 한 장씩 찍어 위치를 적어 두면 높이·기울기 검증에 쓴다.

## 3. 운동 보정 영상 (15분) — 속도·스키드 회전 효율

고정된 상단 카메라는 필요 없다. 서서 휴대폰으로 내려다보며 찍거나, 영상 없이 줄자로 잰다.
직진은 출발선 테이프 → 멈춘 뒤 앞 범퍼까지 거리, 제자리 회전은 `--spin-seconds 6`으로 길게 돌려 몇 바퀴인지 센다
(예: 1¾바퀴 = 630°). 주행·인식은 차량 카메라 하나만 쓴다(시뮬레이터 영상의 상공 화면은 표시용).

```bash
.venv-pi/bin/python tools/motion_calibration.py --motors fake --output motion-dry.json   # 먼저 모터 없이
.venv-pi/bin/python tools/motion_calibration.py --motors gpio --spin-seconds 6 --output motion-1.json  # 바닥에서
```

직진 0.3/0.5 duty, 좌·우 제자리 회전, 좌·우 원호를 단계별로 실행하고 매 단계 정지한다. 영상에서 이동 거리(m)와
회전 각도(°)를 재어 `measured.json`에 적고 `--measured measured.json`으로 **duty당 속도와 회전 효율 η**를 계산한다.
같은 단계를 3번 반복하면 편차를 알 수 있다. 결과는 `configs/rtx.json`의 `skid_yaw_efficiency`와 실차 보정 JSON
(`DutyMixer`: `max_duty`, `turn_mix`, 좌우 trim)에 반영한다. PWM 값을 측정 속도로 기록하지 않는다.

## 4. 실제 주행 세션 녹화 (40분) — 세션별 고정 분할의 재료

세션마다 **조건을 바꾼다**. 같은 영상을 잘라 세션 ID만 바꾸지 않는다.

| 세션 예 | 내용 |
|---|---|
| `2026-10-30-310-outer-cw` | 바깥 순환 시계방향 3–5분 |
| `2026-10-30-310-outer-ccw` | 반시계방향 |
| `2026-10-30-310-inner` | 가운데 교차로·횡단보도·신호등 포함 경로 |
| `2026-10-30-310-dim` | 조명 일부 끄거나 블라인드, 반사 변화 |
| `2026-10-30-310-props2` | 표지판·신호 위치를 바꾼 배치 |

```bash
.venv-pi/bin/python collect_real.py --output real/sessions --session-id 2026-10-30-310-outer-cw \
  --location 310호 --lighting "형광등 전부" --motors gpio --teleop-linear 0.5 --teleop-angular 0.8
```

SSH 터미널에서 W/S/A/D로 운전하고 X 정지, Q 종료. 키를 0.3초 이상 안 누르면 멈춘다.
첫 프레임이 뒤집혀 보이면 `--flip180`을 붙여 새 세션으로 다시 찍는다(교재 8_4.py는 영상을 180° 뒤집는다).
표지판이 없는 장면, 반사가 심한 장면, 먼 표지판도 포함한다. 세션 폴더는 USB/scp로 노트북 `outputs/real_sessions/`에 옮긴다.

## 5. 표지판 배치·규칙 확인 (10분)

- 트랙 전체를 위에서 찍고, 각 표지판·신호등이 어느 파란 원(`track_layout.json`의 `prop_spots` ID)에 어떤 방향으로
  놓였는지 적는다. 각 표지판이 대응하는 분홍 점선도 표시한다.
- 조교/담당자에게 확인: STOP 정지 시간, 감속20의 감속 정도와 해제 지점, 경적 길이, 노란 신호 처리,
  빨간 신호 정지 위치(횡단보도 전/신호등 전), 빨간 실선의 의미. 답을 `docs/mission_rules.md`에 출처와 함께 추가한다.
- 팀 딥러닝 쪽과 같은 클래스 정의(표지판 6종 + 신호 2종 vs 노랑 포함 9종)를 맞춘다.

## 6. Pi 지연·주행 측정 (20분)

바퀴를 띄운 상태에서 가짜 모터 → 실제 모터 순서로 한다.

```bash
.venv-pi/bin/python run_vehicle.py --policy models/lane/policy.onnx --motors fake --duration 60 --output runs/pi-fake
.venv-pi/bin/python run_vehicle.py --policy models/lane/policy.onnx --detector models/detector/best.onnx \
  --motors fake --duration 60 --output runs/pi-fake-det
.venv-pi/bin/python run_vehicle.py --policy models/lane/policy.onnx --motors gpio --speed-scale 0.4 \
  --duration 60 --output runs/pi-drive-1
.venv-pi/bin/python summarize_vehicle_run.py runs/pi-fake runs/pi-fake-det runs/pi-drive-1
```

`summary.json`에 단계별(전처리·차선·검출) 시간, 촬영→모터 명령 p50/p95/최대, 제어 주기 지연, 만료·정지 사유,
CPU 온도·스로틀링이 남는다. 바닥 주행은 휴대폰으로 촬영하고 시도마다 `trials.jsonl`에 결과·영상 경로를 적는다
(형식은 `summarize_vehicle_run.py --help`). 엔코더가 없으면 검출기는 기록만 하며 정지·감속을 제어하지 않는다.

## 7. 마무리

- 세션·측정 사진·영상·`motion-*.json`·`runs/`를 노트북으로 복사.
- 노트북에서 `split_real.py`로 세션 분할을 **한 번** 고정하고, 이후 세션이 늘면 새 manifest로 만든다.
