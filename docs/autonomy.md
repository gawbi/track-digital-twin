# 표지판 추적·거리 추정·미션 주행 통합

실행 대상: **Ubuntu / RTX 5070 / Isaac Sim 5.1**.
추가 구현 범위는 코드와 CPU에서 가능한 검증까지다. RTX에서의 실주행·가중치 학습·성능 검증은 별도로 진행한다.

## 연결된 처리 흐름

`RGB + 촬영 시각 + 바퀴 각속도 → ONNX 검출 → 추적/중복 차단 → 거리/교차로 판단 → 미션 상태기계 → 선속도·각속도·경적 이벤트`

- `dtwin/perception.py`: YOLO11 ONNX의 letterbox 역변환, 신뢰도 필터, NMS, RGB 입력 검사.
- `dtwin/tracking.py`: 신호 색 변화 시 같은 ID 유지, 짧은 가림 후 재연결, 반복 프레임 확인 횟수 제외.
  장기 가림으로 ID가 바뀌어도 같은 위치의 encounter로 묶는다. 대상을 지나친 후 충분히 이동해야 다음 접근으로 재인식한다.
- `dtwin/geometry.py`: 카메라 보정값으로 바닥/표지판 높이 평면과 광선을 교차시켜 m 단위 거리 계산.
  선택적으로 실제 `fx/fy/cx/cy`와 OpenCV 왜곡 계수를 받는다. 바퀴 각속도로 위치와 회전을 적분하고 과거 카메라 시각에 맞춰 보간한다.
- `dtwin/road_vision.py`: 분홍 점선과 여러 흰 줄의 횡단보도를 RGB에서 찾고, 표지판 bbox는 노면 후보에서 제외한다.
- `dtwin/autonomy.py`: 표지판 encounter에 대응 선을 고정해, 선을 지난 뒤 다음 선으로 바꿔 인정하는 오류를 막는다.
  미션 제어와 주행 명령을 결합하고 실제 실행·완료·오류를 기록한다.
- `dtwin/navigation.py`: 교차로 접근 → 정지 → 회전 → 진출. 방향 지시가 없으면 진입 전 기다리고,
  서로 충돌하는 지시나 도로가 없는 방향은 주행하지 않는다.

## 입력 근거와 거리 기준

미션 제어는 시뮬레이터의 `pose()`나 렌더러 정답 bbox를 받지 않는다.
`SingleArticulation.get_joint_velocities()`에서 읽은 **측정 바퀴 속도(rad/s)**를 사용하며,
모터의 목표 속도를 측정값 대신 넣지 않는다. 위치는 알려진 시작 위치와 바퀴 적분으로 추정한다.

차선 모델 단독 모드는 카메라 정책이다. `--missions`는 **시각 관측 + 바퀴 회전량 + 사전에 알려진 트랙 지도**를 사용하는 제어다.
지도와 공칭 시작 위치는 사전 정보이며, 무작위화한 실제 위치나 카메라 각도를 정답으로 전달하지 않는다.
시뮬레이터 위치·bbox 정답은 기존 평가/데이터 라벨 필드에만 별도로 저장한다.

거리는 차체 중심이 아닌 **앞 범퍼부터 선의 앞 가장자리**까지다.
횡단보도는 `center-N/S/E/W` 네 곳만 신호 규칙에 연결한다.
노면 표시를 영상에서 확인하면 해당 관측을 사용하고, 아직 보이지 않으면 지도/바퀴 적분으로 예측한다.
`range_source`에 두 경우를 구분해 기록한다. 미터 단위 출력이 실제 측정 정확도를 보장하는 것은 아니다.
미끄러짐, 카메라 보정 오차와 표지판 높이 오차는 RTX·실차 데이터로 확인해야 한다.

## 적용 동작

| 관측 | 주행 동작 / 완료 판정 |
|---|---|
| 경적 | 분홍선 여유가 있을 때 한 번 출력. 터미널 벨과 경적 이벤트 기록 |
| STOP | 분홍선 전에 정지 명령. 바퀴 추정 선속도와 회전 속도가 정지 범위에 들어간 뒤 시간을 재고 재출발 |
| 감속20 | 설정된 배율로 속도·각속도를 함께 줄임. 측정 속도가 내려간 뒤 완료 기록 |
| 빨간 신호 | 안쪽 횡단보도 이전 제동 거리로 감속, 정지 뒤 약 1초 대기, 해당 교차로 우회전 |
| 초록 신호 | 해당 교차로에 우회전 요청을 저장하고, 교차로 중앙에 접근한 뒤 회전 |
| 방향 표지판 | 좌/우/직진 요청을 해당 교차로에 연결. 미션 완료 후 다른 교차로에 재사용하지 않음 |
| 오래된 영상/거리 미확인 | 정지하고 진단 기록. 유효한 관측이 회복되면 다시 판단 |
| 내 경로 앞 분홍선과 연결되지 않은 미션 표지판 | 무시하고 `sign_not_on_route_ignored` 기록(2026-10-30, 이전에는 정지). 가까워져 연결되면 수행 |
| 선을 지난 뒤 동작, 방향 충돌, 진출 실패 | 오류를 유지하고 정지. 성공으로 집계하지 않음 |

원래 규칙이 미확정인 STOP 시간·감속률·감속 해제·경적 시간은 `configs/rtx.json`의 추정 설정이다.
감속은 후속 해제 규칙이 없으므로 계속 유지한다. 노란 신호의 기본 `yellow_behavior=stop`은 대회 규칙 확정이 아닌
운용 기본값이다. 규칙 확인 후 설정한다. `follow_lane`을 명시적으로 선택할 수도 있다.
경적은 가상 이벤트/터미널 벨이며 실제 차량 부저나 일정 길이의 음원을 재생하는 기능은 아니다.

## RTX 노트북에서 실행하는 명령

차선 정책과 새 검출기 가중치를 먼저 준비한다. 기존 6종 검출 클래스 뒤에 좌/우/직진을 추가한 **9종**이다.
`train_detector.py`가 내보낸 `.onnx` 옆의 `.json`에는 클래스 순서와 출력 형식을 기록한다.
미션 모드는 가중치/메타데이터가 없으면 시작하지 않으며, 위치 정답 교사 모드와 함께 사용할 수 없다.

```bash
export ISAAC_PY="$HOME/isaacsim/python.sh"
"$ISAAC_PY" run_sim.py --mode onnx --route inner \
  --policy models/lane/policy.onnx \
  --missions --detector models/detector/train/weights/best.onnx \
  --record --seed 9001 --duration 120 --output outputs/missions
```

기본 `props`는 세 종류 방향 표지판을 포함한 **예제 배치**다. 음성의 “여기”를 복원한 실제 평가 배치가 아니다.
소품 위치/방향, 카메라 설정, 실제 미션 동작값을 평가 환경에 맞게 보정한다.
`--randomize`에서는 다른 소품 위치와 카메라/마찰 조건도 시험한다.

```bash
python3 run_batch.py --python "$ISAAC_PY" --mode evaluate \
  --policy models/lane/policy.onnx --missions \
  --detector models/detector/train/weights/best.onnx \
  --route inner --duration 120 --count 10 --seed-start 12000 --output outputs/mission-eval
python3 evaluate_runs.py outputs/mission-eval
```

정책과 검출기 학습 시드를 모두 확인해 평가 시드 중복을 거부한다.
차선 단독과 미션 제어의 결과를 같은 폴더에서 섞어 집계하지 않는다.

## 보정 파일

`--calibration camera.json`에 카메라 설정 객체를 전달할 수 있다.
기본 필드는 `width`, `height`, `x_m`, `z_m`, `pitch_down_deg`, `yaw_deg`, `hfov_deg`이며,
`configs/rtx.json`의 `camera` 객체와 같은 형식이다. 추가로 다음 값을 넣을 수 있다.

| 필드 | 값 |
|---|---|
| `intrinsics` | 측정된 `fx`, `fy`, `cx`, `cy` 객체. 단위 px |
| `distortion` | OpenCV pinhole 왜곡 계수 4/5/8개 |

영상 크기는 보정값과 같아야 한다. 표지판 아래쪽 높이는 `missions.sign_bottom_m`,
신호등 판 아래쪽 높이는 `missions.signal_bottom_m`로 조절한다.
바퀴 지름·좌우 간격과 마찰을 보정하지 않으면 스키드 스티어의 회전 오차가 적분될 수 있다.

## 기록과 성공 판정

`frames.jsonl`의 `autonomy`에는 검출·관측, 측정 바퀴 속도, 추정 위치, 거리 출처,
촬영/제어 시각, 상태기계, 교차로 단계와 정지 원인을 저장한다. `mission_events.json`은 시간순 이벤트다.
완료 이벤트는 정지/감속 확인 또는 신호 교차로 진출 후에 남긴다.

`summary.json`은 차선 성공과 미션별 완료 횟수·오류를 구분한다.
전체 성공은 설정의 `evaluation_gates`를 순서대로 통과해 한 바퀴를 이탈 없이 돌고, `required_missions`를 모두 수행하고,
교차로 동작이 끝났으며 정지 오류가 없는 경우다. 기본 필수 목록은 경적·STOP·감속20·신호다.
실제 시나리오가 달라지면 이 목록과 게이트 순서도 맞춰야 한다. 기본 게이트도 예제 경로이며 공식 대회 경로가 아니다.
교차로에서 경로가 바뀌어도 기존 고정 경로에 투영한 누적 거리로 완주를 잘못 세지 않도록,
위치 정답은 평가 전용의 순서 있는 게이트 통과 판정에만 사용한다. 클래스가 검출되지 않았는데 전체 성공으로 인정하지 않는다.

## RTX 없이 확인할 수 있는 범위

```bash
python -m unittest discover -s tests -v
```

검출 ONNX 입출력, 추적/가림/색 변화, 투영/왜곡 보정, 바퀴 적분,
경적·STOP·감속·신호와 교차로 제어를 CPU에서 검사한다.
빨간 신호 감지부터 실제 정지 대기·우회전·진출까지는 이상적인 바퀴 운동과 투영 bbox를 사용한 소프트웨어 재생 테스트다.
학습된 검출기의 인식률이나 PhysX 주행을 검증한 것으로 해석하지 않는다.

추후 저장한 미션 주행은 RTX 없이 다시 처리할 수도 있다.

```bash
python replay_missions.py outputs/missions/onnx-inner-seed9001-missions \
  --initial-pose 0 0.8725 -1.57079632679 \
  --policy models/lane/policy.onnx --detector models/detector/train/weights/best.onnx \
  --output outputs/replay.jsonl
```

재생은 기록된 영상과 바퀴 측정값만 사용하며 위치/bbox 정답 필드를 제어에 넣지 않는다.
새 명령이 기록된 궤적을 바꾸지는 않으므로, 이 결과 역시 폐루프 완주율 평가가 아니다.
