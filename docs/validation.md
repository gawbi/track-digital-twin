# 검증 기록 — 2026-10-22

대상 실행 환경은 사용자 지정 Ubuntu / RTX 5070 / Isaac Sim 5.1이다.
아래 검사는 개발 기기에서 가능한 소프트웨어·USD 구조 검사다.

## 통과한 검사

- 단위 검사 16개: 트랙 이미지/JSON 치수 일치, 도로 폭 여유, 좌우 바퀴 명령 부호·단위·포화,
  경로 복구와 완주 거리, 난수 재현성, GPIO 핀 연결, 미션 중복 방지,
  실제 정지 후 1초 계산, 신호/신뢰도 처리, 정답 관측 혼입 방지,
  RGB/YUV 입력과 CNN 출력, 시드별 데이터 분리, 객체 라벨 정규화,
  지연 영상과 라벨의 동기화, 물리/렌더링 호출 비율, USD의 4개 바퀴 조인트·카메라 방향·UV 좌표.
- 작은 생성 데이터 5개 장면/15프레임으로 1 epoch 학습, ONNX 내보내기와 모델 검사,
  ONNX Runtime 추론, YOLO 라벨 데이터셋 변환까지 통과.
- Python 정적 검사와 `git diff --check` 통과.
- `run_sim.py --check`: 트랙 2355×2400px ↔ 2.355×2.4m 일치,
  차 폭 0.16m 대비 가운데 동서 도로의 한쪽 여유 0.0125m.

소규모 생성 데이터는 실행 경로 검사용이다. 이 테스트의 가중치·오차를 주행 성능으로 보고하지 않으며,
가중치와 테스트 이미지는 임시 폴더에서 제거한다.

```bash
python -m unittest discover -s tests -v
python tools/smoke_learning.py
ruff check dtwin tests tools run_sim.py run_batch.py train_policy.py evaluate_runs.py make_detector_dataset.py train_detector.py
```

`test_scene.py`는 `usd-core` 또는 Isaac Sim의 USD 모듈이 필요하다. 외부 `usd-core`에는 PhysX 스키마가 없어,
이 구조 검사는 실제 PhysX 접촉·마찰·구동을 검증하지 않는다. 학습 검사에는 별도 학습 환경의 torch/ONNX가 필요하다.

## RTX에서 남은 확인

1. `--check`로 Ubuntu·드라이버·VRAM·설치 버전 기록.
2. 수동 주행으로 실제 렌더 프레임, 바퀴 접촉과 회전 방향, 카메라 시야 확인.
3. 교사 주행으로 바깥 순환로와 가운데 교차로 회전의 이탈·도로 여유 확인.
4. 카메라 높이/각도와 차량 치수·질량·마찰·PWM 속도를 실차와 보정.
5. 무작위화 데이터를 충분히 수집한 후 별도 CUDA 환경에서 학습.
6. 학습에 사용하지 않은 시드·스트레스 조건·실차 영상으로 성능 측정.
7. 검출 추적·분홍선/횡단보도 거리·방향 계획을 미션 상태기계와 통합해 미션 전체를 평가.

현재는 실제 RTX 시뮬레이션 완주나 강건한 모델의 완성을 확인한 상태가 아니다.
