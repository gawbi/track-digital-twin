# Git 클론부터 시작하는 RTX 노트북용 프롬프트

2026-10-29 업데이트. **저장소가 아직 없는 Ubuntu 노트북에서도** 아래 프롬프트 전체를 코딩 에이전트에 붙여 넣으면 된다.
사용자가 먼저 Git clone, 폴더 이동이나 문서 검색을 할 필요 없이, 에이전트가 그 단계부터 진행하도록 요청한다.

이 프롬프트는 코딩 에이전트의 대화 입력창에 붙여 넣는다. 터미널에 전체 내용을 붙여 넣는 명령 파일이 아니다.

```text
Ubuntu RTX 노트북에서 아래 프로젝트를 처음부터 준비하고 구현·실행·검증까지 이어서 진행해 주세요.
한국어로 설명하고, 계획이나 설치 명령을 보여주는 데서 멈추지 말고 가능한 명령은 직접 실행해 주세요.

[장비와 저장소]
- Ubuntu / RTX 5070 노트북 / Isaac Sim 5.1, 별도 실차는 Raspberry Pi 5와 보유 카메라·트랙입니다.
- 저장소: https://github.com/gawbi/track-digital-twin.git
- 작업 브랜치: feat/isaac-sim-rtx5070
- 구현 요청: https://github.com/gawbi/track-digital-twin/issues/2
- 기존 PR: https://github.com/gawbi/track-digital-twin/pull/1

[1. Git 설치 확인과 클론부터 직접 시작]
현재 위치·Ubuntu 버전·GPU/VRAM·드라이버·디스크 여유와 Git 설치 여부를 확인해 주세요.
Git이 없으면 Ubuntu에 맞게 설치를 진행해 주세요.
작업 폴더가 없으면 다음 명령으로 저장소를 받아 주세요.
mkdir -p "$HOME/projects"
cd "$HOME/projects"
git clone --branch feat/isaac-sim-rtx5070 --single-branch https://github.com/gawbi/track-digital-twin.git
cd track-digital-twin

같은 이름의 폴더가 있으면 원격 주소·브랜치·git status부터 확인해 주세요.
이 저장소의 깨끗한 작업 폴더면 fetch 후 대상 브랜치를 fast-forward로 갱신하고,
다른 프로젝트이거나 수정 중이면 내용을 보존한 채 별도 폴더에 clone해 주세요.
기존 파일 삭제·hard reset·강제 푸시는 하지 마세요.

[2. 저장소를 읽고 설치부터 해결]
RTX_HANDOFF_PROMPT.md, docs/rtx_ubuntu.md, docs/real_vehicle_plan.md,
docs/autonomy.md, docs/mission_rules.md, docs/validation.md와 이슈 #2를 읽어 주세요.
GitHub 이슈를 읽을 수 없어도 같은 요구사항이 로컬 docs/real_vehicle_plan.md에 있으니 계속 진행해 주세요.
Isaac Sim 5.1이 설치되어 있으면 실제 설치 경로와 버전을 찾아 사용해 주세요.
없으면 공식 5.1 문서에 따라 설치를 진행해 주세요. Python 3.11과 전용 실행 환경을 준비하고,
Isaac 번들 torch/numpy 및 시스템 기본 Python은 임의로 교체하지 마세요.
시뮬레이션과 CUDA 학습 환경을 분리하고 필요한 패키지를 설치해 주세요.

[3. 실제 첫 실행부터 학습·평가까지]
run_sim.py --check → 짧은 headless 교사 주행과 저장 영상 확인 → GUI 수동 확인 가능 여부 점검
→ outer/inner 교사 주행 → 소량 합성 수집과 라벨 확인 → 별도 CUDA 환경 학습
→ 학습에 쓰지 않은 조건에서 차선·미션 평가 순으로 진행해 주세요.
드라이버·API·메모리·카메라·바퀴 접촉 문제는 로그를 보고 수정하고 다시 검증해 주세요.
가중치가 없으면 먼저 데이터를 준비하고 학습하세요. 실패한 수집으로 장시간 학습을 시작하지 마세요.

[4. 후속 네 가지도 직접 구현]
실제 영상·조작값 수집 및 촬영 세션별 고정 분리, 합성 가중치의 실제 데이터 미세조정,
최신 프레임 추론과 별도 모터 제어 및 만료 정지, Pi 전체 지연·주행/미션 성능 측정을
docs/real_vehicle_plan.md의 완료 기준에 따라 구현해 주세요.
실험은 합성만 / 실제만 / 합성→실제 미세조정부터 비교하고,
모델·임계값은 validation으로 선택하며 test는 최종 평가에 남겨 주세요.
각도·속도·PWM의 차이와 엔코더 유무를 확인하고, PWM을 측정 이동량으로 취급하지 마세요.

[진행과 완료 보고]
이미 지정된 작업의 다음 단계마다 다시 시작 허락을 묻지 말고 이어서 진행해 주세요.
로그인·비밀번호·필수 동의·권한 승인이 실제로 필요하면 정확한 화면/명령과 필요한 조치만 알려 주세요.
Pi나 실측 데이터가 아직 없어도 독립적으로 가능한 코드·가짜 모터 검사·RTX 작업은 계속해 주세요.
새 하드웨어 구매·유료 클라우드는 기본 전제로 삼지 마세요.
실행한 명령·기기 환경·변경 파일·검증 결과·남은 항목을 기록하고, 미실행 검증은 구분해 주세요.
원본 영상·개인정보·교재 코드·가중치는 공개하지 말고 기존 브랜치/PR에 코드·문서·결과 요약을 반영해 주세요.
실제 RTX 실행·학습·실차 검증을 하지 않고 전체 완료라고 보고하지 마세요.
```

직접 터미널에서 시작하려면 [클론·설치·첫 실행 안내](docs/rtx_ubuntu.md)를 따른다.
클론 이후의 상세 인계는 [RTX_HANDOFF_PROMPT.md](RTX_HANDOFF_PROMPT.md), 네 가지 후속 구현의 완료 기준은
[실차 구현 요구사항](docs/real_vehicle_plan.md)에 있다.
