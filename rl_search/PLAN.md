---
name: RL vs lawnmower search
overview: rl_search/에서 60 km 고정 지형 위의 무작위 시나리오 생성기와 평가 세트(0~1단계), 임무/기체를 나눈 reset/step 환경(2단계), PX4 Gazebo rc_cessna에 맞춘 고정익 모델과 선회 항력 반영 에너지 예산(4단계), 이동 중 계획 완성 후 경로점 인계와 QGC .plan 내보내기(5단계), lawnmower 비교(3단계), 바람 반영(6단계)을 모두 만들고, 내보낸 경로를 PX4 SITL에서 이륙부터 착륙까지 비행해 확인했다. 설정은 프리셋(config.py)과 --set으로 바꾸고, 기체/플래너/방법/바람 방식은 등록표로 갈아 끼운다. RL 학습은 범위 밖이다.
todos:
  - id: step0-folder
    content: "0단계: rl_search/ 폴더 구조, 의존성, 절차적 60 km 지형 1회 생성"
    status: completed
  - id: step1-sampler
    content: "1단계: 시나리오 샘플러 (LKP 먼저, 그다음 LKP에서 일정 거리인 맵 가장자리 출발점, 타겟 heading과 속도 무작위)"
    status: completed
  - id: step1-exit
    content: "1단계: 타겟 시뮬레이션에서 맵 밖으로 나간 입자의 이탈 시각(exit_step) 기록"
    status: completed
  - id: step1-generator
    content: "1단계: 매 에피소드 시나리오 생성기 (reset마다 타겟 시뮬레이션, reuse 설정값 기본 1) + 고정 seed 평가 세트 저장"
    status: completed
  - id: step1-check
    content: "1단계: 시나리오 하나를 기존 uav_search greedy로 돌려 POS 곡선, 실행 시간, 입자 밀도 확인"
    status: completed
  - id: step2-env
    content: "2단계: mission.py(이탈 입자 처리 포함)와 KinematicPlant 분리, SearchEnv reset/step, 기존 run_mission과 결과 일치 확인"
    status: completed
  - id: step4-config
    content: "4단계: PX4 rc_cessna 기준 설정으로 변경 (15 m/s, dt 4 s, 비행 60분, POD 0.3, LKP 5~15 km, 표적 15~30 km/h), 평가 세트 재생성과 분포 확인"
    status: completed
  - id: step4-plant
    content: "4단계: FixedWingPlant (NPFG 유도 근사 + 롤 1차 지연/속도/각도 한도 + 균형 선회, PX4 파라미터 이름), check_plant.py 검증 그림, max_turn 결정"
    status: completed
  - id: step4-energy
    content: "4단계: 롤과 속도에 따른 동력 모델로 에너지 예산 (선회 시 소비 증가), env 종료 조건을 에너지 기준으로"
    status: completed
  - id: step5-handover
    content: "5단계: 이동 단계(LKP로 자동 직진) + 계획 완성 시각(기본 120 s)에 인계, 계산 시간 측정"
    status: completed
  - id: step5-export
    content: "5단계: 계획 경로를 도착 시각 포함 경로점 목록으로 솎아 내보내기 (CSV + QGC .plan), 경로 추종 실행 결과와 계획 비교"
    status: completed
  - id: step3-lawnmower
    content: "3단계: 인계 시점부터 시작하는 lawnmower(경로점 목록)와 compare.py, 같은 에너지 예산과 평가 세트 평균/분산 비교"
    status: completed
  - id: step6-wind
    content: "6단계(마지막): 바람 모델, 바람 반영 계획, 대기 속도로 일정 맞추기, 일정 오차 기준 재계획, 세 방식 비교"
    status: completed
  - id: final-sitl
    content: "최종 검증: 내보낸 .plan을 PX4 SITL gz_rc_cessna에 올려 이륙, 경로점 비행, 착륙까지 자동 확인 (sitl_fly_plan.py)"
    status: completed
isProject: false
---

# RL vs Lawnmower: 단계별 계획

## 목표

같은 배터리(에너지) 조건에서 RL이 고른 heading으로 비행하면 lawnmower 같은 고정 패턴보다 누적 POS가 높아질 수 있음을 보인다. 검증은 PX4 + Gazebo와 실기체에서 하므로, Python 모델은 PX4 고정익 제어 구조와 Gazebo 기체에 최대한 맞춘다. 실제 운용에서는 UAV가 LKP로 이동하는 동안 계획을 완성하고, 완성된 계획(경로점 목록)을 PX4에 넘겨 그대로 비행한다.

## 진행 순서

```mermaid
flowchart LR
  S0["0-2: terrain, scenarios, env (done)"] --> S4["4: PX4-matched plant and energy (done)"]
  S4 --> S5["5: transit, handover, waypoint export (done)"]
  S5 --> S3["3: lawnmower and compare (done)"]
  S3 --> S6["6 (last): wind-aware planning (done)"]
  S6 --> V["PX4 SITL check (done)"]
  V --> N["next: ulog fit, RL"]
```

전체 흐름 그림은 `docs/diagrams/research_flow.png`, 모듈 구조는 `docs/diagrams/modules.png`에 있다.

4단계를 3단계보다 먼저 한다. lawnmower는 기체의 선회 한계와 인계 방식 위에서 설계해야 하므로, 고정익 모델과 인계 구조가 먼저 있어야 한다.

## 확정한 결정

> **RL 단계에서 바뀐 결정 (2026-10-30):** 아래는 0~6단계(플래너·베이스라인) 기준이다. RL 학습 설계에서는 다음이 다르며,
> 근거와 이력은 [todo_response.md](../todo_response.md)와 [단일 UAV RL TODO](../docs/single_uav_rl_todo.md)에 있다.
>
> - **결정 간격:** 4 s → 5 s(4.4). 행동은 ±30° 또는 ±45°로 제한하는 연속 방향 변화량이다. 아래 `max_turn 133°/4 s`는 기체 능력이며 RL 행동 범위가 아니다.
> - **센서:** 5 s 구간의 관측 셀 합집합으로 커널을 적분하고 POD를 구간당 한 번 적용한다(5.4 v3). 그래서 아래 표의 POS 수치는 RL 결과와 직접 비교하지 않고, 베이스라인을 같은 센서 모델로 다시 돌린다(12.4).
> - **출발점과 커리큘럼:** "커리큘럼은 쓰지 않는다"는 플래너 평가용이다. RL 학습은 근거리 시나리오부터 시작하는 커리큘럼을 쓴다(7.5).
> - **종료:** 에너지 소진은 `truncated`가 아니라 `terminated`로 처리한다(6.1, 6.3). 지도 이탈도 `terminated`다.
> - **탐색 보너스:** 아래 `explore_density` 결정은 플래너용이다. RL은 확장 HDR 안의 신규 관측 면적을 별도 보상 항으로 쓴다(1.1, 1.5).

- **기존 파일:** 루트의 기존 파일은 참고용으로 그대로 둔다. 바꿀 부분은 `rl_search/`로 fork해서 수정하고, 수정 후 기존 플래너 결과와 비교한다.
- **맵과 시간:** seed 7로 만든 60 km × 60 km 절차적 지형 하나를 고정한다. t = 0은 LKP 관측 시각이면서 UAV 출발 시각이다.
- **검증 기체:** PX4 Gazebo `gz_rc_cessna`를 기준으로 한다. 공력까지 맞출 필요가 생기면 `gz_advanced_plane`(PX4 v1.15+)으로 바꾼다.
- **UAV (4단계부터):** 순항 15 m/s(`FW_AIRSPD_TRIM` 기본값). 속도는 설정값이라 33.3 m/s(120 km/h)도 그대로 쓸 수 있다. 결정 간격 4 s(한 스텝 60 m, footprint 길이 83 m보다 짧음). 비행 예산은 에너지로 정의하고 기준은 순항 60분이다.
- **행동:** 결정 간격당 heading 변화량(turn)이다. 롤은 기체 모델 내부 상태이고, 4단계부터 관측에 넣는다.
- **PX4 명령 방식:** NPFG 경로 추종에 맡긴다. 고도와 속도는 PX4(TECS)가 유지한다. 운용에서는 계획 전체를 도착 시각이 붙은 경로점 목록으로 넘긴다.
- **출발점:** 언제나 맵 가장자리에서 출발한다. 커리큘럼은 쓰지 않는다.
- **시나리오 공급:** 학습용은 reset마다 새로 만든다(`reuse` 기본 1). 너무 느리면 `reuse`를 올리거나 시나리오 뱅크를 쓴다. 평가용만 고정 seed로 저장한다.
- **타겟 (4단계부터):** LKP는 출발점에서 5~15 km, 속도는 시나리오마다 15~30 km/h, heading은 균등, 불확실성 1σ = 25°다. 이동 모델은 침투자 모델이고 차량 도로 모델 자리를 남겨 둔다.
- **입자:** 5000개, 시뮬레이션 5 s, 저장 10 s, 길이 70분(비행 60분 + 여유 10분)이다.
- **맵 밖으로 나간 입자:** 이탈 순간부터 고려하지 않는다. 그 질량은 escaped로 집계하고, 최대 POS는 1 - escaped다.
- **탐지:** POD 기본 0.3이다. 비교 실험은 POD 1과 0.3 두 가지로 한다.
- **탐색 보너스:** `explore_density` 기본 0이다. 평가 POS가 플래너와 같은 입자로 계산되므로, 보너스는 목적함수를 왜곡할 뿐이다(아래 기록 참고). 쓸 경우 0.01, 사전확률 영역만 쓴다.
- **카메라:** 짐벌이 롤과 피치를 상쇄해서 항상 수직 아래를 본다(짐벌 롤 ±45°). footprint는 heading 방향 83 × 115 m 직사각형이다. `NadirSensor`로 분리되어 있다. 롤 한도 50°가 짐벌 범위를 넘으므로, 4단계에서 롤이 45°를 넘으면 경고가 나오는지 확인하고 필요하면 계획용 롤 한도를 45°로 둔다.
- **배터리:** 선회 시 항력 증가로 소비가 빨라진다(4단계 에너지 모델).
- **운용 흐름:** 출발 후 LKP로 직진하는 동안 계획을 계산하고, 완성 즉시(기본 120 s) 계획으로 인계한다. 남은 이동 구간도 계획에 포함된다.
- **바람:** 마지막 단계(6단계)에서 다룬다.

## 0~2단계 (완료)

```
rl_search/
  README.md, PLAN.md, requirements.txt
  terrain.py      절차적 지형: TerrainConfig, Terrain, make_terrain, save/load
  plotting.py     음영 기복 지형 그림
  target.py       fork: create_guidelines, simulate (LKP/heading/속도 인자, 속도 상한, 맵 밖 유도선, exit_step, move_model)
  scenario.py     ScenarioConfig, ScenarioSampler, make_scenario, ScenarioGenerator(reuse), save/load
  mission.py      fork: SearchConfig, ParticleBelief(이탈, escaped), NadirSensor, footprint/POA, detect, Coverage(reset)
  plants.py       KinematicPlant (4단계에서 FixedWingPlant 추가)
  env.py          SearchEnv: reset(seed, options) / step(action), record 로그
  planners.py     fork: straight/greedy/tree가 turn을 돌려줌, run_episode, save_mission
  scripts/
    make_terrain.py      지형 생성
    make_scenarios.py    평가 세트, 요약 그림, 분포 확인 (병렬)
    export_scenario.py   simulation_data.npz 내보내기, 입자 밀도 확인
    check_env.py         기존 uav_search와 회귀 비교, 질량 보존, 환경 속도
  data/                  생성물 (git 제외)
```

- **지형:** 산맥, 구릉, 배수망을 따라 생긴 강과 개울, 호수와 평야, 여울, 숲이 있다. 이동 불가 구역(산 상위 8%, 강, 호수, 32° 이상 경사)은 맵의 약 14%다.
- **1단계 결과 (33.3 m/s 설정 기준):** 시나리오 하나 생성은 약 9 s(코어 하나)이고, 같은 seed면 같은 시나리오가 나온다. 따라잡는 시간 중앙값은 11분이다. 맵 밖 이탈 질량은 중앙값 0.9%, 최대 96%다. 따라잡은 시각의 최고 footprint POA는 약 4e-3이라 KDE 대역폭은 150 m를 유지한다.
- **2단계 결과:** greedy와 tree 모두 기존 `uav_search.py`와 POS, 자세, 탐색 면적이 완전히 같다(차이 0). 이탈을 켜도 질량 보존 오차는 1e-9 미만이다. 환경만 돌리면 약 710 스텝/s(맥, 코어 하나)이고, greedy 한 판은 약 6 s, tree 한 판은 약 46 s다.
- **발견 1:** greedy는 초기 조건에 매우 민감하다. 시작 heading을 0.01° 바꾸면 최종 POS가 0.36에서 0.002로 바뀐다. 그래서 비교는 평가 세트 전체의 평균과 분산으로 본다.
- **발견 2 (POD 0.3, tree, 33.3 m/s 설정):** 탐색 보너스를 키울수록 탐색 면적은 늘고 POS는 떨어진다. s1000001 / s1000004 최종 POS는 다음과 같다.
  - density 0: 0.178 / 0.385
  - 0.01, 사전확률 영역: 0.196 / 0.385
  - 0.1, 사전확률 영역: 0.121 / 0.343
  - 0.1, 원판(11.1 m/s): 0.066 / 0.121
  
  greedy는 같은 조건에서 0.009 / 0.335였다. 원판 반경 기본값(0.75 m/s)은 보행자 모델용이라 차량 표적에는 맞지 않는다.

## 4단계: PX4 기준 고정익 모델과 에너지

### 설정 변경과 평가 세트 재생성

- **`SearchConfig` 기본값:** `uav_speed_mps = 15`, `dt_s = 4`, `flight_min = 60`, `pod = 0.3`. `max_turn_deg`는 아래 `check_plant.py`로 정한다.
- **`ScenarioConfig` 기본값:** `start_dist_m = (5000, 15000)`, `speed_kmh = (15, 30)`, `flight_s = 3600`, `uav_speed_mps = 15`.
- **`TargetConfig` 기본값:** `duration_s = 4200`, `speed_cap_mps = 30/3.6`.
- **따라잡기 확인:** 가장 불리한 경우(15 km, 30 km/h로 도주)에도 15000 / (15 − 8.3) ≈ 2240 s로 60분 안이다.
- **평가 세트와 분포 확인을 새 설정으로 다시 만든다.** 출발점이 가장자리이고 LKP가 15 km 이내라 LKP가 가장자리 15 km 띠에 몰리고, 이탈 질량이 늘 수 있다. 분포 그림에서 이탈 질량 중앙값과 상위 10%를 확인한다. 문제가 되면 맵 크기(예: 40 km)나 표적 heading 조건을 다시 정한다.
- 33.3 m/s 설정은 `--preset fast` 같은 묶음으로 남겨 비교에 쓸 수 있게 한다.

### FixedWingPlant

PX4 고정익 제어 계층 순서를 따른다. 파라미터 이름은 PX4와 같게 해서, 로드맵 4번에서 Gazebo ulog로 그대로 맞출 수 있게 한다.

```mermaid
flowchart LR
  Turn["turn command per step"] --> Course["course setpoint = heading + turn"]
  Course --> Npfg["NPFG guidance approx: NPFG_PERIOD, NPFG_DAMPING"]
  Npfg -->|roll setpoint| Roll["roll loop: FW_R_TC lag, FW_R_RMAX, FW_R_LIM"]
  Roll --> Turning["coordinated turn: psi_dot = g tan(phi) / V"]
  Turning --> Pose["x, y, psi"]
```

- **상태:** (x, y, ψ, φ, r_cmd). V는 대기 속도이고, 6단계 전까지는 지면 속도와 같다.
- **유도 근사 (NPFG):** 목표 진행 방향 χ_cmd = ψ + δ를 2차 시스템으로 추종한다. ω = 2π / `NPFG_PERIOD`, ζ = `NPFG_DAMPING`일 때 `dr_cmd/dt = ω²·wrap(χ_cmd − ψ) − 2ζω·r_cmd`이고, 롤 명령은 `φ_cmd = atan(r_cmd V / g)`를 `FW_R_LIM`에서 자른 값이다. 이건 근사이고, 정확한 형태는 Gazebo 로그로 맞춘다.
- **롤 루프:** `dφ/dt = (φ_cmd − φ) / FW_R_TC`, `|dφ/dt| ≤ FW_R_RMAX`.
- **선회:** `dψ/dt = g tan φ / V`, `dx/dt = V cos ψ`, `dy/dt = V sin ψ`. 내부 적분은 0.1 s(결정 간격 4 s당 40회)다.
- **초기값 (rc_cessna):** `FW_AIRSPD_TRIM = 15`, `FW_R_LIM = 50°`, `FW_R_TC = 0.4 s`, `FW_R_RMAX = 56.15°/s`, `NPFG_PERIOD = 12 s`, `NPFG_DAMPING = 0.7`.
- **max_turn:** 직진 상태에서 최대 명령을 줬을 때 한 스텝(4 s) 동안 실제로 바뀌는 heading을 `check_plant.py`로 재서 정한다. 이 값을 플래너 후보 격자, env 자르기, lawnmower 유도에 똑같이 쓴다. 롤 50°의 이론 선회율(15 m/s에서 약 45°/s)보다 NPFG 주기 때문에 훨씬 작을 것으로 예상한다.
- **환경 연결:** `SearchEnv(plant=FixedWingPlant(...))`로 바꿔 끼운다. 탐지는 스텝 끝의 실제 자세로 한다. 관측에 φ와 r_cmd를 추가한다. 롤이 짐벌 범위(45°)를 넘으면 `NadirSensor`가 경고한다.

### 에너지 모델 (선회 시 소비 증가)

- **동력:** \(P(\phi, V) = P_{trim}\left[(1-k)\left(\frac{V}{V_{trim}}\right)^3 + k\,\frac{V_{trim}}{V}\cdot\frac{1}{\cos^2\phi}\right]\)
  - k는 순항 때 유도 항력 비율이고, 기본 0.5(최소 항력 속도 근처)다.
  - 롤 35°에서 1.25배, 50°에서 1.71배다.
  - 속도 항은 6단계 일정 맞추기에서 쓰고, 그 전까지는 V = V_trim이다.
- **예산:** \(E_{budget} = P_{trim} \times 3600\ \text{s}\)이다. 동력은 정규화(P_trim = 1)해서 쓴다.
- **차감:** 매 스텝 내부 적분 동안 \(\sum P(\phi)\,dt\)를 뺀다. 이동 구간도 포함한다.
- **env 변경:** 에너지를 다 쓰면 `truncated`다. 정보와 로그에 남은 에너지와 소비량을 넣는다. 비교 지표에 "에너지 대비 POS"를 추가한다.
- **Gazebo 맞추기 (로드맵 4번):** SITL 배터리는 시간 기반으로 줄어드는 걸로 알고 있어서, 선회 항력이 반영되지 않을 가능성이 크다. 대신 ulog의 롤 대비 추력(TECS가 선회 중 올리는 추력)으로 k를 맞춘다.

### check_plant.py 검증

- 진행 방향 계단 응답: χ_cmd를 30°, 90°, 180° 계단으로 줬을 때 ψ(t)와 φ(t)를 그린다. 상승 시간, 오버슈트, 최대 롤을 기록한다.
- 일정 선회: 롤 한도에서 정상 선회 반경이 \(V^2/(g\tan\phi_{max})\)와 1% 안에서 맞는지 본다(15 m/s, 50°에서 약 19 m).
- 같은 turn 명령열(직진, 최대 선회, 좌우 교대)에서 `KinematicPlant`와 `FixedWingPlant`의 궤적과 에너지를 겹쳐 그린다.
- 결과로 `max_turn_deg`를 정하고 plan에 기록한다.

### 4단계 완료 기준

- 검증 그림 세 장과 `max_turn_deg`가 나온다.
- 새 설정의 평가 세트와 분포 그림이 나온다.
- greedy와 tree를 `KinematicPlant`와 `FixedWingPlant` 위에서 돌려 POS와 에너지 차이를 표로 남긴다.

### 4단계 구현 (모듈)

```
rl_search/
  config.py       프리셋: PlantConfig(PX4 이름), EnergyConfig, HandoverConfig, ExportConfig, FollowConfig,
                  LawnmowerConfig, WindConfig + 기존 SearchConfig/ScenarioConfig. rc_cessna(기본), fast.
                  --preset, --set 섹션.이름=값, meta.json에 저장한 프리셋으로 다시 만들기
  plants.py       KinematicPlant(기존과 같은 계산), FixedWingPlant(NPFG 근사 + 롤 루프), measure_max_turn, PLANTS 등록표
  energy.py       PowerModel: P(φ, V), step_energy, endurance_s
  wind.py         WindModel(일정 바람 + OU 돌풍, 상태 저장/복원), ConstantWind
  env.py          SearchEnv(plant, power, handover, wind), snapshot/restore, make_env(preset, ...)
  scripts/check_plant.py   검증 그림 3장 + plant_check.json
```

- **설정 바꾸기:** 값은 `--set plant.FW_R_LIM=45 search.pod=1`처럼 바꾼다. 같은 값이 두 곳에 있지 않게, 순항 속도는 `plant.FW_AIRSPD_TRIM`, 비행 시간은 `energy.budget_s`에서만 정하고 나머지(`search.uav_speed_mps`, `flight_min`, 시나리오 속도)는 `finalize`가 맞춘다. `search.max_turn_deg = None`이면 기체 모델로 잰다.
- **부품 바꾸기:** 새 기체 모델은 `plants.PLANTS`, 새 플래너는 `planners.PLANNERS`, 새 비교 방법은 `runner.METHODS`에 등록한다. 기체는 `step(turn, airspeed, wind)`, `get_state/set_state`, `obs_extra`만 맞추면 된다.
- **회귀:** `fast` 프리셋에서 `check_env.py`가 기존 `uav_search.py`와 POS, 자세 차이 0, 질량 보존 오차 2e-10으로 그대로 통과한다. 0~2단계 평가 세트는 `data/scenarios/fast/`로 옮겼다.

### 4단계 결과

**평가 세트 (rc_cessna, `data/scenarios/rc_cessna/`):** 평가 50개, 분포 확인 100개. 시나리오 하나 생성은 약 52 s(코어 하나)다.

| 지표 | 중앙값 | 상위 10% | 최대 |
|------|--------|----------|------|
| 맵 밖 이탈 질량 | 2.1% | 66.5% | 98.9% |
| 따라잡는 시간 | 790 s | 1020 s | 1200 s |
| 출발점-LKP 거리 | 11.6 km | 14.4 km | 15.0 km |

- 이탈 질량이 50%를 넘는 시나리오는 17%로, 33.3 m/s 설정(20%, 중앙값 1.5%)과 비슷하다. LKP가 가장자리 15 km 띠에 몰려도 이탈이 크게 늘지 않아서 60 km 맵을 유지한다.
- 샘플러는 LKP에서 `start_dist_m` 범위 안에 가장자리 출발점이 있는지를 네 변까지의 정확한 거리로 걸러 낸다. 그래서 출발점 후보가 없는 LKP가 0개다(`fast` 프리셋에서는 예전 LKP가 그대로 나온다).

**check_plant.py (`data/plant_check/rc_cessna/`):**

| 진행 방향 계단 | 상승 시간 | 오버슈트 | 2% 정착 | 최대 롤 |
|----------------|-----------|----------|---------|---------|
| 30° | 3.7 s | 7.9% | 11.6 s | 11.4° |
| 90° | 3.7 s | 7.9% | 11.6 s | 31.2° |
| 179° | 3.8 s | 7.8% | 11.7 s | 49.8° |

- 정상 선회 반경 19.257 m, 이론값 19.252 m(오차 0.03%, 1% 기준 통과).
- **max_turn = 133.35° / 4 s**(최대 선회율 44.6°/s). 예상과 달리 NPFG 주기 때문에 크게 줄지 않았다. 롤 루프(시정수 0.4 s, 56°/s)가 빨라서 1초 안에 롤 한도에 닿고, 그 뒤는 이론 선회율(15 m/s, 50°에서 45°/s)로 돈다. 이 값이 `mission.MAX_TURN_RC_CESSNA` 기본값이고, 프리셋은 `None`으로 두어 기체 설정이 바뀌면 다시 잰다.
- 큰 선회에서 롤이 49.8°까지 가서 짐벌 범위(45°)를 넘는다. `NadirSensor`가 경고를 한 번 낸다. 실제 greedy 계획의 최대 롤은 41° 정도라 지금은 계획용 롤 한도를 따로 두지 않았다(`--set plant.FW_R_LIM=45`로 바꿀 수 있다).

**KinematicPlant 대 FixedWingPlant (같은 turn 명령열 60 s):**

| 명령 | 위치 차이 평균 | 최대 | 에너지 Kinematic | 에너지 FixedWing |
|------|----------------|------|------------------|------------------|
| 직진 | 0 m | 0 m | 60.0 | 60.0 |
| 최대 선회 | 56 m | 100 m | 83.8 | 77.9 |
| 좌우 교대 | 206 m | 403 m | 83.8 | 77.6 |

에너지 단위는 순항 1초 = 1(trim-s)이다. Kinematic은 롤이 즉시 바뀐다고 보므로 선회 에너지를 더 크게 잡는다. 좌우 교대처럼 방향을 자주 바꾸면 롤 지연 때문에 궤적이 크게 달라진다(최대 403 m). 이 차이는 플래너가 예측에 쓰는 운동 모델과 실제 기체의 차이이기도 하다(아래 "남은 문제").

## 5단계: 이동 중 계획 완성과 경로점 인계

```mermaid
flowchart LR
  Launch["t = 0: launch on map edge"] --> Transit["transit: auto-steer to LKP, sensor on"]
  Transit -->|"t_plan (default 120 s)"| Handover["handover pose and time"]
  Handover --> PlanRun["planner or RL rolled out in sim to energy end"]
  PlanRun --> Waypoints["thinned waypoints with ETA"]
  Waypoints --> Px4["PX4 mission, NPFG follows path"]
```

- **이동 단계:** env가 heading을 LKP로 자동으로 맞춰 비행한다. 탐지와 에너지 차감은 이동 중에도 계산한다. 정책과 플래너는 `t_plan`(설정값, 기본 120 s)부터 행동한다. env의 `reset`은 인계 시점의 관측을 돌려준다. 그래서 RL과 플래너가 보는 에피소드는 인계 후부터다.
- **계산 시간:** 플래너가 인계 시점 이후 전체 경로를 계산하는 데 걸린 시간을 재서 `t_plan`과 비교한다. 운용에서는 인계 시점의 예상 자세에서 계산을 시작해야 하므로, 계산 시작은 출발 시점의 자세로 인계 자세를 예측해서 한다(이동은 직진이라 예측이 정확하다).
- **"미리 짠 계획"이 손해가 아닌 이유:** "못 찾음"으로 바뀌는 입자 가중치는 비행 경로만으로 정해진다. 찾으면 임무가 끝나므로, 찾기 전까지의 계획은 미리 계산한 계획과 같다.
- **경로점 내보내기 (`scripts/export_plan.py`):**
  - 계획 경로(스텝마다의 자세와 시각)를 허용 오차(기본 15 m) 안에서 솎는다. 시간은 경로점마다 보간해서 붙인다.
  - CSV(x, y, 계획 도착 시각)와 QGC `.plan`(NAV_WAYPOINT)을 저장한다. 지역 좌표를 위경도로 바꾸는 기준점은 설정값이다.
  - 경로점 수가 PX4 mission 한도(설정값)를 넘으면 경고한다.
- **실행 확인:** 솎은 경로점을 경로 추종 모드(NPFG 근사로 경로를 따라감)로 다시 비행해서, 계획 POS와 실행 POS, 시간 어긋남(계획 도착 시각 대비)을 비교한다.
- **완료 기준:** 평가 세트에서 greedy와 tree의 계획 POS 대비 실행 POS 손실이 표로 나오고, 경로점 파일이 만들어진다.

### 5단계 구현 (모듈)

```
rl_search/
  waypoints.py    Waypoints(xy, eta), thin_path(Douglas-Peucker), 위경도 변환, qgc_plan, write_plan
  follower.py     PathFollower: 경로 위 진행량 투영(창 제한) + lookahead 추적점, 바람 보정, 도착 시각 맞추기
  planners.py     rollout(env, planner, until_s): 인계 시점부터 계획 펼치기. run_episode는 그 위에 있다
  runner.py       plan_planner, follow, run_method(straight / lawnmower / greedy / tree), summary
  scripts/export_plan.py    한 시나리오: 계획 -> 솎기 -> mission.csv / mission.plan -> 경로 추종으로 다시 비행 -> 그림, 요약
  scripts/sitl_fly_plan.py  (fw-gz) .plan을 PX4 SITL에 올려 비행하고 궤적 비교
envs/run_sitl_headless.sh   화면 없는 PX4 인스턴스 N(포트 14540+N, 따로 된 Gazebo)
```

- **실행 순서:** 계획용 env에서 출발 → 이동 → 인계 → 플래너를 에너지 끝까지 펼친다. 실행용 env는 같은 시나리오로 다시 시작해 솎은 경로점을 `PathFollower`로 따라간다. 둘 다 같은 기체·에너지·센서 모델이라 차이는 솎기와 경로 추종에서만 생긴다.
- **`.plan` 형식:** `NAV_TAKEOFF`(이륙, 피치 15°) → `NAV_WAYPOINT`(수락 반경 15 m, 고도 100 m) → 착륙 패턴(`DO_LAND_START`, 홈 앞 500 m에서 `NAV_LOITER_TO_ALT` 50 m, `NAV_LAND`). PX4 고정익은 `MIS_TKO_LAND_REQ 2`라 착륙 패턴이 없으면 시동이 거부된다. `--set export.finish=rtl`이나 `none`으로 바꿀 수 있다.
- **경로 추종:** 계획 경로가 스스로 겹치는 곳에서 다음 바퀴로 건너뛰지 않도록, 진행량은 한 스텝에 `1.5 (V_max + |w|) dt + lookahead`까지만 늘린다. 추적점은 UAV에서 lookahead(90 m) 이상 떨어진 첫 경로점이라, 선회 반경보다 작은 고리는 돌지 않고 잘라서 지나간다(돌면 수백 초씩 늦어졌다).

### 5단계 결과 (s1000000, greedy, POD 0.3)

- **인계:** 120 s에 인계. 출발 시점에 예측한 인계 자세와 실제 차이 0 m(직진이라 정확). 이동 중 탐지 0, 에너지 120 trim-s.
- **계산 시간:** 54.8 s로 `t_plan` 120 s 안이다. tree는 한 판에 약 256 s라 `t_plan`을 넘는다(아래 비교 표의 계산 시간 참고). 운용에 쓰려면 tree의 beam/horizon을 줄이거나 `t_plan`을 늘려야 한다.
- **솎기:** 799 자세 → 659 경로점, 최대 편차 15.0 m(허용 15 m).
- **미션 한도:** 661 항목으로 실기체 한도 500을 넘는다(경고). greedy 경로는 4 s마다 방향을 바꿔서 솎아도 많이 남는다. 허용 오차를 30 m로 올리거나, 비행 중 미션을 나눠 올리거나, 고리를 `LOITER` 항목으로 바꾸는 방법이 남아 있다.
- **계획 대비 실행:** POS 0.1549 → 0.1422(−8.2%), 에너지 3601 → 3602 trim-s. 일정 오차 평균 −57 s, 최대 180 s, 끝 +76 s. 경로 이탈 평균 19.9 m, 최대 84 m. 최대 롤 41°.

## 3단계: lawnmower 베이스라인과 비교

- **시작:** 인계 시점부터 시작한다. 그전 이동 구간은 다른 방법과 같다.
- **탐색 영역:** 인계 시점부터 에너지가 끝날 예상 시각까지 맵 안에 남은 입자 위치를 합쳐, 80%를 담는 가장 작은 영역(HDR)을 구한다. 이를 표적 heading 방향으로 정렬한 사각형으로 감싸고 맵 안으로 자른다. 영역은 인계 시점에 고정한다.
- **왕복 패턴:** 줄 간격은 footprint 폭 115 m다. 건너뛸 줄 수는 `k = max(1, ceil(2R_eff / 115))`이고, R_eff는 4단계에서 잰 실효 선회 반경이다. 15 m/s에서는 k = 1(바로 옆 줄)이 될 것으로 예상한다. 영역을 다 덮으면 반복한다.
- **형태:** lawnmower는 처음부터 경로점 목록으로 만들고, 같은 경로 추종 모드로 비행한다. 그래서 운용 형태가 RL 계획과 같다.
- **`compare.py`:** 평가 세트에서 straight, lawnmower, greedy, tree를 돌린다. 조건은 POD 1과 0.3, 같은 에너지 예산이다. 지표는 시간에 따른 POS, 에너지에 따른 POS, 최종 POS, 1 - escaped 대비 비율, 탐색 면적, 계산 시간이다. 설정 조합별 결과를 `search_<설정 이름>/`으로 저장해서 뷰어에서 겹쳐 볼 수 있게 한다. 시나리오별 그림과 평균/분산 표를 만든다.
- **완료 기준:** 평가 세트(맥 5개, 우분투 50개)에서 비교 그림과 표가 나온다.

### 3단계 구현 (모듈)

```
rl_search/
  lawnmower.py    HDR 80% 사각형(표적 heading 정렬), 줄, 건너뛰기, 왕복 경로, build_lawnmower
  scripts/compare.py    --methods, --configs NAME:k=v, --wind-modes, --workers, --save-missions
```

- lawnmower는 경로점 목록으로 만들고 같은 `PathFollower`로 비행한다. 줄 간격은 footprint 폭(115 m), 건너뛸 줄 수는 `k = max(1, ceil(2 R_eff / 115))`이다. 15 m/s, 롤 50°에서 실효 반경이 약 19 m라 k = 1(바로 옆 줄)이다.
- `compare.py`는 시나리오 × 방법 × 설정 × 바람 방식을 프로세스 풀로 돌린다. 워커마다 BLAS 스레드는 1개다. 결과는 `data/compare/<tag>/`에 `table.md`, `table.csv`, `pos_time.png`, `pos_energy.png`로 남는다.

### 3단계 결과 (연기: 2시나리오 스모크, POD 0.3)

평가 50개 전체 비교는 `data/compare/methods/`에 돌아가는 중이다. 스모크(2개)에서 최종 POS 평균은 다음과 같다.

| 방법 | POS | 계산 시간 |
|------|-----|-----------|
| straight | 0.016 | ~1 s |
| lawnmower | 0.006 | ~1 s |
| greedy | 0.130 | ~51 s |
| tree | 0.173 | ~256 s |

- lawnmower POS가 낮은 이유: 인계 시점의 80% HDR을 표적 heading 사각형으로 감싸면 면적이 커진다(예: HDR 59 km² → 사각형 172 km², 줄 103개, 한 바퀴에 약 85,000 s). 에너지 60분으로는 일부를 훑을 뿐이다. 직진보다도 낮다. 베이스라인으로는 약하고, 사각형을 더 작게 하거나 줄 간격을 넓히는 자리가 `lawnmower.py`에 있다.
- tree는 POS가 가장 높지만 계산이 `t_plan` 120 s를 넘는다. 운용에 쓰려면 beam/horizon을 줄이거나 계획을 더 일찍 시작해야 한다.
- greedy는 POS와 계산 시간의 균형이 가장 낫다. 초기 heading에 민감하므로(2단계 발견) 평가는 세트 평균으로 본다.

## 6단계 (마지막): 바람 반영

- **바람 모델:** 기체에 에피소드마다 무작위 일정 바람(예: 0~6 m/s)과 약한 돌풍을 넣는다. 지면 속도 = 대기 속도 + 바람이다.
- **세 방식을 비교한다:**
  - (a) 바람을 모르는 계획을 그대로 실행
  - (b) 추정 바람을 넣어 계획 + 대기 속도(약 11~20 m/s)로 일정 맞추기. 에너지 모델의 속도 항을 사용한다.
  - (c) (b)에 일정 오차가 기준(예: 30 s)을 넘으면 현재 상태에서 다시 계획하기를 추가
- **결과:** 각 방식의 POS 손실과 에너지 차이를 표로 남겨, 실제 비행에 어떤 장치가 필요한지 정한다. RL 학습 때 바람을 무작위로 섞을지도 이 결과로 정한다.

### 6단계 구현 (모듈)

```
rl_search/
  wind.py      WindModel(에피소드마다 일정 바람 0~6 m/s + OU 돌풍), ConstantWind, 상태 저장/복원
  follower.py  추정 바람이 있으면 게걸음(crab) 보정, schedule=True면 대기 속도 11~20 m/s로 도착 시각 맞춤
  runner.py    WIND_MODES = none / unaware / aware / replan
               replan: 일정 오차 > 30 s이고 직전 재계획과 120 s 이상 떨어지면 현재 상태에서 다시 계획
```

- **none:** 바람 없음(비교 기준). **unaware:** 바람은 있는데 계획은 바람이 없다고 짠다. **aware:** 추정 바람을 넣어 계획하고, 경로 추종이 게걸음과 대기 속도로 일정을 맞춘다. **replan:** aware에 일정 오차 재계획을 더한다.
- 스모크(재계획, POD 0.3): greedy 0.197, tree 0.198. tree 재계획은 한 판에 약 26분이 걸린다. 평가 20개 × 네 방식 비교는 `data/compare/wind/`에 이어서 돈다.

## 최종 검증: PX4 SITL

```
envs/run_sitl_headless.sh 1          # 인스턴스 1, 화면 없는 Gazebo (GZ_PARTITION fwverify_1)
python rl_search/scripts/sitl_fly_plan.py <dir>/mission.plan --url udpin:0.0.0.0:14541
python -m rl_search.scripts.plot_sitl <dir>
envs/run_sitl_headless.sh stop 1
```

- 14분으로 자른 greedy 경로(`data/plans/rc_cessna/s1000000/greedy_14min/`, 102 경로점 + 이륙 + 착륙)를 인스턴스 1(포트 14541)에 올렸다. 0번(지금 켜 둔 GUI PX4)은 건드리지 않는다.
- 시동은 첫 시도에서 된다. 이륙 후 100 m, 15 m/s로 첫 긴 구간(약 5 km, ETA 336 s)을 간다.
- **착륙 중단:** `gz_rc_cessna`에는 거리 센서가 없다. 기본 `FW_LND_ABORT 3` + `FW_LND_USETER 1`이면 "지형을 못 찾음"으로 착륙을 멈추고 착륙점 위에 선회한다. `sitl_fly_plan.py`는 이 인스턴스에서만 `FW_LND_ABORT 0`, `FW_LND_USETER 0`으로 바꾼다. 착륙 패턴은 `DO_LAND_START` + `NAV_LOITER_TO_ALT`(홈 앞 500 m, 50 m) + `NAV_LAND`다.
- **데이터링크 끊김:** `NAV_DLL_ACT 2`는 GCS가 사라지면 RTL이다. 스크립트는 GCS 하트비트를 보내고, 이 인스턴스에서만 `NAV_DLL_ACT 0`으로 바꿔 스크립트가 끝나도 RTL이 안 나게 한다.
- 궤적은 `sitl_track.csv`, 숫자는 `sitl_report.json`, 그림은 `sitl_track.png`다. 검색 구간이 끝난 뒤의 경로 이탈·도착 시각 차이·착륙 여부를 본다.

**s1000000 greedy 14분 SITL 결과 (인스턴스 1, 이륙→검색→착륙):**

| 지표 | 값 |
|------|-----|
| 도달한 검색 경로점 | 102 / 102 |
| 검색 시간 | 763 s (계획 840 s, 76 s 빠름) |
| 경로 이탈 평균 / 상위 5% / 최대 | 2.4 m / 8.5 m / 41.6 m |
| 착륙 | 됨, 시동 꺼짐. 홈에서 62 m, 고도 0.0 m |
| 전체 비행 | 1327 s (검색 후 착륙·선회 507 s) |

착륙 직후 `ESC failure` / `Attitude failure (roll)`은 시뮬 기체가 땅에 닿을 때 나오는 경고이고, 착륙 자체는 `Mission finished, landed` → `Disarmed by landing`으로 끝났다.

## 이번 범위에서 뺀 것

- RL 학습(관측 설계, 보상 조정, PPO 학습 일정). 4~5단계 결과를 보고 따로 계획한다.
- 포장/비포장 도로 지형과 차량 이동 모델. `surface` 층과 `move_model` 자리만 있다.
- PX4/Gazebo 브리지 구현(로드맵 3, 5번), ulog로 모델 맞추기(로드맵 4번), 여러 대 UAV. 5단계 경로점 파일과 4단계 PX4 파라미터 이름은 이를 위한 준비다.

## 파일 구조 (4~6단계 추가분)

```
rl_search/
  config.py          프리셋과 --set. 순항 속도·비행 시간의 단일 출처
  plants.py          KinematicPlant, FixedWingPlant, PLANTS
  energy.py          PowerModel
  wind.py            WindModel, ConstantWind
  env.py             SearchEnv(plant, power, handover, wind), snapshot/restore
  waypoints.py       솎기, CSV, QGC .plan (이륙 + 착륙 패턴)
  follower.py        PathFollower (진행량 창, lookahead, 게걸음, 일정)
  lawnmower.py       HDR 사각형 왕복
  runner.py          METHODS, WIND_MODES, run_method
  scripts/
    check_plant.py   기체 검증 그림
    export_plan.py   한 시나리오 계획 → 경로점 → 다시 비행
    compare.py       방법 × 설정 × 바람 비교
    sitl_fly_plan.py PX4 SITL에 .plan 업로드·비행·비교 (fw-gz)
    plot_sitl.py     sitl_track.png
envs/run_sitl_headless.sh
docs/diagrams/       research_flow, modules, architecture
```

설정은 `--preset` / `--set 섹션.이름=값`으로 바꾸고, 부품은 `PLANTS` / `PLANNERS` / `METHODS` / `WIND_MODES`에 등록한다.

## 남은 문제 / 다음

- **ulog로 기체 맞추기:** `FixedWingPlant`의 NPFG 근사와 k(유도 항력 비율)를 `gz_rc_cessna` 비행 로그의 롤·추력으로 맞춘다. 파라미터 이름은 이미 PX4와 같다.
- **tree가 느림:** 한 판 약 256 s로 `t_plan` 120 s를 넘는다. beam/horizon을 줄이거나, 플래너가 쓰는 운동 모델을 133° 격자에 맞춘다. 지금 플래너는 이산 turn을 쓰고, 실제 기체는 롤 지연이 있다(좌우 교대에서 궤적 차이 최대 403 m).
- **미션 항목 500 한도:** 60분 greedy는 661 항목이다. `export.tolerance_m`을 올리거나, 비행 중 미션을 나눠 올리거나, 작은 고리를 `LOITER`로 바꾼다. SITL 확인은 `--max-minutes`로 자른 경로로 한다.
- **lawnmower 베이스라인이 약함:** heading 정렬 사각형이 HDR보다 훨씬 크다. 사각형을 줄이거나 줄 간격을 넓히는 설정(`lawnmower.hdr`, `lane_spacing_m`)이 있다.
- **RL 학습:** 관측·보상·PPO는 이 파이프라인 위에 따로 계획한다. 환경은 이미 `reset`/`step`이고 인계 이후부터 행동을 받는다.
