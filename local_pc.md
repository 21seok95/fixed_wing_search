# local PC 안내 (2026-11-12 작성, 2026-11-13 R8 추가, 2026-11-16 R13 추가, 2026-11-17 충주 시험 추가)

이 문서는 local PC에서 이 프로젝트를 이어 가는 사람이나 에이전트(Claude Code 등)에게 남기는 안내입니다.
**지금 할 일은 000장(R18: 1~4대를 섞은 세계, 처음부터)입니다.** 00장(충주 시험), 0장(R13), 0′장(R8), 1~9장(R7)은 이전 안내입니다.
7장(작업 방식)과 8장(문제 해결)은 그대로 지킵니다.

## 000. 최신 (2026-11-24): R18 — 한 정책으로 1~4대 팀, 처음부터, 이륙 거리 커리큘럼

### 000-1. 그동안 맥에서 있었던 일 (local PC가 모르는 부분, 자세히는 `docs/rl_experiments.md`, `docs/multi_agent_design.md`)

| 실행 | 바꾼 것 | 결과 (지리산 v2j 3사례, RL ÷ SAROPS) |
|---|---|---|
| R15 | 학습 A: 충주 120 km 도로형 풀 4,000개, 이륙만, **이륙 거리 단계** 10 → 10~20 → 10~40 km, 지도 7장(1.6~102.4 km), 가시도 관측 없음 | 680만 1.82 |
| M1~M2c | 여러 대(같은 가중치) — 여러 가지를 한꺼번에 바꿔 무엇이 효과였는지 가릴 수 없었고, 평균 행동이 한쪽으로 쏠려 원 선회로 무너짐 | 멈춤 |
| R16 | "처음부터 다시": 다수 기체 환경(설계 m2)을 **1대**로, **좌우 대칭 정책**(`rl_search/sym_policy.py`: 평균 행동이 거울 관측에서 정확히 반대), 8비트 지도 | 600만 1.757 (같은 스텝 R15보다 평균 +0.07) |
| R17 | 2대 세계 20개, 설계 **m3**(아래), 처음부터, 맥에서 진행 중 | — |

- 설계 m3(`rl_search/rl_multi.py` DESIGNS): 기체가 6.4 km 지도에 남은 확률 1% 이상을 보면 "탐색 중"(그 전은 이동 중). 둘 다 이동 중이면 100 m 절대 금지 ·
  250 m부터 다가갈 때만 벌점, 하나라도 탐색 중이면 **500 m 절대 금지(필터, 35초 예측) · 1 km부터 벌점**(step: 500 m 안 1, 바로 밖 0.7 → 1 km 0.3, 그 밖 0),
  위험 구역 채널도 같은 step. 보상 = 100 × 1.5 × (팀이 찾은 확률 + 0.25 × 내가 찾은 확률) ÷ 걸음 전 남은 확률 + A′ 0.1 − 벌점.
- 다수 기체 학습은 `train_rl --teams k1 k2 ...`(세계마다 대수, 기체마다 학습 칸 하나, `rl_search/rl_multi_vec.py`), 팀 비행 채점은
  `scripts/multi_design/ma_eval.py`, 체크포인트를 따라가며 채점은 `ma_eval_watch.py`.

### 000-2. R18: 무엇을 왜 (사용자 결정, 2026-11-24)

| 항목 | R18 | 이유 / R16과 비교 |
|---|---|---|
| 대수 | **세계마다 대수 고정, 1·2·3·4대 세계를 8 : 6 : 5 : 4로 섞음**(SCALE × (8, 6, 5, 4), SCALE 10 = 230세계 · 510칸; 표본 약 15 / 22 / 27 / 30%) | 1~4대를 고루 쓸 예정. 한 세계가 비행마다 대수를 바꾸면 빈칸이 약 40% 생겨 낭비 |
| 시나리오 배분 | 대수마다 풀 전체를 자기 순서로 돎(`--pool-by-team`), 같은 시나리오를 바로 다시 날지 않음(`--reuse 1`) | 같은 시나리오를 여러 대수로 날아 다양성↑, 풀 절약 |
| 풀 | 충주 120 km 도로형 **15,000개**(씨앗 14,000,000 + i, 약 255 GB) | 업데이트 2,000번이면 세계 비행 약 11만 번 |
| 업데이트 | 칸마다 **154걸음**(R16과 같음) × 칸 수, **미니배치 4,096**(에폭당 약 19묶음 × 5 = R16의 12 × 5와 비슷한 경사 단계), 학습률 1e-4, KL 0.02 | R16과 같은 리듬 |
| 길이 · 주기 | **업데이트 수로**(`--in-updates`): 단계 평가 32번마다, 지리산 검증 · 체크포인트 130번마다, 단계당 최대 487번, **총 2,000번** | R16의 10만 / 40만 / 150만 / 600만 스텝을 업데이트로 |
| 커리큘럼 | R16과 같은 규칙: 지리산 단계 사례(v2j1, v2j2)에서 정책을 **혼자** 날려 SAROPS의 0.9배(최근 3번)면 다음 단계 | 1대 세계가 섞여 있어 혼자 나는 것도 학습하는 상황 |
| 나머지 | 설계 m3, 좌우 대칭, 이륙 5~10초 간격, 8비트, γ 0.995, 엔트로피 0.01, A′ 0.1, 처음부터 | R16 · R17과 같음 |
| 기계 | GPU 1장(L40S, `GPU=0`), 세계 수는 규모 시험(000-3 4번)으로 정함, 프로세스당 세계 `W` 1(기본) | |

시작 스크립트 `rl_search/scripts/rl_eval/r18_start.sh`(설정이 모두 여기 있음), 실행 이름 `r18_mix`.

### 000-3. 순서와 확인 기준 (bash, 저장소 최상위; 단계마다 결과를 적어 두고, 다르면 멈추고 보고)

1. **[ ] 코드:** `git fetch && git checkout local-pc && git pull`, 가상환경 켜기. `python -c "import torch; print(torch.cuda.device_count())"` → 2.
   웹 서버는 끔(메모리).
2. **[ ] 지형** (고도 · 토지피복을 내려받음):
   ```bash
   python -m rl_search.scripts.make_real_terrain --region chungju120 --cell 25
   python -m rl_search.scripts.make_real_terrain --region jirisan120 --cell 25
   python -m rl_search.scripts.terrain_fingerprint check --region chungju120 jirisan120
   ```
   확인: 두 줄 모두 `same as the Mac (18 arrays)`. 다르면 학습하지 말고 보고.
3. **[ ] 학습 풀 15,000개** (하나 약 4초 · 코어 1개, 17 MB):
   ```bash
   python -m rl_search.scripts.rl_eval.make_pool --v2 --kind mountain:road --region chungju120 --n 15000 --workers 360
   python -m rl_search.scripts.rl_eval.pool_fingerprint check --region chungju120
   ```
   확인: `15000 scenario files here` 이상 · `PASS`(맥에서 만든 4,000개 중 40개 표본과 같음).
4. **[ ] 규모 시험** (세계 수 정하기; SCALE마다 업데이트 4번, 평가 없음, 하나에 몇 분):
   ```bash
   python -m rl_search.scripts.rl_eval.scale_test --scales 2 4 6 8 10 11 --updates 4
   ```
   표: SCALE, 세계 · 칸 수, 초당 결정, 업데이트 한 번 시간, 가장 낮았던 남은 메모리. 마지막 줄 `proposed: SCALE ...`(메모리 15% 이상 남는 것 중 가장 빠른 것).
   초당 결정이 SCALE을 키워도 거의 안 오르면(메인 프로세스가 관측을 모으는 데 막힘) 작은 쪽을 고른다. 표를 보고에 넣는다.
5. **[ ] 짧은 시험** (단계 넘김 · 평가 · 저장이 도는지):
   ```bash
   TAG=r18_smoke SCALE=2 STOP=6 EXTRA="--ring-eval-every 1 --eval-every 3 --save-every 3 --ring-max 2" bash rl_search/scripts/rl_eval/r18_start.sh
   grep -E "multi-UAV|in-updates|launch ring|eval stage|checkpoint|stopped" rl_search/data/rl_real/r18_smoke_train.log
   python -m rl_search.scripts.multi_design.sym_policy_check exact --model $(ls -t rl_search/data/rl_real/r18_smoke/ckpt/model_*.zip | head -1)
   ```
   확인: `multi-UAV: worlds [...], N UAV slots`, `--in-updates: ... decisions per update`, `launch ring: stage 0 -> 1`, `eval stage 3 at ...`(지리산 검증 줄),
   `checkpoint`/`stopped at`, 대칭 시험 7줄 모두 PASS. 끝나면 `rm -rf rl_search/data/rl_real/r18_smoke*`.
6. **[ ] 학습 시작과 팀 채점** (SCALE은 4번에서 고른 값):
   ```bash
   SCALE=<4번 값> nohup bash rl_search/scripts/rl_eval/r18_start.sh > /dev/null 2>&1 &
   nohup python -m rl_search.scripts.multi_design.ma_eval_watch --run rl_search/data/rl_real/r18_mix --prefix r18 > rl_search/data/rl_real/r18_eval_watch.log 2>&1 &
   ```
   - 처음 10분: 로그 `rl_search/data/rl_real/r18_mix_train.log`에 업데이트가 쌓이는지, `multi/act_sat` 거의 0, `multi/act_mean` 0 근처, 메모리.
   - 첫 체크포인트(업데이트 130번)에 5번의 대칭 시험을 다시 돌려 7줄 PASS.
   - 끊기면 `RESUME=1 SCALE=<같은 값> nohup bash rl_search/scripts/rl_eval/r18_start.sh > /dev/null 2>&1 &`.
7. **[ ] 끝나면 가져올 것:** `python -m rl_search.scripts.rl_eval.export_run --run r18_mix`(설정, 검증 기록, TensorBoard CSV, 최고 · 마지막 모델) +
   `rl_search/data/real/amboy/rl_eval/report/ma_eval/r18_*.json`(팀 채점) + 로그 `r18_mix_train.log`, `r18_eval_watch.log` + 4번 표를 zip 하나로.

### 000-4. 볼 것과 비교 기준 (결론을 미리 내리지 말고 그림 · 표로 보고)

- 지리산 검증(혼자, SAROPS 대비): 학습 로그 `eval stage 3 at ...`. 비교: R16 같은 업데이트 수(R16 업데이트 130번 = 40만 스텝 1.32, 260번 1.41, 390번 1.59,
  520번 1.72, 780번 1.82, 1,950번 1.757).
- 단위 주의: 학습 로그의 `eval stage 3 ... RL x vs SAROPS y`와 단계 평가의 RL · SAROPS 값은 **점수 = 1.5 × POS**(A 끔)이고, 비율은 그 둘의 비입니다.
  `ma_eval`(팀 채점)의 `team POS`는 POS 그대로입니다.
- 팀 채점(1~4대, 채점용 팀 POS, `r18_eval_watch.log`). 맥의 기준선은 R16 600만을 대수만큼 각자(서로 모름, 입력은 "동료 없음") m3 규칙으로 날린 값입니다
  (`ma_eval --solo`, 사례 1 / 2 / 3, 평균):

  | 대수 | R16 각자 | 평균 | 참고 |
  |---|---|---|---|
  | 1 | 0.341 / 0.426 / 0.296 | 0.354 | R16 자신의 지리산 검증 점수 0.511 / 0.640 / 0.444의 1/1.5 |
  | 2 | 0.530 / 0.544 / 0.323 | 0.466 | 1 km 안 23~36%, 필터 개입 11~37번 |
  | 3 | 0.700 / 0.658 / 0.352 | 0.570 | 1 km 안 25~40%, 개입 33~163번 |
  | 4 | 0.517 / 0.619 / 0.666 | 0.601 | 1 km 안 42~93%, 개입 119~377번, **위반 1번(사례 1, 3)**: 서로 모르는 4대가 몰리면 안전한 명령이 없을 때가 있음 |
- 학습 로그의 대수별 줄 `team_k1..k4/POS`, `pen_share`(벌점 ÷ 탐색 보상), `close1km`, `viol`(0이어야 함), `cuts`.
- 처음 몇 백 번 업데이트 동안 대수별 POS가 같이 오르는지, 어느 대수가 처지는지.
- 주의(맥 R17에서 본 것): 학습한 적 없는 먼 이륙(31 km)에서 R17 40만이 입자 반대로 꺾은 뒤, 입자가 정뒤에 오자 직진만 함(대칭 정책은 거울 관측에서 행동이 반대라,
  정뒤처럼 거의 대칭인 관측에서는 행동이 0에 가까움). 이륙 거리 3단계(10~40 km) 전후의 검증에서 이런 비행이 있는지 볼 것.

## 00. 최신 (2026-11-17): 충주 가시도·가장자리 시험을 R13으로

### 00-1. 왜 하나 (맥에서 있었던 일)

- 사용자가 맥의 웹 화면에서 R14 도로형(r14_road 320만, R13과 같은 설정으로 도로형만 학습)을 **충주 지도**(학습하지 않은 지도: 숲과 산이 많고 길이
  촘촘함)에 15번 날림. 입자 분포와 상관없이 **작은 원을 그리며 맴도는** 비행이 많았음(10분 동안 실제로 이동한 거리가 날아간 거리의 20%도 안 되는 구간이
  1~13번 평균 45%).
- 맥에서 찾은 것(`docs/rl_experiments.md` 11-17 "원 선회 원인 분석"):
  - 정책에게 보이는 남은 에너지만 바꿔도(시간 정보) 맴돎은 그대로 → 시간 탓 아님.
  - 충주 지도 안쪽의 가시도가 0.05~0.3(숲)으로, 학습한 사막 지도(0.85~0.97)와 크게 다르고 지도 밖 값(0)과도 거의 같음.
  - 정책에게 가시도를 1(트인 땅)로 보여 주면(실제 탐지는 진짜 충주 값) 작은 원이 줄고 POS가 오름. 다만 덩어리 위의 큰 원은 남음.
  - 남은 큰 원의 후보 원인: 보는 보상 A′(0.1)는 입자의 남은 확률과 상관없이 입자 수를 세고, 지나갈 때마다 다시 주며, 반지름 약 1.3 km보다
    큰 원(초당 1.49° 이하)이면 받음.
- 사용자: **이 시험(특히 D)을 R13(세 유형 함께, 600만)으로도 확인**하고 싶다.

### 00-2. 네 가지 변형 (같은 사례 15개, 비행은 웹 화면과 같은 방식: `web_rl._run`)

| 변형 | 정책 관측의 가시도 | 실제 탐지의 지도 밖 가시도 | 지도 밖으로 나간 입자 |
|---|---|---|---|
| A 학습 그대로 | 안쪽 진짜 충주, 밖 0 | 가장자리 칸 값 | 사라짐 (학습과 같음) |
| B 지도 밖 직진 | 안쪽 진짜, 밖 0 | 가장자리 칸 값 | **나간 방향으로 계속 직진**, 믿음에 남음 |
| C 직진 + 밖 가시도 1 | 안쪽 진짜, **밖 1** | **1** | 직진 |
| D 직진 + 안팎 가시도 1 | **안팎 모두 1** | 1 | 직진 |

- D에서도 지도 **안쪽의 실제 탐지**는 진짜 충주 가시도라, 찾은 POS는 실제 땅에서 얻은 값입니다(바꾸는 것은 정책이 보는 입력뿐).
- 사례 15개: `rl_search/data/real/chungju/web_cases_20261117.json`(사용자가 웹에 넣은 표적 설정, LKP, 도주 방향, 이륙점, 그리고 맥의 R14 도로형 POS).
  14·15번은 입자 5,000개가 같은 방향·속도로 움직이는 특수 설정(사실상 한 점)입니다.
- 맥의 R14 도로형 결과(1~13번 평균): A POS 0.596, 맴돎 45% / D POS 0.670, 맴돎 24%. 그림 `rl_search/data/real/chungju/d_probe/r14_road_paths.png`,
  표 `r14_road_table.md`(이 브랜치에 들어 있음).

### 00-3. 순서와 확인 기준 (bash, 저장소 최상위)

1. **[ ] 코드 받기:** `git pull` (브랜치 `local-pc-r13`).
2. **[ ] 충주 지도:** `python -m rl_search.scripts.make_real_terrain --region chungju --cell 25` (원자료가 저장소에 있어 내려받지 않음).
   - 확인: `python -c "from rl_search.rl_real import preload; preload('chungju'); print('ok')"`
3. **[ ] 재현 확인 (R14 도로형, 맥과 같은 값이 나와야 함):**
   ```bash
   python -m rl_search.scripts.rl_eval.d_probe run --model rl_search/data/rl_real/r14_road/ckpt/model_3200000.zip --variant A
   ```
   - 15줄마다 `POS x.xxx (Mac R14 road x.xxx)`: **15개 모두 소수 셋째 자리까지 같아야 합니다.** 다르면 멈추고 보고합니다(지도나 코드가 맥과 다름).
     비행 하나에 맥에서 20초쯤.
4. **[ ] R13 네 변형:** R13 600만 모델 경로를 넣습니다(학습 폴더 이름이 다르면 그 경로). 네 개를 동시에 돌려도 됩니다(프로세스마다 메모리 약 1.5 GB).
   ```bash
   M=rl_search/data/rl_real/r13_2map/ckpt/model_6000000.zip
   for v in A B C D; do python -m rl_search.scripts.rl_eval.d_probe run --model $M --variant $v --web > d_probe_$v.log 2>&1 & done; wait
   ```
   - `--web`: 비행마다 웹 화면 목록에도 저장합니다(이름 끝 "r13_2map · D 직진 + 안팎 가시도 1" 등). 웹 서버를 켜 두었다면 바로 목록에 보입니다.
   - 확인: 로그마다 15줄, 오류 없음. 결과 `rl_search/data/real/chungju/d_probe/r13_2map_{A,B,C,D}.json`.
5. **[ ] 그림과 표:** `python -m rl_search.scripts.rl_eval.d_probe figure --run r13_2map` → `d_probe/r13_2map_paths.png`(사례 × 변형 경로, 경로 색은
   앞 1/3 파랑·가운데 보라·끝 1/3 검정), `d_probe/r13_2map_table.md`(POS / 맴돎 / 가장자리).
6. **[ ] 가져올 것:** `rl_search/data/real/chungju/d_probe/` 폴더 전체를 zip으로(json, png, md). 웹으로 본 눈에 띄는 비행도 적어 주세요.

### 00-4. 볼 것 (결론을 미리 내리지 말고 그림과 함께 보고)

- A에서 R13도 맴도는가(R14 도로형 45%)? 세 유형을 함께 배운 R13이 충주의 넓게 퍼진 사례(9~13번)에서 다르게 나는가?
- D에서 맴돎과 POS가 어떻게 바뀌는가(R14: 45 → 24%, 0.596 → 0.670)? 남는 맴돎은 덩어리 위 큰 원인가(A′ 후보)?
- C·D에서 지도 밖으로 나가 나간 입자를 쫓는가, 가장자리에 붙어 맴도는가(표의 가장자리 %, 그림의 빨간 점선 = 지도 경계)?
- 14·15번(한 점)은 참고만.

## 0. 최신 (2026-11-16): R13 — 다섯 번째 지도 102.4 km + A′ 0.1, 학습 설정은 R10

### 0-1. R8 뒤에 있었던 일 (local PC가 모르는 부분)

| 실행 | 어디서 | 바꾼 것 | 검증 (SAROPS 대비) |
|---|---|---|---|
| R8 | 맥 | A 끔, 지도마다 POS 정규화 | 300만 1.328 (mval 15) |
| R9 | 맥 | R8 + **A′**(보기 보상: 입자를 화면 가운데에 두고 초당 1.49° 이하로 완만하게 지날 때 통과마다 입자 개수 ÷ 5000 × 0.01) + 사막형 학습 비율 50% 고정 + 엔트로피 0.01 | 400만 1.245, 중간에 도로형 사례가 무너지는 흔들림 |
| R10 | 맥에서 준비, 우분투 PC에서 학습 | ① 표적이 멈추지 않음, 입자마다 고유 속도(길 종류 평균 × (1 + 0.15z), 0.7~1.3배) ② 지도 채널: 입자 위치(지금·lead, 무게 없음), 지금 POS(지도마다 0~1, 실제 값), 가시도, 카메라 ③ A 완전 제거(학습 풀에도 없음) ④ **두 지도**(암보이 + 바스토, 각 1000개), 경로 변수 무작위, 검증 `nval` 15(암보이) + `bval` 15(바스토) | 600만 1.236 |
| R11 | 우분투 | R10 400만에서 미니배치 770 · KL 상한 0.03으로 200만 더 | 1.279 (지금까지 최고) |
| R12 | 우분투 | R11의 PPO 설정으로 처음부터 600만 | 1.253 (R10과 같음 → PPO 업데이트 크기는 병목이 아님) |

- R10~R12의 비율은 검증 30개(nval + bval)의 RL hold-out POS 합 ÷ SAROPS POS 합입니다. SAROPS도 POS만으로 계획한 기준값(이름 `..._a0_w1.5`)이라 R8 이전 비율과 직접 비교하지 않습니다.
- R10~R12의 설정, 검증 기록, 모델, 학습 로그는 저장소에 있습니다(`rl_search/data/rl_real/r10_2map`, `r11_2map`, `r12_2map`, `r*_train.log`).
- 자세한 경과: `docs/rl_experiments.md`(R10~R13 줄, 11-14~11-16 메모), `docs/r10_ubuntu.md` 8~9장.

### 0-2. R13: 무엇을 왜 바꾸나 (사용자 결정, 2026-11-16)

맥에서 R12 600만을 분석해 찾은 것입니다(그림 `rl_search/data/real/amboy/rl_eval/report/r12_circle_*.png`, `r12_outside_map.png`).
- **끝 무렵 쓸모없는 원 선회:** 원 선회(앞뒤 3분 동안 한 방향 330° 이상)가 이륙 뒤 0~15 / 15~30 / 30~45 / 45~60분에 13 / 21 / 21 / 33%입니다.
  끝 무렵 원 선회 때는 3 km 안에 남은 확률이 평균 0.075로 적고, 남은 시간에 갈 수 있는 3~15 km에 0.248이 남아 있습니다.
  보상(남은 확률에 비례)이 끝 무렵 작아져, 결정 하나의 학습 신호가 첫 5분보다 약 7배 약합니다. A′는 그때 보상의 2~4%뿐이고, 초당 1.49°보다 빨리 도는 원에는 없습니다.
- **가장 큰 지도 밖의 확률:** 평균 0.8%지만 무인기가 지형 가장자리 5 km 안이면 13%, 최대 29%입니다. 도로형·혼합형 후반에 한 갈래 길을 따라 가장자리로 나갈 때 생깁니다.

| 바꾼 것 | 내용 |
|---|---|
| 다섯 번째 지도 | 102.4 km, 64×64(한 칸 1.6 km), 채널은 다른 지도와 같은 5개, lead 600초(`--extra-map-km 102.4 --extra-map-lead 600`). 벡터 12 → 13개, 신경망 가중치 135만 → 166만이라 처음부터 학습 |
| A′ 계수 | 0.01 → **0.1**(`--look-w 0.1`). R12 비행 기준으로 보상의 초반 약 10%, 끝 무렵 약 30%. 같은 입자를 3번째 볼 때부터 A′가 POS 보상보다 큼 |
| 학습 설정 | **R10과 같게**: 미니배치 256, KL 상한 0.02(R11·R12의 770·0.03이 아님), 업데이트 약 3,070 결정 |
| 그대로 | 두 지도·경로 변수, 정지 없는 표적, R10 지도 채널, A 없음, 사막형 50%, 엔트로피 0.01, 거리 인계 5 km, 검증 nval + bval, 600만 스텝 |

- **기대:** 끝 무렵에도 줄지 않는 A′가 원 선회 대신 입자 위를 곧게 지나가게 밀고, 먼 확률이 지도에 보입니다.
- **위험:** A′는 이미 훑은 입자도 세므로, 이미 훑은 짙은 무리를 다시 훑는 쪽으로 끌려 POS(점수)가 떨어질 수 있습니다. 두 변경을 한 실행에 묶었으므로 효과를 나눠 볼 수 없습니다(사용자 결정).
- 맥에서 확인한 것: 102.4 km 지도가 51.2 km 지도 밖의 입자를 보여 줌(분포에서 곧게 멀어지면 51.2 km 지도는 15분부터 놓치고 102.4 km 지도는 24분까지 99~100%), 작은 두 지도 풀로 2,048 스텝 학습과 nval·bval 평가 정상, 바뀐 코드로 R12 600만을 다시 채점해도 같은 값. 맥에서 실제 학습은 하지 않았습니다(풀을 만들다 멈춤).

### 0-3. 순서와 확인 기준 (bash, 저장소 최상위)

```bash
DEF="--area-def realtime_t --area-roll-max 5 --blur-tau 3 --blur-dead-roll 5 --blur-tau-low 15 --area-sigma 75 --area-sigma-off 150 --area-img-min 0.75 --pos-w 1.5"
NEW="--handover-km 5 --no-edge --a-weight 0 --fp-rate --no-area --obs-r10"            # R10~R13 공통 평가 옵션
R13="$NEW --look-w 0.1 --type-fixed desert=0.5 --ent-coef 0.01 --no-stop --extra-map-km 102.4 --extra-map-lead 600"
```

1. **[ ] 코드:** R13에 필요한 파일만 담은 브랜치 **`local-pc-r13`**을 새 폴더에 받습니다(커밋 1개, 받는 양 약 53 MB, 풀면 약 80 MB).
   ```bash
   git clone --branch local-pc-r13 --single-branch https://github.com/21seok95/fixed_wing_search.git fixed_wing_search_r13 && cd fixed_wing_search_r13
   ```
   - 들어 있는 것: 지도 만들기·점검·학습 풀·학습·평가·분석·내보내기 코드와 웹 보기(`vehicle_web`), 두 지도 원자료, SAROPS 기준값,
     R10·R12의 설정·검증 기록·TensorBoard·학습 로그, R12 600만 모델, R13 근거 그림, 이 문서들. R11, R10 모델, 그 밖의 실행과 그림은 `feat/rl-env`에만 있습니다.
   - 이 브랜치는 `feat/rl-env`와 역사가 이어져 있지 않으므로 합치지 않습니다. 결과는 9번의 zip으로 가져옵니다.
   - 맥에서 `feat/rl-env`의 변경을 이 브랜치로 옮겨 올리면, local PC에서는 `git pull`로 받습니다.
     local PC에서 코드를 고쳐 올렸다면, 맥에서 그 변경을 `feat/rl-env`에 먼저 옮긴 뒤 다음 갱신을 합니다(옮기는 스크립트가 덮어쓰기를 막습니다).
   - 설치는 `docs/run_on_pc.md` 1장에서 `git clone`·`git checkout` 줄만 빼고 그대로 합니다.
2. **[ ] 지도 두 개:** `python -m rl_search.scripts.make_real_terrain --region amboy --cell 25`, 같은 명령에 `--region mojave`(원자료가 저장소에 있어 내려받지 않음).
   - 확인: `python -c "from rl_search.rl_real import preload; preload('amboy'); preload('mojave'); print('ok')"`
3. **[ ] 점검:** `python -m rl_search.scripts.rl_eval.check_r10 --extra-map-km 102.4` → 마지막 줄 `ALL CHECKS PASSED`(지도 5장, 벡터 13).
4. **[ ] 학습 풀 두 개** (각 1000개, 장당 약 26.5 MB, 합계 약 53 GB):
   ```bash
   python -m rl_search.scripts.rl_eval.make_pool --multi --no-stop --no-area --route-var --n 1000 --workers 8 --region amboy
   python -m rl_search.scripts.rl_eval.make_pool --multi --no-stop --no-area --route-var --n 1000 --workers 8 --region mojave
   ```
   - 확인: `rl_search/data/real/{amboy,mojave}/rl_pool_multi_ns_rt/`에 각 1000개(도로·혼합·사막 약 333개씩).
   - 지도마다 90개 이상 차면 학습을 먼저 시작해도 됩니다. 학습 환경이 10분마다 풀 폴더를 다시 읽습니다(R10도 그렇게 함).
5. **[ ] SAROPS 기준값:** 저장소에 이미 있습니다. `rl_search/data/real/amboy/rl_eval/sarops_ref_nval_realtime_t5_d3z5l15_s75_o150_i0.75_a0_w1.5.json`, `rl_search/data/real/mojave/rl_eval/sarops_ref_bval_...`(같은 이름). 새로 만들지 않습니다.
6. **[ ] 비교용 R12 비행:** R12 600만(저장소에 있음)으로 검증 30개를 날려 웹 비행 기록을 만듭니다(원 선회 비교에 필요).
   ```bash
   for sp in nval bval; do python -m rl_search.scripts.rl_eval.eval_rl --model rl_search/data/rl_real/r12_2map/ckpt/model_6000000.zip \
     --label r12_2map-6000000 --split $sp --stages 3 --workers 4 $DEF $NEW; done
   ```
   - 사례별 점수가 `r12_2map/state.json`(우분투 학습 중 평가)과 POS 0.05 안팎 다를 수 있습니다. 정상입니다. 비행 하나는 행동에 1e-6만 달라도 갈려서,
     기계마다 부동소수점 차이로 달라집니다(맥에서 확인: 같은 모델에 1e-6 잡음을 넣으면 도로·혼합형 ±0.003~0.039, 사막형 ±0.002~0.008).
7. **[ ] R13 학습** (실행 이름 `r13_2map`, 로그는 `rl_search/data/rl_real/r13_train.log`):
   ```bash
   python -m rl_search.scripts.rl_eval.train_rl --tag r13_2map --stage 3 --n-envs 20 --envs-per-proc 2 --n-steps 154 \
     --batch-size 256 --target-kl 0.02 --eval-workers 2 --eval-every 400000 --save-every 500000 --stop-at 6000000 --device cuda \
     --mask-cost 0.1 --multi --eval-split nval bval --regions amboy mojave --route-var $DEF $R13 >> rl_search/data/rl_real/r13_train.log 2>&1
   ```
   - **바꾸지 말 것:** `--batch-size 256 --target-kl 0.02`, 업데이트 크기(환경 수 × `--n-steps` ≈ 3,070), `$R13` 옵션.
   - **기계에 맞게 바꿔도 되는 것:** 메모리가 모자라면 `--n-envs 10 --n-steps 307`(업데이트 크기 같음), `--envs-per-proc`, `--eval-workers`, `--device`.
   - 평가는 40만마다 30개입니다. 로그의 `eval stage 3 at <스텝>: ... (ratio ...)`와 `by type`, `by split` 줄을 봅니다. 끊기면 같은 명령에 `--resume latest`.
   - **진행 기준** (같은 스텝의 검증 비율, `state.json`에 있음):

     | 스텝 | 40만 | 80만 | 120만 | 160만 | 200만 | 280만 | 400만 | 480만 | 600만 |
     |---|---|---|---|---|---|---|---|---|---|
     | R10 두 지도 | 1.027 | 1.048 | 1.083 | 1.156 | 1.113 | 1.216 | 1.223 | 1.231 | 1.236 |
     | R12 | 0.968 | 1.040 | 1.094 | 1.156 | 1.163 | 1.168 | 1.175 | 1.215 | 1.253 |

     같은 설정의 실행 사이에도 ±0.05 출렁이므로 한 번의 차이로 판단하지 않습니다.
   - 시간 참고: 우분투(6코어, RTX 2070 SUPER)에서 R12 600만은 8시간, R10 400만은 5.6시간이었습니다.
8. **[ ] 분석:** 0-4장.
9. **[ ] 내보내기:** `python -m rl_search.scripts.rl_eval.export_run --run r13_2map`. 0-4의 그림과 표를 먼저 만들면 zip에 같이 들어갑니다. 보고 양식은 0-5장.

### 0-4. 분석할 것

1. **같은 스텝 비교** (R10은 학습 설정이 같아 관측·A′의 효과를 보고, R12는 지금까지의 기준):
   ```bash
   python -m rl_search.scripts.rl_eval.r12_compare --base r10_2map --run r13_2map --logs r10_2map_train.log r13_train.log --names R10 R13 \
     --descs "R10 두 지도" "R13 (R10 + 102.4 km 지도 + A′ 0.1)" --title "R13 대 R10 두 지도"
   python -m rl_search.scripts.rl_eval.r12_compare --base r12_2map --run r13_2map --logs r12_train.log r13_train.log --names R12 R13 \
     --descs "R12" "R13 (R10 설정 + 102.4 km 지도 + A′ 0.1)" --title "R13 대 R12"
   ```
   전체·지도별·유형별 비율과 사례별 짝 비교(POS 차이 ± 표준오차, 이긴 사례 수)가 나옵니다.
2. **끝 무렵 원 선회와 지도 밖 확률** (가장 중요한 판단 기준):
   ```bash
   python -m rl_search.scripts.rl_eval.circle_check --label r13_2map-6000000 --name "R13 600만" --map-km 102.4
   python -m rl_search.scripts.rl_eval.circle_check --label r12_2map-6000000 --name "R12 600만" --map-km 51.2
   ```
   - R12 600만 기준(맥): 원 선회 비율 13 / 21 / 21 / 33%(0~15 / 15~30 / 30~45 / 45~60분). 분당 POS 증가는 원 선회 중 / 나머지가 0.0255 / 0.0339, 0.0073 / 0.0102, 0.0020 / 0.0047, 0.0008 / 0.0021.
     지도 밖 확률 평균은 0 / 0.09 / 1.0 / 1.8%.
   - 웹 비행 기록이 남는 학습 스텝(80만, 120만, 160만, 200만, 280만, 320만, 360만, 400만, 480만, …, 600만)이면 중간 스텝도 같은 명령으로 볼 수 있습니다.
3. **A′ 몫:** TensorBoard `episode/look`과 `episode/POS`로 0.1 × look ÷ (1.5 × POS + 0.1 × look)을 계산합니다(R12는 0.01로 0.7~1.5%).
4. **위험 확인:** 이미 훑은 무리를 다시 훑는지 봅니다. 유형별 hold-out POS가 R10·R12보다 떨어지는지, 비행 경로가 같은 곳을 반복하는지 확인합니다.
   ```bash
   python -m rl_search.scripts.rl_eval.compare_flights --split nval --a "R12 600만=r12_2map-6000000" --b "R13 600만=r13_2map-6000000"
   python -m rl_search.scripts.rl_eval.compare_flights --split bval --a "R12 600만=r12_2map-6000000" --b "R13 600만=r13_2map-6000000"
   ```
5. **학습 신호:** TensorBoard의 유형별·지도별 학습 비행 POS, 행동 표준편차, approx KL을 R10과 비교합니다(r12_compare 그림에 일부 있음).
6. **웹으로 보기:** `python -m rl_search.scripts.real_map.vehicle_web` → 브라우저 `http://127.0.0.1:8000`. 학습이 끝난 뒤 켭니다(R12 첫 시작은 웹을 켜 둔 채 평가하다 메모리 부족으로 죽었음).
   - 실행 목록에서 검증 비행(`eval-nval-s3-k<번호>-rl-r13_2map-<스텝>`, `eval-bval-…`, 6번에서 만든 `…-rl-r12_2map-6000000`)을 다시 봅니다.
   - ② 패널에서 R13 체크포인트를 고르면 화면의 시나리오(암보이)로 새로 날려 봅니다.

### 0-5. 보고 양식 (zip을 못 꺼내면 손으로라도)

```markdown
# R13 local PC 보고 (작성일)
## 환경: 기계(CPU/코어/GPU/메모리), 바꾼 실행 옵션과 이유, 총 시간, 초당 스텝
## 검증 비율 (40만마다): 전체, nval, bval, 도로형, 혼합형, 사막형 — R10·R12 같은 스텝과 나란히
## 사례별 짝 비교: R13 − R10, R13 − R12 (POS 차이 ± 표준오차, 이긴 사례 수; 전체·지도별·유형별)
## 끝 무렵 원 선회 (circle_check 표): R13 600만과 R12 600만, 시간대별 원 선회 비율·분당 POS·지도 밖 확률
## A′ 몫 (시간대별이면 더 좋음)
## 눈에 띈 비행: 사례(지도·유형·번호), 이륙 뒤 분, 무엇이 이상했는지, 그때 남은 확률이 어디 있었는지
## 문제와 바꾼 것
```

## 0′. 이전 (2026-11-13): R8 — A 끄고 POS를 지도마다 정규화

R7을 local PC에서 돌린 결과, 사막형에서 탐색이 잘 되지 않았습니다(사용자 보고).
- 학습률과 엔트로피를 조절해 봤고, A 보상 가중치도 올려 봤지만 사막형은 여전히 원형으로 돌았습니다. 급선회는 아니었습니다.
- 사용자와 논의해 다음 실험을 **R8**로 정했습니다. 순서와 확인 기준은 아래 1~9장(R7)과 같고, 바뀌는 것만 적습니다.

| 바꾼 것 | 내용 | 이유 |
|---|---|---|
| A 끔 (`--a-weight 0`) | 보상은 1.5 × POS 증가 × 100만 씁니다. A 채널과 A 상태(벡터의 구역 비율)는 없습니다. A 값은 참고로만 재서 기록합니다 | A는 입자에서 나온 밀도라 새 정보가 없습니다. 또 칸 하나의 A 값이 "1 ÷ 구역 넓이"에 비례해서, 모인 경우 POS·A가 둘 다 크고 퍼진 경우 둘 다 작았습니다 |
| 지도마다 POS 정규화 (`--pos-norm`) | 지도 4장 각각의 POS 채널(지금·예측)을 그 지도의 가장 큰 칸 값으로 나눈 log 값(0~1)으로 바꿉니다. 나누는 값에는 바닥(지금 평균 가중치의 입자 3개분)을 둡니다. 지도마다 나눈 값(log)은 벡터에 4개 더 들어갑니다 | 사막형처럼 넓게 퍼지거나 후반에 많이 찾은 분포도 또렷하게 보이게 합니다. 바닥은 흩어진 입자 몇 개가 가장 밝은 칸이 되는 것을 막습니다 |
| 상대 POS 채널 뺌 (`--rel-maps` 없이) | 위 정규화와 겹칩니다 | |
| 그대로 | 처음부터 학습, 환경 10개 × n_steps 307, 카메라 채널 × 선회율 계수(`--fp-rate`), 거리 인계, 지도 밖 허용, 유형별 학습 비율 조정, 검증 `mval` 15 | |

**명령** (bash, 저장소 최상위)

```bash
DEF="--area-def realtime_t --area-roll-max 5 --blur-tau 3 --blur-dead-roll 5 --blur-tau-low 15 --area-sigma 75 --area-sigma-off 150 --area-img-min 0.75 --pos-w 1.5"
# SAROPS 기준값: SAROPS도 POS만으로 계획하고 채점 (이름 realtime_t5_d3z5l15_s75_o150_i0.75_a0_w1.5)
python -m rl_search.scripts.rl_eval.sarops_ref --split mval --stages 3 $DEF --a-weight 0 --workers 8
python -m rl_search.scripts.rl_eval.sarops_ref --split mtest --stages 3 $DEF --a-weight 0 --workers 8
# 학습 (R7과 같은 학습 풀 rl_pool_multi_o150 사용. A 정의는 참고용 A 값에만 쓰임)
python -m rl_search.scripts.rl_eval.train_rl --tag amboy_r8 --stage 3 --n-envs 10 --envs-per-proc 1 --n-steps 307 \
  --eval-workers 2 --eval-every 100000 --save-every 500000 --stop-at 3000000 --device cuda --mask-cost 0.1 --multi \
  --eval-split mval --handover-km 5 --no-edge $DEF --a-weight 0 --pos-norm --fp-rate
# 시험 30개 (R8, 그리고 R7을 같은 POS 기준으로)
python -m rl_search.scripts.rl_eval.eval_rl --model rl_search/data/rl_real/amboy_r8/best_stage3.zip --label r8_best --split mtest \
  --stages 3 $DEF --handover-km 5 --no-edge --a-weight 0 --pos-norm --fp-rate --workers 8
python -m rl_search.scripts.rl_eval.export_run --run amboy_r8
```

- **학습 중 점수:** 검증 점수와 SAROPS 대비 비율은 1.5 × POS 기준입니다(A 제외). R7·R6의 비율과 숫자를 직접 비교하지 마세요. 같은 기준에서 비교하려면 R7 모델도 위 시험 명령에 `--a-weight 0`을 붙여 채점합니다. R7 관측 옵션 `--rel-maps --fp-rate`도 붙입니다.
- **꼭 볼 것:** 사막형의 POS(SAROPS 대비)와 비행 모양입니다. 비행 모양은 `flight_shape.csv`의 "A 인정 한계보다 빨리 돈 분"과 "360° 회전 수"를 봅니다. 참고용 A 값은 계속 기록되니 R7과 비교할 수 있습니다.
- **사막형이 여전히 원형으로 돈다면,** 남은 원인은 POS의 재통과 보상입니다. 입자가 화면 밖에 5초 넘게 있다가 다시 들어오면, 남은 확률의 0.7배를 또 줍니다. 사용자가 정한 확인 실험은 두 가지입니다.
  1. RL 경로에서 시작해 한 구간의 행동을 POS가 커지게 최적화합니다. 경로가 크게 바뀌는지 봅니다. 바뀌면 학습 부족이고, 거의 안 바뀌거나 여전히 돌면 보상 구조 문제입니다. 학습에 쓰지 않은 입자로도 확인합니다.
  2. 정지 입자가 적은 경우를 봅니다. 지금은 분당 정지 0.05, 출발 0.16이라 약 24%가 멈춰 있습니다. 0.01로 낮추면 약 6%입니다. 같은 모델이 덜 도는지 봅니다.
  - 두 실험 모두 아직 전용 스크립트가 없습니다. 만들기 전에 사용자와 상의하세요.
- **엔트로피와 학습률은 기본값으로 둡니다.** 학습 중 흔들림이 커지면 직진을 유지하기 어려워서, 이번 변경의 효과와 섞입니다.
- **맥에서 확인한 범위:**
  - R8 환경은 맥에서 시험했습니다. 관측 형태, 보상 = 1.5 × POS 증가 × 100, 지도마다 최댓값 1을 확인했습니다.
  - SAROPS를 POS만으로 계획하는 기준값 계산은 맥에서 계산을 멈춘 상태라 실제로 돌려 보지 못했습니다. 웹 화면에서 쓰던 POS 전용 계획 모드를 이용합니다.
  - 그래서 첫 사례의 출력에서 오류가 없는지, 그리고 이름에 `_a0`이 붙는지 먼저 확인하세요.

- local PC는 원래 작업 환경(맥북과 그동안의 대화)과 **연결되어 있지 않습니다.** 결과를 밖으로 옮기기도 어렵습니다.
- 그래서 이 문서 하나만 읽고도 다음을 알 수 있게 정리했습니다.
  - 무엇을 왜 하는지
  - 무엇을 지켜야 하는지
  - 무엇을 기록해 남길지
- 명령의 자세한 설명은 `docs/run_on_pc.md`에 있고, 이 문서 4장에는 핵심만 다시 적었습니다.

## 1. 한눈에

- **할 일:**
  1. R7을 처음부터 300만 스텝 학습합니다.
  2. 시험 30개로 SAROPS·R6와 비교합니다.
  3. 결과를 분석해 정리하고 zip 하나로 내보냅니다.
- **R7 학습은 local PC에서만 합니다**(사용자 결정). 맥북에서 잠깐 시작했던 R7은 14만 스텝에서 멈췄습니다(`amboy_r7_mac_stopped`, 쓰지 않음).
- **실행 이름은 `amboy_r7`입니다.** 결과는 `rl_search/data/rl_real/amboy_r7/`에 생깁니다.

## 2. 배경 (처음 보는 사람을 위해)

- **문제:** 마지막 확인 위치(LKP)에서 놓친 차량을 고정익 무인기로 60분 비행 안에 찾는 경로 계획입니다.
  - 무인기: 33 m/s, 고도 500 m, 2축 짐벌로 진행 방향 45° 아래를 보는 카메라.
  - 지도: 미국 암보이 일대 60 km × 60 km의 실제 지형과 도로입니다.
- **표적 후보:** 입자 5000개가 도로망과 지형을 따라 움직이는 확률 분포입니다. 유형은 셋입니다.
  - 도로형: 도로만 다님
  - 혼합형: 가끔 도로를 벗어남
  - 사막형: 자주 벗어나 도로 밖을 시속 10~25 km로 이동
- **비교 대상:**
  - SAROPS: 규칙 기반 탐색 계획기(기준선)
  - RL: PPO. 5초마다 진행 방향 변화(최대 49°)를 고릅니다.
- **점수:** 학습에 쓰지 않은 입자(hold-out)로 계산한 **A + 1.5 × POS**입니다.
  - **POS:** 표적을 찾을 확률(탐지 확률 누적)입니다. 두 가지가 반영됩니다.
    - 화면 위치 계수: 화면 가운데 1, 가장자리 0.5, 모서리 0.25
    - 선회율 계수(D4): 롤 5°에서 0.91, 10°에서 0.49, 20°에서 0.14
  - **A:** 확률이 높은 구역(입자 70%를 담는 25 m 칸들, 10초마다 갱신)을 빠짐없이 훑은 비율입니다. 칸이 다음 두 조건을 모두 만족할 때만 훑었다고 셉니다.
    - 카메라 화면 가운데(위치 계수 0.75 이상)에 들어옴
    - 그때 선회율이 1.49°/s(롤 5°) 이하
- **보상:** 5초마다 (A 증가 + 1.5 × POS 증가) × 100입니다.
- **관측:**
  - 무인기를 가운데 두고 진행 방향이 위인 지도 4장(1.6 / 6.4 / 25.6 / 51.2 km, 각 64 × 64칸)이 있습니다.
  - 채널은 남은 POS, 예측 POS, A 값, 가시도, 카메라 화면(1.6 km 지도만)입니다.
  - 여기에 벡터 8개(롤, 선회 상태, 남은 에너지 등)가 더해집니다.
- **지금까지의 기준:** R6(R5에서 이어 학습, 300만 스텝)는 검증 15개에서 SAROPS 대비 **1.31**입니다(도로형 1.35, 혼합형 1.28, 사막형 1.26).

## 3. R7: 무엇을 왜 바꿨나

| 바꾼 것 | 왜 | 근거 |
|---|---|---|
| 처음부터 학습 | 이전 모델에서 규칙을 바꿔 이으면 과도 구간이 생겨 비교가 흐려짐 | R6는 R5에서 이어 시작해 20만~30만 스텝에 떨어졌고, 100만 무렵 원 선회가 늘었다가 200만에야 회복 |
| A 구역: 도로 밖 입자 σ 150, 도로 위는 σ 75 그대로 (A6) | σ 75만 쓰면 사막형 구역이 잘게 쪼개져 작은 조각을 줍는 선회를 부를 수 있음 | 사막형 구역이 136조각(작은 조각 20%)에서 30조각(4%)으로 줄고, 도로형은 그대로 |
| 상대 POS 채널: 지도마다 마지막 채널, 남은 POS를 남은 총량 대비로 | 후반에 남은 확률이 줄면 지도가 흐려지고 학습 신호도 약해짐 | 마지막 20분 결정이 정책을 3~10배 약하게 움직임(R6 신호 점검) |
| 카메라 채널 × 지금 롤의 선회율 계수 | 돌 때 탐지 품질이 떨어지는 것을 지도에서 직접 보게 함 | 전에는 벡터의 롤만으로 배워야 했음 |
| 환경 10개, n_steps 307 | 업데이트 한 번에 시나리오 약 11개에서 약 15개로 늘림 | 업데이트마다 허용 한도 가까이 움직여 정책이 ±0.02 흔들림 |

- **그대로 둔 것:**
  - 화면 위치 조건 0.75: R6 200만에서 원 선회가 사라져, 낮출 이유가 없어짐
  - D4, A의 롤 5° 기준, POS 가중치 1.5
  - 거리 인계 5 km: 처음에는 규칙이 남은 확률 쪽으로 날고, 10%가 5 km 안에 들어오면 RL이 맡음
  - 지도 밖 비행 허용(지도 밖은 0으로 표시)
  - 유형별 학습 비율 자동 조정: 하한 0.2, 상한 0.5. 가장 뒤처진 유형을 0.05씩 올림
  - 검증 `mval` 15개
- **뺀 것:**
  - 유형별 어드밴티지 정규화: 다른 환경을 비슷하게 만들어 위험함
  - 속도 조절: 보류
  - 큰 지도: 다음 단계

## 4. 순서와 확인 기준

명령은 저장소 최상위에서 **bash**로 실행합니다(zsh에서는 `$DEF` 대신 `${=DEF}`).

```bash
DEF="--area-def realtime_t --area-roll-max 5 --blur-tau 3 --blur-dead-roll 5 --blur-tau-low 15 --area-sigma 75 --area-sigma-off 150 --area-img-min 0.75 --pos-w 1.5"
```

1. **[ ] 설치:** `docs/run_on_pc.md` 1장을 따릅니다. 맥에서 쓴 버전은 Python 3.14, torch 2.15, stable-baselines3 2.9.2, gymnasium 1.4입니다.
2. **[ ] 지도:** `python -m rl_search.scripts.make_real_terrain --region amboy --cell 25`
   - 확인: `python -c "from rl_search.rl_real import preload; preload('amboy'); print('ok')"`
3. **[ ] 학습 풀:** `python -m rl_search.scripts.rl_eval.make_pool --multi --n 600 --area-def realtime --area-sigma 75 --area-sigma-off 150 --workers 8`
   - 확인: `rl_search/data/real/amboy/rl_pool_multi_o150/`에 600개(도로·혼합·사막 200개씩), 약 61 GB.
   - 파일 크기는 도로형 약 34 MB, 혼합형 55~75 MB, 사막형 115~195 MB 정도입니다.
4. **[ ] SAROPS 기준값:**
   - 검증 15개: `python -m rl_search.scripts.rl_eval.sarops_ref --split mval --stages 3 $DEF --workers 8`
     - 저장소에 `rl_search/data/real/amboy/rl_eval/sarops_ref_mval_realtime_t5_d3z5l15_s75_o150_i0.75_w1.5.json`(맥에서 만든 것)이 있으면 건너뜁니다.
     - 새로 만들었다면 맥 파일과 점수가 같은지 확인하고 기록합니다.
   - 시험 30개: 같은 명령에 `--split mtest`. 마지막 비교에만 쓰므로 학습 중이나 뒤에 만들어도 됩니다.
5. **[ ] R7 학습:**
   ```bash
   python -m rl_search.scripts.rl_eval.train_rl --tag amboy_r7 --stage 3 --n-envs 10 --envs-per-proc 1 --n-steps 307 \
     --eval-workers 2 --eval-every 100000 --save-every 500000 --stop-at 3000000 --device cuda --mask-cost 0.1 --multi \
     --eval-split mval --handover-km 5 --no-edge $DEF --rel-maps --fp-rate
   ```
   - **바꾸지 말 것:** `--n-envs 10 --n-steps 307`과 그 밖의 학습 설정. 결과에 영향을 줍니다.
   - **기계에 맞게 바꿔도 되는 것:** `--envs-per-proc`(코어가 모자라면 2), `--eval-workers`, `--device`. 속도에만 영향을 줍니다.
   - **진행 기준:** R5가 처음부터 학습했을 때 검증 SAROPS 대비 비율입니다(규칙이 조금 달라 참고용).

     | 스텝 | 10만 | 20만~50만 | 60만~100만 | 110만~140만 |
     |---|---|---|---|---|
     | R5 비율 | 0.91 | 0.77~0.90 | 0.97~1.05 | 1.08~1.16 |

     R7이 이보다 뚜렷하게 처지면 기록합니다. 기준값이 없는 동안 첫 평가 몇 번은 비율 없이 점수만 나오고, 유형 비율도 그동안 1/3씩입니다. 정상입니다.
   - **멈춤:** 300만 스텝에서 끝납니다. 그 전에 50만 스텝 넘게 "최근 3번 평균"이 오르지 않으면, 멈출지 판단 근거를 기록합니다. 멈출지는 사용자가 정합니다.
   - **이어서:** 끊기면 같은 명령에 `--resume latest`를 붙입니다.
6. **[ ] 시험 30개 비교** (`docs/run_on_pc.md` 6장):
   - R7(`best_stage3.zip`)
   - R6(`rl_search/data/rl_real/amboy_r6/best_stage3.zip`, 저장소에 있음). R7과 **같은 채점 규칙**으로, 관측 옵션 없이 채점하고 결과 json을 `amboy_r7/`로 복사합니다.
7. **[ ] 분석:** 5장
8. **[ ] 내보내기와 보고서:** 6장

## 5. 분석할 것 (사용자가 궁금해하는 것)

1. **검증 비율 추세:** 전체와 유형별. 어디서 평탄해지는지, R6(1.31)와 비교합니다.
2. **시험 30개:** R7 대 SAROPS, R6 대 SAROPS(같은 규칙), R7 − R6을 사례마다 짝지은 차 ± 표준오차와 이긴 사례 수로 정리합니다.
3. **비행 모양** (`flight_shape.csv`): A 인정 한계보다 빨리 돈 분, 한 방향 360° 회전 수, 급선회(>5°/s) 분. R6 기준은 아래와 같습니다.

   | R6 스텝 | 50만 | 100만 | 150만 | 200만 | 250만 | 300만 |
   |---|---|---|---|---|---|---|
   | 한계보다 빨리 돈 분 | 19.0 | 23.9 | 20.2 | 12.3 | 20.8 | 15.5 |
   | 360° 회전 수 | 1.1 | 3.0 | 0.5 | 0.0 | 0.1 | 0.0 |

4. **사막형 선 고르기:**
   - R6는 검증 사막형 12번(`--k 11`)의 이륙 뒤 12~19분(웹 화면 시계 25~32분)에 도로에서 250~750 m 떨어진 평행선을 날았습니다.
   - 그래서 앞쪽 남은 확률의 1.5~10%만 봤고, 가장 좋은 줄은 10~30%였습니다.
   - R7에서 같은 사례와 구간을 `replay_flight --config ...`와 `moment_report --minutes 10-20`으로 확인합니다.
5. **끝부분 행동:** eval 출력 끝줄의 세 가지입니다. 마지막 15분의 POS 증가, 남은 확률이 10 km 안에 5% 미만인 분, 지도 가장자리 1 km 안의 분.
6. **학습 신호 점검:** `python -m rl_search.scripts.rl_eval.signal_check --run amboy_r7 --per-type 3`
   - 상대 POS 채널이 후반 신호를 키웠는지 봅니다.
   - R6 200만의 "정책을 움직이는 힘"(|어드밴티지| ÷ 전체 표준편차, 이륙 뒤 10분 구간)은 다음과 같습니다.

     | 유형 | 0–10 | 10–20 | 20–30 | 30–40 | 40–50 | 50–60분 |
     |---|---|---|---|---|---|---|
     | 도로형 | 2.51 | 1.60 | 0.99 | 0.64 | 0.56 | 0.21 |
     | 혼합형 | 1.44 | 1.03 | 0.66 | 0.34 | 0.23 | 0.18 |
     | 사막형 | 0.80 | 0.24 | 0.27 | 0.21 | 0.11 | 0.24 |

7. **해석할 때 주의:**
   - 학습 곡선(TensorBoard)은 최근 100개 비행 평균입니다. 시나리오마다 점수 차이가 커서 ±0.06 정도는 우연입니다.
   - 검증 비율도 사례에 따른 우연이 ±0.017 정도 있습니다(R6 분석).

## 6. 남길 것

결과는 아래 명령으로 zip 하나로 묶습니다.

```bash
python -m rl_search.scripts.rl_eval.export_run --run amboy_r7                 # 가중치 포함 약 30 MB
python -m rl_search.scripts.rl_eval.export_run --run amboy_r7 --models none   # 가중치 없이 약 1 MB
```

- zip에는 `summary.md`, CSV, 그림, 가중치가 들어갑니다. 하나만 꺼낼 수 있다면 `summary.md`와 `best_stage3.zip`입니다.
- zip을 꺼낼 수 없으면, 아래 양식을 손으로라도 채워 남깁니다. 숫자는 소수 셋째 자리까지 적습니다.

```markdown
# R7 local PC 보고 (작성일)
## 환경
- 기계: CPU / 코어 수 / GPU / 메모리, Python·torch·SB3 버전
- 바꾼 실행 옵션(--envs-per-proc 등)과 이유
## 학습
- 총 스텝, 걸린 시간, 초당 스텝
- 검증 비율: 50만 / 100만 / 150만 / 200만 / 250만 / 300만 (전체, 도로형, 혼합형, 사막형)
- 최고 비율(스텝), 마지막 최근 3번 평균, best_stage3.zip의 스텝
## 시험 30개 (같은 규칙)
- R7: RL 평균, SAROPS 평균, 짝지은 차 ± 표준오차, 이긴 사례 수
- R6: 같은 항목
- R7 − R6: 짝지은 차 ± 표준오차, R7이 높은 사례 수
- 끝부분 지표(R7, R6): 마지막 15분 POS, 먼 시간, 가장자리 시간
## 비행 모양 (50만 스텝마다)
- A 인정 한계보다 빨리 돈 분, 360° 회전 수, 급선회 분
## 학습 신호 점검 (signal_check 표)
## 눈에 띈 비행
- 사례(유형·번호), 시각(이륙 뒤 분), 무엇이 이상했는지, 그때 남은 확률이 어디 있었는지
## 문제와 바꾼 것
- 오류, 재시작, 설정 변경(있다면 이유)
```

## 7. 작업 방식 (사용자와 정한 규칙)

- 사용자에게는 **한국어**로 답합니다. A6, D4, R7 같은 이름표에는 무슨 뜻인지 짧은 설명을 붙입니다.
- 결과는 **표와 그림**으로 보입니다.
- **실험 설계를 사용자와 상의 없이 바꾸지 않습니다.** 보상, A 정의, 관측, 학습 설정이 여기에 해당합니다. 바꾸면 비교가 깨집니다. 여러 변경을 한 실험에 묶을지도 사용자가 정합니다.
- **비교는 같은 규칙에서 합니다.** 같은 A 정의와 같은 이름의 SAROPS 기준값을 씁니다.
- **학습 부족을 분석할 때는 결론부터 내지 않습니다.** 그림(뷰)을 준비해 사용자와 같이 봅니다.
- **질문에는 답만 합니다.** 사용자가 묻는 말에 계산을 임의로 시작하지 않고, 불분명하면 먼저 묻습니다.
- **커밋과 푸시는 사용자가 요청할 때만 합니다.**
- **저장 공간을 아낍니다.** 체크포인트와 검증 비행은 50만 스텝마다만 남깁니다(`--save-every 500000`).
- **범위:** 탐색, 그리고 발견 뒤 감시·추적까지입니다. 무기 교전은 다루지 않습니다.

## 8. 문제 해결

- **zsh에서 옵션이 한 덩어리로 넘어가 바로 실패하는 경우:** bash로 실행하거나 `${=DEF}`를 씁니다.
- **"no pool at ..." 오류:** 3단계 학습 풀을 먼저 만듭니다.
- **메모리:** 환경 하나에 약 0.8 GB, 평가 작업자 하나에 약 2.6 GB, 학습 풀 작업자 하나에 약 1.5 GB, 본체 약 2 GB입니다.
- **코어:** 바쁜 프로세스가 성능 코어 수를 넘으면 모든 환경이 가장 느린 환경을 기다립니다. 이때는 `--envs-per-proc 2`를 씁니다.
- **GPU가 없을 때:** `--device cuda`는 자동으로 CPU로 바뀝니다. 느리지만 결과는 같은 방식입니다.
- **그림의 한글이 깨질 때:** `fonts-nanum`을 설치합니다. 표와 CSV는 영향이 없습니다.
- **대기 스크립트에서 `pgrep -f`를 쓸 때:** 자기 자신과 엇갈려 찾지 않도록 `amboy_r[7]`처럼 괄호 패턴을 씁니다.

## 9. 참고 문서

- `docs/run_on_pc.md`: 명령과 옵션 설명
- `docs/rl_experiments.md`: 모든 실험(R2~R7)의 변경·이유·결과와 결정. 아래쪽 "관찰과 결정"이 최신
- `docs/rl_mdp_design.md`: RL 설계(MDP, 신경망, 학습 설정)
- `CLAUDE.md`: 저장소 전체 안내
