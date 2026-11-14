# R10: 우분투 PC에서 할 일 (2026-11-14, 맥에서 준비)

이 문서 하나로 R10을 시작할 수 있게 정리했다. 배경과 결정은 사용자와 하나씩 정한 것이고, 코드는 `feat/rl-env`에 들어 있다.
실험 기록은 `docs/rl_experiments.md`(R10 줄, 11-14 관찰)에 함께 적는다.

## 0. 작업 규칙 (사용자와 정한 것)

- 사용자에게는 한국어로 답한다. 결과는 늘 **그림과 표**로 보여 준다(시간이 더 걸려도).
- 이름표(R10, A′, nval 같은 것)를 쓸 때는 쉬운 말 설명을 붙인다.
- 한 단계씩 하고 단계마다 사용자에게 확인한다. 실험 설계와 무엇을 한 실험에 묶을지는 사용자가 정한다.
- 토론 중에는 계산을 돌리지 않는다. 물어본 것에 바로 답하고, 확인이 필요하면 돌릴지 먼저 묻는다.
- 학습이 부족한 곳을 찾을 때는 그림을 준비해 사용자와 같이 본다. 먼저 결론을 내고 오지 않는다.
- 체크포인트와 웹 실행은 50만 스텝마다만 남긴다(`--save-every 500000`).
- 실행을 시작하거나 끝낼 때마다 `docs/rl_experiments.md`를 고친다.
- 커밋·푸시는 사용자가 말할 때만 한다. `rl_search/data/`와 `*.png`는 git이 무시하므로 결과는 `git add -f`로 넣는다.
- 연결되지 않은 다른 PC는 "local PC"라고만 부른다.

## 1. 왜 R10인가

| 실행 | 바꾼 것 | 검증 결과 (유형별 검증 15개, SAROPS 대비 hold-out) |
|---|---|---|
| R8 | A 끔(보상 1.5 × POS만) + POS 채널을 지도마다 0~1로 정규화 | 300만 **1.328** (도로 1.36 / 혼합 1.32 / 사막 1.26), 원 선회 없음 |
| R9 | R8 + A′(입자를 화면 가운데로 완만하게 볼 때마다 작은 보상 0.01) + 사막형 50% + 엔트로피 0.01 | 400만 1.245 (도로 1.22 / 혼합 1.30 / 사막 1.23). 100만·300만에 사막형 원 선회, 240만·280만에 도로형 7번이 0점으로 무너짐 |

- hold-out: 계획에 쓰지 않은 별도 입자(seed + 1,000,000)로 같은 경로를 다시 날아 매긴 점수.
- 사막형은 R5부터 SAROPS 대비 1.2대에서 멈춰 있고, 원 선회가 자주 섞인다. 사용자 판단(11-14): 분포가 좁아 원 선회를 할 수밖에 없는 면이 있다.
  확인(R9 400만, 사막형 5개를 보고 후 40분에 이륙시켜 분포를 2~17배 넓힘): 넓어지면 모든 사례에서 4~7%p 덜 돌지만 여전히 탐색 시간의 32~50%를
  빨리 돈다(`report/r9_desert_late40.png`). 표적이 시간의 약 24%를 멈춰 있어서 분포가 한곳에 뭉치는 것도 원인 후보.
- R8·R9의 지도에는 정규화된 POS만 있어서 칸마다 확률이 실제로 얼마나 있는지를 지도에서 직접 볼 수 없었다(벡터의 지도별 나눗수로만 앎).

## 2. R10에서 바뀌는 것 (사용자 결정, 2026-11-14)

| 항목 | R9 | R10 |
|---|---|---|
| 표적 정지 | 분당 5% 멈춤, 분당 16% 다시 출발 | **없음** |
| 표적 속도 | 길 종류(빠른 길·느린 길·도로 밖)마다 모든 입자가 같은 속도 | 입자마다 운전 성향 z를 한 번 뽑아 모든 길에 같이 씀: **평균 × (1 + 0.15 z)**, 평균의 **0.7~1.3배**로 자름(평균 30이면 21~39 km/h). 끝값에 붙는 입자 약 5% |
| 지도 채널 | 지금 POS(정규화), 앞으로 POS(정규화), 가시도, 카메라 × 선회(1.6 km만) | 아래 표 6개 |
| A | 보상에는 없고 참고로만 계산, 학습 풀에 A 영역 저장(61 GB) | 환경과 풀에서 **완전히 뺌**(풀 약 16 GB, 한 걸음 약 4 ms) |
| 검증 | `mval`(정지 있음) | **`nval`**: mval과 같은 사례 번호·유형·LKP·이륙 시각·출발점, 표적만 정지 없음 |
| 학습 설정 | – | **R9 그대로**(사용자 선택): A′ 0.01, 사막형 50% 고정, 엔트로피 0.01, 처음부터 400만 스텝 |

**R10 지도 채널** (지도 4장 1.6 / 6.4 / 25.6 / 51.2 km, 64×64칸, 진행 방향이 위, 무인기가 가운데; 입자 수는 5000개 기준으로 환산)

| # | 채널 | 값 |
|---|---|---|
| ① | 현재 입자 위치 | log(1 + 칸의 입자 수) ÷ log(1 + 5000). 무게 없음, 정규화 없음 |
| ② | lead 입자 위치 | 같은 식, lead초 뒤 위치(1.6 km 15초, 6.4 km 1분, 25.6 km 5분, 51.2 km 10분) |
| ③ | 현재 POS (정규화) | log(1 + 칸의 남은 확률 × 가시도) ÷ 그 지도에서 가장 큰 칸의 값 → 0~1 |
| ④ | 현재 POS (실제 값) | log(1 + 칸의 남은 확률 × 가시도) ÷ log(1 + 5000). ①과 같은 단위 |
| ⑤ | 가시도 | 지형·식생으로 보이는 정도, 지도 밖 0 |
| ⑥ | 카메라 화면 × 선회 계수 | 1.6 km 지도만 |

벡터 12개는 R9와 같다(롤, 선회율 상태, 목표 진로 − 기수, 남은 에너지, A 몫 0, 남은 확률 합, 허용 행동 −1/+1, 지도마다 ③의 나눗수 log값 ÷ 10).

결정 이유:
- **①②와 ③④를 나눔:** "어디에 있을 수 있나"와 "얼마나 남았나"를 따로 본다. 아직 안 훑었고 잘 보이는 칸은 ① = ④, 훑은 칸은 ④ < ①이라
  두 채널을 비교하면 훑은 곳이 바로 드러난다. A′도 입자 개수로 세므로 ①이 A′를 받을 곳을 보여 준다.
- **lead POS(남은 확률을 lead 위치에 놓은 것)를 뺌:** 사용자: 냉정하게 보면 꼭 필요하지 않다. 짧은 lead(1.6·6.4 km)는 지금 POS와 거의 같고,
  긴 lead도 ①→②의 움직임만큼 ③④를 옮기면 대부분 추정된다. 교차로에서 갈라지거나 훑은 입자와 안 훑은 입자가 한 칸으로 모일 때만 틀어진다.
- **①②는 정규화하지 않음:** 입자 수는 줄지 않아(지도 밖으로 나간 것만 빠짐) 정규화로 풀 문제가 없고, log라 퍼진 분포도 보인다.
  정규화하면 사막형의 성긴 지도와 도로형의 짙은 지도가 같은 밝기가 되어 "얼마나 몰려 있나"를 잃는다.
- **③④를 하나로 합치지 않음:** 칸마다 숫자 하나로는 상대 모양과 절대 크기를 함께 담을 수 없다. 채널 하나 더는 비용이 거의 없다.

## 3. 코드 (맥에서 넣고 시험함)

| 파일 | 내용 |
|---|---|
| `rl_search/vehicle.py` | `VehicleConfig.speed_band`: 0보다 크면 상태마다 속도를 평균 × (1 ± band)로 자름(기본 0이라 다른 실험은 그대로) |
| `rl_search/rl_eval_set.py` | `NOSTOP`(표준편차 15%, 띠 ±30%), `nostop_vc`, 검증 `nval`·시험 `ntest`(mval·mtest와 같은 seed), `multi_training_scenario(nostop=)` |
| `rl_search/rl_real.py` | 설정 `no_stop`(풀 `rl_pool_multi_ns`), `no_area`(A 없음), `obs_r10`(위 채널), `obs_layout` |
| `rl_search/area_score.py` | A 없이 학습할 때도 보상이 1.5 × POS가 되도록 함(A가 있을 때와 SAROPS 실행은 그대로) |
| `scripts/rl_eval/make_pool.py` | `--no-stop --no-area` |
| `scripts/rl_eval/train_rl.py`, `eval_rl.py`, `sarops_ref.py` | `--no-stop`, `--no-area`, `--obs-r10`, 분할 `nval`/`ntest` |
| `scripts/rl_eval/check_r10.py` | 시작 전 점검(아래 1번) |
| `scripts/rl_eval/curve_runs.py` | 학습 곡선 그림(전체·유형별 SAROPS 대비, 다른 모델 기준선) |
| `scripts/rl_eval/compare_flights.py` | 비행 두 묶음 비교 그림(유형별 POS, 선회, 사례별 POS, 사막형 경로) |
| 웹(`scripts/real_map/vehicle_web.py`, `web_rl.py`) | ② 탐색 계획 맨 아래에서 RL 모델로 화면 시나리오를 비행. 실행 목록 필터 "정지 없음 유형별 검증 15개 (R10)" |

맥 점검 결과: `check_r10` 모두 통과. 정지 0. 속도 퍼짐은 평균의 14~15%였고, 띠 안 비율은 97~100%였다. 띠 밖은 5초 사이에 길 종류가 바뀐 기록뿐이다.
풀 6개는 장당 26.45 MB였다. 학습 4096 스텝, nval 평가 2개, nval SAROPS 기준값 1개(사례 1: POS 0.747, 151초)를 돌렸다.
시험으로 만든 풀과 기준값은 지웠다.

## 4. 할 일 (순서대로, 단계마다 사용자에게 보고)

명령에 쓰는 설정 묶음(bash):

```bash
DEF="--area-def realtime_t --area-roll-max 5 --blur-tau 3 --blur-dead-roll 5 --blur-tau-low 15 --area-sigma 75 --area-sigma-off 150 --area-img-min 0.75 --pos-w 1.5"
REF="$DEF --a-weight 0"                                   # SAROPS 기준값: POS만으로 계획 (R8부터의 기준)
R10="--handover-km 5 --no-edge --a-weight 0 --fp-rate --look-w 0.01 --type-fixed desert=0.5 --ent-coef 0.01 --no-stop --no-area --obs-r10"
```

1. **코드와 점검** (몇 분)
   - `git fetch && git checkout feat/rl-env && git pull`, `source .venv/bin/activate`. 가상환경이 없으면 `docs/rl_mdp_design.md` 마지막 절대로 만든다.
   - 지형이 없으면 `python -m rl_search.scripts.make_real_terrain --region amboy --cell 25`(저장소의 원자료로 같은 지도).
   - `python -m rl_search.scripts.rl_eval.check_r10` → 마지막 줄 `ALL CHECKS PASSED`.
2. **학습 풀** (약 30~45분, 디스크 약 16 GB)
   - `python -m rl_search.scripts.rl_eval.make_pool --multi --no-stop --no-area --n 600 --workers 10`
   - 확인: `rl_search/data/real/amboy/rl_pool_multi_ns/`에 파일 600개(도로·혼합·사막 200개씩), 장당 약 26 MB.
3. **SAROPS 기준값, 정지 없는 검증 15개** (약 10분)
   - `python -m rl_search.scripts.rl_eval.sarops_ref --split nval --stages 3 --workers 5 $REF`
   - 확인: `rl_search/data/real/amboy/rl_eval/sarops_ref_nval_realtime_t5_d3z5l15_s75_o150_i0.75_a0_w1.5.json`에 15개.
     사례 1(k0)의 hold-out POS가 맥과 같은 0.747인지 본다.
   - 사용자에게: 유형별 SAROPS POS 표와 mval(정지 있음) 기준값과의 비교.
4. **비교 기준선: R9 400만(과 R8 300만)을 nval에서 채점** (각 약 5분)
   - 모델 파일이 저장소에 있을 때(`rl_search/data/rl_real/amboy_r9/ckpt/model_4000000.zip` 등):
     `python -m rl_search.scripts.rl_eval.eval_rl --model rl_search/data/rl_real/amboy_r9/ckpt/model_4000000.zip --label amboy_r9-4000000 --split nval --stages 3 $DEF --handover-km 5 --no-edge --a-weight 0 --pos-norm --fp-rate --workers 5`
     (R8은 `amboy_r8/ckpt/model_3000000.zip`, 라벨 `amboy_r8-3000000`, 같은 옵션)
   - R10과 같은 정지 없는 사례에서 옛 모델이 얼마인지가 R10을 판단할 기준이다.
5. **R10 학습** (400만 스텝, 예상 2.5~3.5시간)
   ```bash
   nohup python -m rl_search.scripts.rl_eval.train_rl --tag amboy_r10 --stage 3 --n-envs 10 --envs-per-proc 1 --n-steps 307 \
     --eval-workers 2 --eval-every 200000 --save-every 500000 --stop-at 4000000 --device cuda --mask-cost 0.1 --multi --eval-split nval \
     $DEF $R10 > rl_search/data/rl_real/amboy_r10_train.log 2>&1 &
   ```
   - 환경 10개, 업데이트 한 번 3,070 결정(R8·R9와 같음). `--envs-per-proc`는 속도만 바꾸고 학습은 같다(1 = 프로세스 10개, 2 = 5개).
     처음 5분 `fps`가 300 아래면 2로 다시 시작해 본다.
   - 로그의 `eval stage 3 at <스텝>: RL .. vs SAROPS .. (ratio ..)`와 `by type (...)` 줄이 평가 결과다. 끊기면 `--resume latest`로 잇는다.
   - 시작할 때와 100만·200만·300만·400만에 사용자에게 그림과 표로 보고한다(아래 5절).
6. **끝나면**: 400만 결과를 정리한다. 시험 30개(`ntest`)의 기준값과 평가를 할지는 사용자에게 묻는다.

## 5. 보고할 때 보여 줄 것

- 학습 곡선:
  `python -m rl_search.scripts.rl_eval.curve_runs --runs amboy_r10 --names R10 --split nval --ref "R9 400만=amboy_r9-4000000" --out rl_search/data/real/amboy/rl_eval/report/r10_curve.png`
- 같은 사례의 비행 비교(웹 실행이 남는 스텝: 60만, 100만, 160만, 200만, 260만, 300만, 360만, 400만):
  `python -m rl_search.scripts.rl_eval.compare_flights --split nval --a "R9 400만=amboy_r9-4000000" --b "R10 200만=amboy_r10-2000000" --out rl_search/data/real/amboy/rl_eval/report`
  (유형별 hold-out POS와 SAROPS, 1.49°/s보다 빨리 돈 분과 360° 회전 수, 사례별 POS, 사막형 경로)
- 표: SAROPS 대비 비율(전체·도로형·혼합형·사막형), 유형별 hold-out POS, 유형별 빠른 선회 분.
- 볼 것: ① 사막형 원 선회가 줄었는지, ② 사막형 SAROPS 대비가 1.2대를 넘는지, ③ 한 사례가 크게 무너지는 흔들림(R9 도로형 7번)이
  있는지(그 사례 중앙값의 35% 미만), ④ A′ 몫(텐서보드 `episode/look` → 0.01 × 주시 ÷ (1.5 × POS + 0.01 × 주시)).
- 한글 글꼴: 그림 스크립트는 AppleGothic → NanumGothic → Noto Sans CJK KR 순으로 찾는다. 글자가 네모로 나오면 `sudo apt install fonts-nanum`.
- 웹: `python -m rl_search.scripts.real_map.vehicle_web`, 실행 목록 필터 "정지 없음 유형별 검증 15개 (R10)". ② 탐색 계획 맨 아래에서
  R10 모델로 화면 시나리오를 바로 비행할 수 있다(암보이 지도만).

## 6. 열린 질문 (결과를 보고 사용자와 정함)

- lead POS를 뺀 영향: 도로형처럼 움직이는 표적을 쫓는 사례가 R9보다 나빠지면 이것부터 의심한다.
- 정지를 없애 분포가 넓어지면 원 선회가 줄어드는지.
- R9의 흔들림(엔트로피 0.01, 사막형 50% 때문인지)이 R10에서도 나오는지.
- 시험 30개(`ntest`) 기준값과 평가를 할지, 언제 할지.
