# 다른 PC에서 R7 학습하고 비교하기 (2026-11-12)

local PC에서 R7을 학습하고, SAROPS·R6와 비교하고, 결과를 파일 하나로 꺼내는 순서입니다.
명령은 저장소 최상위에서 **bash**로 실행합니다. zsh에서는 `$DEF`가 낱말로 나뉘지 않으므로 `${=DEF}`로 쓰세요.

R7 학습은 local PC에서만 합니다(사용자 결정, 맥북에서는 하지 않음). 실행 이름은 `amboy_r7`입니다. 작업 방식과 남길 것은 저장소 최상위 `local_pc.md`를 먼저 읽으세요.

## R7이 R6와 다른 점

| 항목 | 내용 |
|---|---|
| 시작 | 처음부터 (R6처럼 이전 모델에서 잇지 않음) |
| A 구역 (A6) | 도로 칸 위 입자는 σ 75, 도로 밖 입자는 σ 150. 기준값 이름에 `_o150` |
| 관측 ① | 지도마다 마지막 채널 하나 추가: 남은 POS를 남은 총량 대비로 (`--rel-maps`) |
| 관측 ② | 1.6 km 지도 카메라 채널에 지금 롤의 선회율 계수를 곱함 (`--fp-rate`) |
| 환경 | 환경 10개, n_steps 307 (업데이트당 약 3,070 결정, 시나리오 약 15개) |
| 그대로 | 화면 위치 조건 0.75, D4 흔들림, A4t 롤 5°, A + 1.5 POS, 거리 인계 5 km, 지도 밖 비행 허용, 유형별 학습(비율 자동 조정), 검증 `mval` 15 |

## 1. 준비 (한 번)

```bash
git clone https://github.com/21seok95/fixed_wing_search.git && cd fixed_wing_search   # 이미 있으면 git fetch
git checkout feat/rl-env && git pull
python3 -m venv .venv && source .venv/bin/activate
pip install -r rl_search/requirements.txt -r rl_search/requirements-rl.txt
pip install torch            # GPU가 있으면 pytorch.org의 CUDA 빌드
sudo apt install fonts-nanum # (선택) 그림의 한글. 없으면 그림 글자만 깨지고 표·CSV는 그대로
```

맥에서 쓴 버전은 Python 3.14.0, numpy 2.5.4, scipy 1.18.1, matplotlib 3.11.2, torch 2.15.0, stable-baselines3 2.9.2, gymnasium 1.4.0,
tensorboard 2.21.0입니다. Python 3.11 이상이면 될 것입니다.

## 2. 지도 (한 번)

```bash
python -m rl_search.scripts.make_real_terrain --region amboy --cell 25
python -c "from rl_search.rl_real import preload; preload('amboy'); print('ok')"
```

저장소의 원자료(`rl_search/data/real/amboy/raw`)로 맥과 같은 지도를 만듭니다. 인터넷은 필요 없습니다. 도로망(`road_graph_25m.npz`)과
지표 피복(`landcover_25m.npz`)도 저장소에 있습니다.

## 3. 학습 풀 (한 번, 디스크 약 61 GB)

```bash
python -m rl_search.scripts.rl_eval.make_pool --multi --n 600 --area-def realtime --area-sigma 75 --area-sigma-off 150 --workers 8
```

`rl_search/data/real/amboy/rl_pool_multi_o150/`에 시나리오 600개(도로·혼합·사막 200개씩)가 생깁니다. seed가 같아 맥과 같은 시나리오입니다.
파일 하나에 약 10초, 작업자 하나에 메모리 약 1.5 GB입니다. 중간에 멈춰도 다시 돌리면 이어서 만듭니다.

## 4. SAROPS 기준값 (한 번)

```bash
DEF="--area-def realtime_t --area-roll-max 5 --blur-tau 3 --blur-dead-roll 5 --blur-tau-low 15 --area-sigma 75 --area-sigma-off 150 --area-img-min 0.75 --pos-w 1.5"
python -m rl_search.scripts.rl_eval.sarops_ref --split mval --stages 3 $DEF --workers 8     # 검증 15개 (학습 중 평가에 필요)
python -m rl_search.scripts.rl_eval.sarops_ref --split mtest --stages 3 $DEF --workers 8    # 시험 30개 (마지막 비교에 필요)
```

결과는 `rl_search/data/real/amboy/rl_eval/sarops_ref_<split>_realtime_t5_d3z5l15_s75_o150_i0.75_w1.5.json`입니다. 사례 하나가 코어 하나로
3~5분입니다. 저장소에 이미 이 파일이 있으면(맥에서 만든 것) 건너뛰어도 됩니다. 이미 있는 사례는 다시 계산하지 않습니다.

## 5. R7 학습

```bash
python -m rl_search.scripts.rl_eval.train_rl --tag amboy_r7 --stage 3 --n-envs 10 --envs-per-proc 1 --n-steps 307 \
  --eval-workers 2 --eval-every 100000 --save-every 500000 --stop-at 3000000 --device cuda --mask-cost 0.1 --multi \
  --eval-split mval --handover-km 5 --no-edge $DEF --rel-maps --fp-rate
```

- **결과에 영향을 주는 설정:** `--n-envs 10 --n-steps 307` 등 학습 설정. 맥 R7과 비교하려면 그대로 둡니다.
- **속도에만 영향을 주는 설정:** `--envs-per-proc`, `--eval-workers`, `--device`.
  - `--envs-per-proc 1`은 환경마다 프로세스 하나입니다. 남는 코어가 12개 이상이면 이것이 빠릅니다.
  - 코어가 모자라면 `2`로 두면 프로세스 하나에 환경 2개가 들어가, 바쁜 코어가 절반이 됩니다(`rl_search/vec_group.py`).
- **메모리:** 환경 하나에 약 0.8 GB, 평가 작업자 하나에 약 2.6 GB, 본체 약 2 GB가 듭니다.
- **진행 확인:** `tensorboard --logdir rl_search/data/rl_real/amboy_r7`를 띄우거나, 로그의 `eval stage 3 at ... (ratio ...)` 줄을 봅니다.
  ratio가 같은 규칙의 SAROPS 대비 비율입니다. 유형별 비율은 `by type` 줄에 있습니다.
- **이어서 학습:** 같은 명령에 `--resume latest`를 붙입니다. 300만보다 더 하려면 `--stop-at`을 늘립니다.
- **더 큰 실험(선택, 별도 이름):** 예를 들어 `--tag amboy_r7b --n-envs 20 --n-steps 154`로 하면 묶음 크기는 비슷하고, 업데이트당 시나리오가 2배입니다.
- **저장되는 것:**
  - 50만 스텝마다 `rl_search/data/rl_real/<tag>/ckpt/model_<스텝>.zip`(약 20 MB, 가중치와 옵티마이저).
  - 검증 성적 최고(최근 3번 평균) 모델 `best_stage3.zip`.
  - 그 시점의 검증 비행 웹 실행(`rl_search/data/real/chungju/web_runs/eval-mval-s3-k*-rl-<tag>-<스텝>`).

## 6. 시험 30개로 비교

R7 (예: 가장 좋은 모델):

```bash
python -m rl_search.scripts.rl_eval.eval_rl --model rl_search/data/rl_real/amboy_r7/best_stage3.zip --label r7_best --split mtest \
  --stages 3 $DEF --handover-km 5 --no-edge --rel-maps --fp-rate --workers 8
```

R6를 R7과 같은 채점 규칙으로 비교합니다. 저장소의 R6 모델을 쓰고, R6 관측이라 `--rel-maps`와 `--fp-rate`는 붙이지 않습니다.

```bash
python -m rl_search.scripts.rl_eval.eval_rl --model rl_search/data/rl_real/amboy_r6/best_stage3.zip --label r6_on_r7rules --split mtest \
  --stages 3 $DEF --handover-km 5 --no-edge --workers 8
cp rl_search/data/rl_real/amboy_r6/eval_mtest_r6_on_r7rules.json rl_search/data/rl_real/amboy_r7/   # 내보내기에 함께 담기도록
```

둘 다 끝에 "RL 평균, SAROPS 평균, 짝지은 차 ± 표준오차, RL이 높은 사례 수"와 끝부분 지표를 출력합니다. 결과는 `eval_mtest_<label>.json`입니다.

선택 분석:
- **학습 신호 점검:** `python -m rl_search.scripts.rl_eval.signal_check --run amboy_r7 --per-type 3`. 유형·시간대별로 보상, 정책을 움직이는 힘, 가치 오차를 봅니다.
- **한 비행 확대:**
  - 먼저 `python -m rl_search.scripts.rl_eval.replay_flight --model <모델> --split mval --k <k> --config rl_search/data/rl_real/amboy_r7/config.json --out <npz>`
  - 그다음 `python -m rl_search.scripts.rl_eval.moment_report --run eval-mval-s3-k<k>-rl-amboy_r7-<스텝> --replay <npz> --minutes 10-20`

## 7. 결과 꺼내기

```bash
python -m rl_search.scripts.rl_eval.export_run --run amboy_r7                 # 가중치 포함 약 30 MB
python -m rl_search.scripts.rl_eval.export_run --run amboy_r7 --models none   # 가중치 없이 약 1 MB
```

`rl_search/data/export/amboy_r7_<날짜-시각>.zip` 하나가 생깁니다.

| 파일 | 내용 |
|---|---|
| `summary.md` | 검증 비율(전체·유형별, 체크포인트마다), 시험 비교(짝지은 차), 검증 비행의 선회 지표 |
| `eval_history.csv` | 검증마다 비율과 사례별 점수 |
| `tb_scalars.csv` | TensorBoard 값 전부 (학습 곡선, KL, 유형 비율 등) |
| `flight_shape.csv` | 저장된 검증 비행마다 A 인정 한계보다 빨리 돈 분, 360° 회전 수, 급선회 분, 가장자리 근처 분, 점수 |
| `eval_*.json` | 시험 평가 결과 (R7, R6 비교) |
| `curves.png` | 위 내용 그림 4장 |
| `config.json`, `state.json` | 학습 설정과 검증 기록 원본 |
| `best_stage3.zip`, `model_<스텝>.zip` | 모델 가중치 (`stable_baselines3.PPO.load`로 불러옴) |

하나만 꺼낼 수 있다면 `summary.md`와 `best_stage3.zip`을 가져오세요. 가중치만 있어도 맥에서 6단계를 다시 돌려 비교할 수 있습니다.

## 8. 주의

- 맥 R7과 PC R7은 무작위성(시드 흐름, GPU 계산)이 달라 숫자가 똑같지는 않습니다. 추세와 수준을 비교하세요.
- 같은 사례·같은 모델이면 평가는 결정적입니다. 행동을 고를 때 무작위가 없습니다.
- 디스크 사용량:
  - 학습 풀: 약 61 GB
  - 체크포인트: 20 MB × 7
  - 웹 실행: 50만 스텝마다 검증 비행 15개로 약 200 MB씩
