# fixed_wing_search — local PC용 R13 브랜치 (`local-pc-r13`)

고정익 무인기의 차량 탐색 강화학습 R13을 local PC에서 학습·평가·분석하고 웹에서 보는 데 필요한 파일만 담은 브랜치입니다.
2026-11-16에 `feat/rl-env`에서 골라 커밋 하나로 만들었습니다. `feat/rl-env`와 역사가 이어져 있지 않으므로 합치지 않습니다.

**할 일과 순서는 `local_pc.md` 0장에 있습니다.** 설치는 `docs/run_on_pc.md` 1장을 따르되 `git clone`·`git checkout` 줄은 뺍니다.

```bash
git clone --branch local-pc-r13 --single-branch https://github.com/21seok95/fixed_wing_search.git fixed_wing_search_r13
cd fixed_wing_search_r13
```

## 들어 있는 것

| 무엇 | 어디 |
|---|---|
| 시뮬레이터·환경·SAROPS·평가 코드 | `rl_search/*.py` |
| 지도 만들기 | `rl_search/scripts/make_real_terrain.py` |
| 점검·학습 풀·학습·평가 | `rl_search/scripts/rl_eval/check_r10.py`, `make_pool.py`, `train_rl.py`, `eval_rl.py` |
| 분석·내보내기 | `rl_search/scripts/rl_eval/r12_compare.py`, `circle_check.py`, `compare_flights.py`, `curve_runs.py`, `export_run.py` |
| 웹 보기 | `rl_search/scripts/real_map/vehicle_web.py` + `vehicle_web.html` (검증 비행 다시 보기, 체크포인트로 새 비행) |
| 두 지도 원자료 (암보이·바스토) | `rl_search/data/real/{amboy,mojave}/raw/` (내려받지 않고 지도를 만듦) |
| 차량 모델 설정, SAROPS 기준값 (nval·bval) | `rl_search/data/real/amboy/rl_eval/`, `rl_search/data/real/mojave/rl_eval/` |
| 비교용 R10·R12 기록 | `rl_search/data/rl_real/r10_2map/`, `r12_2map/` (설정, 검증 기록, TensorBoard), `r10_2map_train.log`, `r12_train.log` |
| 비교용 R12 600만 모델 | `rl_search/data/rl_real/r12_2map/ckpt/model_6000000.zip` |
| R13을 정한 근거 그림 | `rl_search/data/real/amboy/rl_eval/report/r12_circle_*.png`, `r12_outside_map.png` |
| 문서 | `local_pc.md`, `docs/run_on_pc.md`, `docs/rl_experiments.md`, `docs/rl_mdp_design.md`, `docs/r10_ubuntu.md` |

`feat/rl-env`에만 있는 것: 충주·합성 지형 작업, 다중 무인기 실험, R1~R9·R11 실행, R10 모델, 대부분의 보고 그림, 그 밖의 도구.
웹 보기는 지도를 만든 암보이·바스토만 엽니다.

## 결과 가져오기

```bash
python -m rl_search.scripts.rl_eval.export_run --run r13_2map     # rl_search/data/export/r13_2map_<날짜-시각>.zip
```
