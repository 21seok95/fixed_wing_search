# fixed_wing_search — local PC용 브랜치 (`local-pc`)

고정익 무인기의 차량 탐색 강화학습을 local PC에서 학습·평가·분석하고 웹에서 보는 데 필요한 **코드만** 담은 브랜치입니다.
`feat/rl-env`(맥의 작업 브랜치)에서 골라 만들며, 역사가 이어져 있지 않으므로 합치지 않습니다. 2026-11-18부터 `local-pc-r13` 대신 씁니다
(`local-pc-r13`은 그대로 남아 있음).

```bash
git clone --branch local-pc --single-branch https://github.com/21seok95/fixed_wing_search.git fixed_wing_search_pc
cd fixed_wing_search_pc
```

이미 받아 둔 저장소가 있으면: `git fetch origin local-pc && git checkout -b local-pc origin/local-pc`.
설치는 `docs/run_on_pc.md` 1장을 따릅니다(clone·checkout 줄은 뺌). 할 일은 `local_pc.md` 0장에 있습니다.

## 용량을 작게 두는 규칙 (사용자 2026-11-18)

| 들어 있지 않은 것 | 이유 · 대신 |
|---|---|
| 모델 가중치, 학습 로그, TensorBoard, 웹 실행 결과, 그림·보고 | 결과는 git으로 오가지 않음. local PC 결과는 `export_run` zip으로 가져옴 |
| 지도 고도(DEM)·땅 덮개 원자료 | 고정 배포판(Copernicus DEM 30 m, ESA WorldCover 2021)이라 `make_real_terrain`이 다시 받으면 같은 값 |

OSM 도로·물길 원자료는 시간이 지나면 바뀌므로 맥이 받은 것을 압축해서(`raw/osm_*.json.gz`) 넣었습니다.

## 지도 만들기와 확인 (기계마다 한 번, 인터넷 필요)

```bash
python -m rl_search.scripts.make_real_terrain --region amboy --cell 25
python -m rl_search.scripts.make_real_terrain --region mojave --cell 25
python -m rl_search.scripts.make_real_terrain --region chungju --cell 25       # 웹 충주 검증
python -m rl_search.scripts.make_real_terrain --region chungju120 --cell 25    # 우리나라 120 km
python -m rl_search.scripts.terrain_fingerprint check --region amboy mojave chungju chungju120
```

마지막 줄이 지도마다 `same as the Mac`이어야 합니다. 다르면 어느 배열이 얼마나 다른지 나오니, 학습 전에 맥 쪽에 알려 주세요.
늘린 지도(`amboy_x2`, `mojave_x2`)는 암보이·바스토 60 km 지도에서 바로 만들어지므로 따로 만들 것이 없습니다.

## 웹 보기

```bash
python -m rl_search.scripts.real_map.vehicle_web          # http://127.0.0.1:8000
```

맥에서 더한 웹 기능(지형·표적 유형별 시나리오 버튼, 설정 묶음과 설명, 학습처럼 뽑기, 날짜별 실행 목록, RL 비행)이 모두 들어 있습니다.
RL 비행은 이 기계에서 학습한 체크포인트(`rl_search/data/rl_real/<실행>/ckpt/`)로 합니다. 학습 중에는 켜지 마세요(메모리).

## 결과 보내기

```bash
python -m rl_search.scripts.rl_eval.export_run --run <실행 이름>      # rl_search/data/export/<실행>_<날짜-시각>.zip
```
