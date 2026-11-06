# 실제 지역 지도 TODO

SAROPS 방식 탐색([SAROPS 방식 적용 검토 TODO](sarops_approach_todo.md))을 실제 지형 위로 옮기는 작업을 관리한다.
날짜는 한국 시간.

## 2026-11-06 · 실제 지역 지도 1차 (충주·제천) (최신 인계)

**목표:** GTA처럼 도망치는 표적을 찾는 문제로 구체화한다.
- 지금은 학습과 경로 계획에 쓸 2D 레이어만 필요하다.
- 나중에 같은 데이터로 PX4/Gazebo 월드(DEM heightmap, 건물 높이 반영)까지 이어갈 수 있어야 한다.

**사용자와 합의한 결정**
- **PUBG 맵은 쓰지 않는다.** `pubg/api-assets`에는 위에서 본 렌더링 PNG만 있다(819 px, LFS 판 8192 px). 고도, 벡터, 3D 데이터가 없고, 이용약관도 PUBG API 약관을 따른다.
- **공개 데이터로 실제 지역을 쓴다.** 고도는 Copernicus DEM GLO-30, 토지 피복은 ESA WorldCover 2021, 도로와 하천은 OpenStreetMap이다. OSM은 ODbL이라 출처 표시가 필요하다.
- **지역은 B 충주·제천**, 중심 37.05°N 128.05°E, 60 km × 60 km, 50 m 격자다. 지금 합성 지형과 크기와 격자가 같다.
  - 후보 4곳의 비교는 `rl_search/data/real/candidates/regions_compare.png`, `region_*.png`에 있다.
  - B를 고른 이유: 산, 호수, 강, 도시, 도로가 고루 있고 고도와 경사 규모가 합성 지형과 비슷하다.
- **큰 도시만 지상 인계 구역(`city`)으로 뺀다.**
  - 기준: 500 m 창 안 시가지 비율 40% 이상, 구멍 메움, 이어진 면적 3 km² 이상.
  - 결과: 충주 도심, 제천, 충주 북쪽, 주덕 근처 4곳이며 지도의 1.1%다.
  - 도주자는 이 구역에 들어갈 수 있다. 들어간 particle은 지도 밖으로 나간 particle처럼 인계 처리할 예정이다(아직 구현 안 함).
  - 그 밖의 시가지(`built`, 3.7%)는 숲처럼 가리는 지형(cover)으로 친다.
- **도로는 두 단계로 단순화한다.**
  - 빠른 길(`surface`=2): motorway, trunk, primary, secondary, tertiary와 각 link.
  - 느린 길(`surface`=1): unclassified, track.
  - 마을 안 길, 진입로(service), 보행로, 산길은 뺀다. OSM 도로 12,120 km 가운데 약 4,900 km다.
  - 그림은 `candidates/simple_B_chungju_map.png`에 있다.
- **지도는 이 정도로 두고**, 추가가 필요한 부분은 particle 모델을 만들면서 고친다. 건물 윤곽, 제한속도, 수목 밀도, 교통량은 당장 넣지 않는다.
- **도로는 그래프로도 저장한다.** 차량 particle은 그래프 위에서, 도보 particle은 격자 위에서 움직이는 구조를 염두에 둔 것이다.

**만든 것**
- `rl_search/real_terrain.py`: 원격 COG 범위 읽기, Overpass 받기, 레이어 생성, 도로 그래프.
- `rl_search/scripts/make_real_terrain.py`: 실행 방법은 아래와 같다.
  ```
  python -m rl_search.scripts.make_real_terrain --region chungju
  ```
  - 원본 데이터는 `rl_search/data/real/chungju/raw/`에 캐시한다(43 MB, git 제외).
  - 결과 파일:
    - `rl_search/data/terrain_real_chungju_60km.npz`: 지금 `Terrain` 형식, git 제외.
    - `rl_search/data/real/chungju/road_graph.npz`
    - 그림 3장과 `terrain_stats.json`
- `rl_search/terrain.py`에 바뀐 점:
  - `TerrainConfig.source/origin_lat/origin_lon`을 추가했다.
  - 선택 레이어 `city`, `built`를 추가했다. 없으면 0으로 읽는다. cover는 숲 | built로 계산한다.
  - 합성 지형은 모든 레이어가 이전 코드와 완전히 같게 읽힘을 확인했다.
- 실제 지형 레이어의 의미:
  - `mountain`은 비어 있다. 지형 통과 불가는 급경사(≥ 32°)뿐이다.
  - `fords`는 물 위 다리 칸이다(고속도로 다리 제외).
  - `stream`은 OSM 하천 선이다. 그리기만 하고 통과할 수 있다.
- 도로 그래프(`road_graph.npz`):
  - 23,573개 노드, 27,452개 간선이다. OSM 공유 노드에서 나눴다.
  - 간선마다 길이, 단계, 도로 종류(`highway_codes` 순서), 일방통행 여부, 다리 여부를 담는다.
  - 간선 경로는 `edge_poly`와 `edge_poly_offset`에, 노드가 인계 구역 안인지는 `node_city`에 있다.

**결과** (`rl_search/data/real/chungju/terrain_overview.png`, `terrain_layers.png`, `terrain_crossings.png`)

| 항목 | 실제 충주·제천 | 합성 지형 (seed 7) |
|---|---|---|
| 고도 1–99% | 57–889 m | 90–1115 m |
| 숲 | 75% | 28% |
| 물(통과 불가) | 2.8% | 2.0% |
| 급경사 ≥ 32° | 7.0% | 7.8% |
| 통과 불가 전체 | 9.7% | 14.1% (산지 8% 포함) |
| 큰 도시(인계) / 그 밖 시가지 | 1.1% / 3.7% | – |
| 빠른 길 / 느린 길 칸 | 4.3% / 5.5% | 0 |
| 다리 | 158곳 (449칸) | 여울 |

- **강 틈 수정:**
  - WorldCover만 쓰면 남한강이 50 m 격자에서 끊기거나 한 칸 두께로 이어진 곳이 있었다. 도주자가 다리 없이 강을 건널 수 있었다(예: x −16 km, y 3.3 km).
  - OSM 강 중심선을 따라 두 칸 폭을 물로 칠하고, 300 m 이하 틈을 메웠다.
  - 확인 결과, 다리를 막으면 육지가 1,954 km²와 1,547 km²로 나뉘고, 다리를 열면 하나로 이어진다.
  - 충주댐 둑도 물로 막혔다. 일반 통행이 어려운 곳이라 그대로 둔다.

**주의할 점**
- Copernicus DEM은 표면 모델(DSM)이다. 숲 가장자리와 건물에서 경사가 실제보다 크게 나올 수 있다.
- WorldCover 시가지는 덩어리 단위다. 건물 하나하나가 아니다.
- 포장 여부 태그가 붙은 OSM 도로는 절반 정도다. 단순화한 두 단계는 등급만 쓰므로 영향이 없다.
- OSM 데이터는 시간이 지나면 바뀐다. `osm_highways.json`에 받은 시각이 있다. `--refresh` 없이 다시 실행하면 캐시를 그대로 쓴다.

**다음 단계 (사용자와 합의한 순서)**
1. ~~지형 파일 만들기~~ (완료)
2. 기존 도보 particle 모델(`target.py`의 infiltrator)을 이 지형에서 그대로 돌린다. 합성 지형과 분포가 어떻게 다른지 그림으로 확인한다.
   이때 인계 구역 처리(들어간 particle을 멈추고 시각 기록)를 같이 넣을지 정한다.
3. 차량 모델: 도로 그래프 위 이동, 교차로 선택, 단계별 속도. 그다음 차에서 내리는 도보 전환을 섞는다.

**미결 질문**
- 도로 단계별 속도와, 도보 particle이 도로를 얼마나 선호할지.
- 주덕 근처 4 km² 덩어리가 산업단지라면 인계 구역에서 뺄지.
- 고속도로(울타리)를 도보 particle이 건널 수 있게 할지. 지금은 막지 않는다.
