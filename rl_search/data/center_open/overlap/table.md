Saved flights, SAROPS, scenarios 500-509, held-out particles, 10 s paths. Means over scenarios.

| constraint | UAVs | config | POS | total swept (sum over UAVs) km2 | team swept km2 | swept by 2+ UAVs km2 (share of team) | second UAV within 10 / 30 min | particles seen by 2+ UAVs | POS, first UAV only (upper bound on the loss) | new rectangle inside another's current rectangle | rectangle km2 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| no constraint | 3 | POS only | 0.611 | 33.6 | 24.9 | 6.8 (27%) | 74% / 99% | 32% | 0.560 (-0.051 ± 0.007) | 12% | 1.47 |
| no constraint | 3 | A + POS, x 70 | 0.614 | 35.0 | 26.6 | 6.5 (25%) | 82% / 98% | 31% | 0.566 (-0.048 ± 0.007) | 9% | 1.59 |
| no constraint | 3 | A + POS, x 85 | 0.616 | 35.2 | 27.3 | 6.1 (22%) | 79% / 98% | 33% | 0.562 (-0.055 ± 0.008) | 9% | 1.61 |
| no constraint | 5 | POS only | 0.720 | 56.8 | 38.6 | 11.6 (30%) | 75% / 97% | 51% | 0.628 (-0.092 ± 0.011) | 15% | 1.52 |
| no constraint | 5 | A + POS, x 70 | 0.721 | 58.3 | 41.0 | 11.4 (28%) | 74% / 98% | 51% | 0.636 (-0.085 ± 0.009) | 11% | 1.56 |
| no constraint | 5 | A + POS, x 85 | 0.726 | 58.9 | 42.7 | 10.7 (25%) | 77% / 98% | 51% | 0.633 (-0.092 ± 0.010) | 10% | 1.57 |
| time rule | 3 | POS only | 0.594 | 33.4 | 24.9 | 6.5 (26%) | 78% / 99% | 31% | 0.546 (-0.047 ± 0.006) | 9% | 1.56 |
| time rule | 3 | A + POS, x 70 | 0.592 | 34.5 | 26.6 | 6.0 (23%) | 81% / 99% | 28% | 0.548 (-0.044 ± 0.005) | 8% | 1.60 |
| time rule | 3 | A + POS, x 85 | 0.596 | 35.1 | 27.2 | 6.1 (22%) | 80% / 98% | 29% | 0.550 (-0.046 ± 0.007) | 10% | 1.61 |
| time rule | 5 | POS only | 0.699 | 56.4 | 38.9 | 11.1 (29%) | 73% / 98% | 49% | 0.618 (-0.081 ± 0.010) | 12% | 1.52 |
| time rule | 5 | A + POS, x 70 | 0.704 | 57.4 | 41.3 | 10.6 (26%) | 77% / 99% | 47% | 0.624 (-0.080 ± 0.010) | 10% | 1.59 |
| time rule | 5 | A + POS, x 85 | 0.706 | 57.6 | 42.8 | 10.2 (24%) | 75% / 97% | 46% | 0.625 (-0.081 ± 0.009) | 10% | 1.64 |
| time + rectangle rule | 3 | POS only | 0.595 | 32.4 | 26.4 | 4.8 (18%) | 72% / 97% | 27% | 0.551 (-0.044 ± 0.007) | 3% | 1.47 |
| time + rectangle rule | 3 | A + POS, x 70 | 0.600 | 33.6 | 27.6 | 4.8 (17%) | 76% / 98% | 26% | 0.561 (-0.039 ± 0.005) | 3% | 1.51 |
| time + rectangle rule | 3 | A + POS, x 85 | 0.595 | 34.1 | 28.9 | 4.1 (14%) | 81% / 99% | 25% | 0.553 (-0.042 ± 0.007) | 3% | 1.58 |
| time + rectangle rule | 5 | POS only | 0.710 | 54.4 | 41.0 | 8.8 (21%) | 69% / 95% | 45% | 0.633 (-0.077 ± 0.009) | 3% | 1.42 |
| time + rectangle rule | 5 | A + POS, x 70 | 0.704 | 55.5 | 42.2 | 8.8 (21%) | 73% / 99% | 42% | 0.634 (-0.070 ± 0.009) | 3% | 1.47 |
| time + rectangle rule | 5 | A + POS, x 85 | 0.707 | 56.0 | 43.4 | 8.2 (19%) | 70% / 98% | 44% | 0.628 (-0.079 ± 0.010) | 3% | 1.52 |

| constraint | UAVs | config | close time s | both on lanes, rectangles overlap | both on lanes, rectangles apart | one on lanes, one moving or holding | both moving or holding |
|---|---|---|---|---|---|---|---|
| no constraint | 3 | POS only | 100 | 73 | 4 | 14 | 10 |
| no constraint | 3 | A + POS, x 70 | 106 | 74 | 2 | 18 | 11 |
| no constraint | 3 | A + POS, x 85 | 122 | 82 | 0 | 28 | 14 |
| no constraint | 5 | POS only | 209 | 130 | 5 | 60 | 17 |
| no constraint | 5 | A + POS, x 70 | 216 | 150 | 2 | 60 | 4 |
| no constraint | 5 | A + POS, x 85 | 206 | 122 | 1 | 50 | 34 |
| time rule | 3 | POS only | 83 | 68 | 0 | 1 | 13 |
| time rule | 3 | A + POS, x 70 | 58 | 44 | 0 | 2 | 10 |
| time rule | 3 | A + POS, x 85 | 54 | 42 | 0 | 2 | 8 |
| time rule | 5 | POS only | 246 | 102 | 0 | 12 | 132 |
| time rule | 5 | A + POS, x 70 | 116 | 74 | 0 | 15 | 28 |
| time rule | 5 | A + POS, x 85 | 74 | 38 | 0 | 8 | 28 |
| time + rectangle rule | 3 | POS only | 30 | 17 | 0 | 0 | 13 |
| time + rectangle rule | 3 | A + POS, x 70 | 28 | 18 | 0 | 1 | 8 |
| time + rectangle rule | 3 | A + POS, x 85 | 34 | 16 | 0 | 8 | 8 |
| time + rectangle rule | 5 | POS only | 181 | 48 | 0 | 4 | 128 |
| time + rectangle rule | 5 | A + POS, x 70 | 42 | 27 | 0 | 2 | 12 |
| time + rectangle rule | 5 | A + POS, x 85 | 51 | 25 | 0 | 11 | 14 |
