Clean scenarios: LKP at the map centre, random target heading, launch 5-15 km from the LKP inside the map. Task-mode SAROPS (search 10 min, transit <= 10 min, k = 1-3), launch 60 s apart. Held-out POS = detection probability of a target drawn from the model.

| mode | held-out POS mean | min | POS at 15 / 30 / 45 min | time any pair < 300 m (s, mean / max) | closest pair (min) | holds per flight (min of holding) | plans without a feasible option |
|---|---|---|---|---|---|---|---|
| 1 | 0.694 | 0.353 | 0.339 / 0.572 / 0.663 | 0 / 0 | nan | 0.0 (0.0) | 0 |
| 5 no sep | 0.908 | 0.666 | 0.528 / 0.827 / 0.889 | 383 / 745 | 0 | 0.0 (0.0) | 0 |
| 5 sep | 0.899 | 0.653 | 0.497 / 0.798 / 0.874 | 154 / 415 | 0 | 3.4 (7.4) | 28 |

Separation cost (5 sep - 5 no sep, paired): -0.0091 ± 0.0018, better in 0/10

| seed | heading | speed km/h | launch-LKP km | escaped | 1 UAV | 5 no sep (close s / min m) | 5 sep (close s / min m, holds) |
|---|---|---|---|---|---|---|---|
| 400 | 72 | 24 | 8.3 | 0% | 0.893 | 0.991 (365 / 30) | 0.990 (15 / 157, 2) |
| 401 | 250 | 21 | 9.8 | 0% | 0.430 | 0.782 (240 / 33) | 0.770 (45 / 2, 2) |
| 402 | 95 | 28 | 12.0 | 0% | 0.640 | 0.928 (745 / 2) | 0.912 (185 / 2, 3) |
| 403 | 64 | 29 | 6.3 | 0% | 0.917 | 0.990 (305 / 45) | 0.982 (35 / 115, 3) |
| 404 | 109 | 22 | 7.4 | 0% | 0.660 | 0.954 (125 / 52) | 0.945 (20 / 168, 5) |
| 405 | 320 | 18 | 11.7 | 0% | 0.713 | 0.887 (405 / 16) | 0.885 (365 / 48, 1) |
| 406 | 26 | 16 | 7.6 | 0% | 0.944 | 0.998 (450 / 12) | 0.995 (140 / 2, 6) |
| 407 | 58 | 19 | 14.5 | 0% | 0.795 | 0.968 (330 / 23) | 0.959 (135 / 35, 3) |
| 408 | 216 | 18 | 10.5 | 0% | 0.353 | 0.666 (330 / 41) | 0.653 (415 / 60, 2) |
| 409 | 112 | 20 | 14.2 | 0% | 0.598 | 0.918 (535 / 0) | 0.900 (180 / 0, 7) |