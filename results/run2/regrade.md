# Run 2 — original vs re-graded accuracy (ADR-27)

Same saved answers, two graders. **Original** = the grader run 2 was trained and
evaluated with (reproduces every stored verdict). **Re-graded** = accents ignored,
and boiling-point answers within ±1 °C accepted (ground truths are whole degrees).
95% bootstrap CIs (10,000 resamples, seed 42).

| agent | n | original | re-graded | gained | lost |
|---|---|---|---|---|---|
| atar@6144 | 200 | 0.595 [0.525, 0.665] | 0.660 [0.595, 0.725] | 13 | 0 |
| atar@12288 | 200 | 0.660 [0.595, 0.725] | 0.715 [0.650, 0.775] | 11 | 0 |
| atar@18432 | 200 | 0.755 [0.695, 0.815] | 0.820 [0.765, 0.870] | 13 | 0 |
| atar@24576 | 200 | 0.820 [0.765, 0.875] | 0.875 [0.825, 0.920] | 11 | 0 |
| zeroshot | 200 | 0.360 [0.295, 0.425] | 0.430 [0.360, 0.495] | 14 | 0 |
| react | 200 | 0.715 [0.650, 0.775] | 0.725 [0.665, 0.785] | 2 | 0 |
| random | 200 | 0.340 [0.275, 0.405] | 0.380 [0.315, 0.445] | 8 | 0 |

## Per tier (original → re-graded)

| agent | tier 1 | tier 2 | tier 3 | tier 4 | tier 5 |
|---|---|---|---|---|---|
| atar@6144 | 0.74 → 0.74 | 0.55 → 0.89 | 0.84 → 0.84 | 0.14 → 0.14 | 0.56 → 0.56 |
| atar@12288 | 0.90 → 0.90 | 0.61 → 0.89 | 0.88 → 0.88 | 0.27 → 0.27 | 0.40 → 0.40 |
| atar@18432 | 0.90 → 0.90 | 0.66 → 1.00 | 0.86 → 0.86 | 0.57 → 0.57 | 0.68 → 0.68 |
| atar@24576 | 0.92 → 0.92 | 0.68 → 0.97 | 0.84 → 0.84 | 0.68 → 0.68 | 1.00 → 1.00 |
| zeroshot | 0.56 → 0.56 | 0.50 → 0.87 | 0.32 → 0.32 | 0.03 → 0.03 | 0.32 → 0.32 |
| react | 0.90 → 0.90 | 0.45 → 0.50 | 0.68 → 0.68 | 0.62 → 0.62 | 0.96 → 0.96 |
| random | 0.46 → 0.46 | 0.39 → 0.61 | 0.30 → 0.30 | 0.16 → 0.16 | 0.36 → 0.36 |

## Paired: atar@24576 vs react (same tasks)

| grader | only atar@24576 right | only react right | exact McNemar p |
|---|---|---|---|
| original | 31 | 10 | 0.0015 |
| re-graded | 40 | 10 | < 0.0001 |

## Every changed verdict

| agent | tier | answer | ground truth | now |
|---|---|---|---|---|
| atar@6144 | 2 | Brasília | Brasilia | correct |
| atar@6144 | 2 | Bogotá | Bogota | correct |
| atar@6144 | 2 | 78.37 | 78 | correct |
| atar@6144 | 2 | 356.73°C | 356 | correct |
| atar@6144 | 2 | 80.1°C | 80 | correct |
| atar@6144 | 2 | 64.7°C | 65 | correct |
| atar@6144 | 2 | Brasília | Brasilia | correct |
| atar@6144 | 2 | 357.0 | 356 | correct |
| atar@6144 | 2 | Bogotá | Bogota | correct |
| atar@6144 | 2 | 357.0 | 356 | correct |
| atar@6144 | 2 | 64.7°C | 65 | correct |
| atar@6144 | 2 | 118.1°C | 118 | correct |
| atar@6144 | 2 | 61.7 | 61 | correct |
| atar@12288 | 2 | Brasília | Brasilia | correct |
| atar@12288 | 2 | The boiling point of ethanol (C2H5OH) is 78.37 degrees Celsius at standard atmos | 78 | correct |
| atar@12288 | 2 | The boiling point of acetone is approximately 56.5 degrees Celsius. | 56 | correct |
| atar@12288 | 2 | 356.73°C | 356 | correct |
| atar@12288 | 2 | 80.1°C | 80 | correct |
| atar@12288 | 2 | Brasília | Brasilia | correct |
| atar@12288 | 2 | The boiling point of mercury is 356.73 degrees Celsius at standard atmospheric p | 356 | correct |
| atar@12288 | 2 | Bogotá | Bogota | correct |
| atar@12288 | 2 | The boiling point of mercury in Celsius is 356.73 degrees Celsius. | 356 | correct |
| atar@12288 | 2 | The boiling point of acetic acid is 118.1 degrees Celsius. | 118 | correct |
| atar@12288 | 2 | The boiling point of chloroform (CHCl₃) is approximately 61.7 degrees Celsius. | 61 | correct |
| atar@18432 | 2 | Brasília | Brasilia | correct |
| atar@18432 | 2 | Bogotá | Bogota | correct |
| atar@18432 | 2 | The boiling point of ethanol is 78.37°C. | 78 | correct |
| atar@18432 | 2 | 356.73°C | 356 | correct |
| atar@18432 | 2 | 80.1°C | 80 | correct |
| atar@18432 | 2 | The boiling point of methanol is 64.5°C. | 65 | correct |
| atar@18432 | 2 | Brasília | Brasilia | correct |
| atar@18432 | 2 | 357.0 | 356 | correct |
| atar@18432 | 2 | Bogotá | Bogota | correct |
| atar@18432 | 2 | 357.0 | 356 | correct |
| atar@18432 | 2 | 64.7°C | 65 | correct |
| atar@18432 | 2 | 118.1°C | 118 | correct |
| atar@18432 | 2 | 61.7 | 61 | correct |
| atar@24576 | 2 | The capital city of Brazil is Brasília. | Brasilia | correct |
| atar@24576 | 2 | 78.3°C | 78 | correct |
| atar@24576 | 2 | 356.73°C | 356 | correct |
| atar@24576 | 2 | 80.1°C | 80 | correct |
| atar@24576 | 2 | 64.7°C | 65 | correct |
| atar@24576 | 2 | Brasília | Brasilia | correct |
| atar@24576 | 2 | 357.0 | 356 | correct |
| atar@24576 | 2 | Bogotá | Bogota | correct |
| atar@24576 | 2 | 357.0 | 356 | correct |
| atar@24576 | 2 | 118.1°C | 118 | correct |
| atar@24576 | 2 | The boiling point of chloroform is 61.7°C. | 61 | correct |
| zeroshot | 2 | Brasília | Brasilia | correct |
| zeroshot | 2 | Bogotá | Bogota | correct |
| zeroshot | 2 | The boiling point of ethanol is 78.37 degrees Celsius. | 78 | correct |
| zeroshot | 2 | The boiling point of acetone in Celsius is 56.5 degrees Celsius. | 56 | correct |
| zeroshot | 2 | The boiling point of mercury is 356.73 degrees Celsius. | 356 | correct |
| zeroshot | 2 | 80.1 degrees Celsius. | 80 | correct |
| zeroshot | 2 | The boiling point of methanol is approximately 64.5 degrees Celsius. | 65 | correct |
| zeroshot | 2 | Brasília | Brasilia | correct |
| zeroshot | 2 | The boiling point of mercury is 356.73 degrees Celsius. | 356 | correct |
| zeroshot | 2 | Bogotá | Bogota | correct |
| zeroshot | 2 | The boiling point of mercury is approximately 356.73 degrees Celsius. | 356 | correct |
| zeroshot | 2 | Boiling point of methanol is 64.7°C. | 65 | correct |
| zeroshot | 2 | The boiling point of acetic acid is 118.1 degrees Celsius. | 118 | correct |
| zeroshot | 2 | The boiling point of chloroform (CHCl₃) is approximately 61.7 degrees Celsius. | 61 | correct |
| react | 2 | Bogotá | Bogota | correct |
| react | 2 | The boiling point of mercury (mercurous) in Celsius is 356.73 degrees Celsius. | 356 | correct |
| random | 2 | The capital city of Colombia is Bogotá. | Bogota | correct |
| random | 2 | 55.4 | 56 | correct |
| random | 2 | The boiling point of mercury is 356.73 degrees Celsius. | 356 | correct |
| random | 2 | The boiling point of methanol is approximately 64.7 degrees Celsius at standard  | 65 | correct |
| random | 2 | Brasília | Brasilia | correct |
| random | 2 | The boiling point of mercury is 357 degrees Celsius. | 356 | correct |
| random | 2 | The boiling point of methanol is 64.7 degrees Celsius. | 65 | correct |
| random | 2 | The boiling point of chloroform is 61.7 degrees Celsius. | 61 | correct |
