# E03 fixed mechanism controls:400 rounds

Single seed, official-test **test-oracle peaks**; exploratory comparisons, not independent validation. No online tuning.

|Method|test-oracle peak|Round|Final|Last20|Last50|Final CE|Peak minus E03 (pp)|
|---|---:|---:|---:|---:|---:|---:|---:|
|C2|36.3200|308|35.7600|35.8935|35.8122|3.753516|-0.9700|
|T01|36.6100|395|36.1900|36.1680|35.9560|3.653426|-0.6800|
|T04|37.2900|388|36.6300|36.8410|36.7704|3.612657|+0.0000|
|T05|36.6800|308|36.3000|35.8965|35.7560|3.726218|-0.6100|
|T06|36.8000|326|36.2300|36.1025|35.9038|3.642075|-0.4900|
|T09|36.8300|385|36.7100|36.2905|36.1516|3.657138|-0.4600|
|T02|36.6400|308|35.6300|35.6875|35.6548|3.695007|-0.6500|
|T03|36.6000|387|35.8800|36.1210|35.9894|3.675725|-0.6900|
|T07|36.4900|330|36.2800|36.0205|35.9776|3.701662|-0.8000|
|T08|36.4200|308|36.0800|35.9040|35.7182|3.668250|-0.8700|
|T10|36.6000|385|35.8000|35.8305|35.7144|3.697380|-0.6900|
|B00|36.4000|232|35.6700|35.8320|35.7084|3.740723|-0.8900|
|E03|37.2900|350|36.8400|36.7885|36.6946|3.666226|+0.0000|
|N08|37.1800|357|36.7000|36.7085|36.6558|3.582589|-0.1100|
|C1|36.0200|387|35.5300|35.5650|35.4244|3.743544|-1.2700|

## Fixed contrasts

|Comparison (first minus second)|Peak difference (pp)|Last20 difference (pp)|Last50 difference (pp)|
|---|---:|---:|---:|
|T01 - T02|-0.0300|+0.4805|+0.3012|
|T03 - T01|-0.0100|-0.0470|+0.0334|
|T04 - E03|+0.0000|+0.0525|+0.0758|
|T05 - E03|-0.6100|-0.8920|-0.9386|
|T06 - T05|+0.1200|+0.2060|+0.1478|
|T07 - T08|+0.0700|+0.1165|+0.2594|
|T09 - E03|-0.4600|-0.4980|-0.5430|
|T09 - C2|+0.5100|+0.3970|+0.3394|
|T10 - T01|-0.0100|-0.3375|-0.2416|

Highest new observed test-oracle peak: T04 (37.2900%). Exceeds historical E03: False.
Highest new Last20: T04 (36.8410%). Historical E03 Last20: 36.7885%.
These are descriptive rankings. A peak/late-mean tradeoff, mechanism and total update-scale changes must be considered together before selecting1-2 research candidates; they do not establish causal or generalization superiority.

Failed/incomplete: {}

Mechanism diagnostics compare each run with E03 on its own uploads. They do not isolate a unique causal explanation from total update scale.
T05 changes physical-reference pooling; T06 also changes LOO averaging to a direct-sum pairwise-overlap cosine. Their differences cannot be attributed only to larger support.
T09 preserves E03 fusion and negative-projection coefficient, restoring only underweighted negative-slot residuals. Residuals are not established semantic knowledge.

![Full Accuracy](accuracy.png)

![Full CE](ce.png)
