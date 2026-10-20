# 最后两组确定性对照

只新增 B00-D、R3，各400轮、seed=1，在同一服务器项目
`/home/hh/TMC_re_e03_scope_components400_20261019` 内开发运行。
原 R0/R1/R2 已完成，不重新运行。

| 实验 | 第1轮分类器 | 第2–400轮分类器 | 其他参数 |
|---|---|---|---|
| R0 | M2 | M2 | M2 |
| R1 | Native | Native | M2 |
| R2 | M2 | Native | M2 |
| R3 | Native | M2 | M2 |
| B00-D | Native | Native | Native |

M2评分与 λ=4 均不改。R3复用 `Local12Hooks(l04)`，仅增加可选的结束轮次1。
原start-round默认1、end-round默认None，已有R0/R1/R2算法保持不变。
R3第1轮只替换classifier.*的覆盖归一化聚合，第2轮起直接使用既有M2结果；
诊断仍计算同输入E03差异，第2轮后应严格为0。该条件纳入自动最终汇总。
B00-D使用原有 `MethodHooks(baseline)` / `Aggregation_AdaptiveFL`，不计算M2方向风险。

三组此前训练与两组新训练均使用相同严格确定性策略、TF32关闭、cuBLAS `:4096:8`；
原始七模型、50000训练样本、100客户端、10槽位、5本地epoch、batch50、原SGD、
缓存α=0.3分区和Full-only隔离评价全部保留。入口仍复用原AdaptiveFL训练循环。

## 已有三组结果

| ID | test-oracle peak | 轮次 | Final | Last20 | Last50 | Final CE |
|---|---:|---:|---:|---:|---:|---:|
| R0 |36.5600|308|35.9100|35.8475|35.7176|3.676311|
| R1 |36.4000|308|35.8400|35.6910|35.6068|3.680789|
| R2 |36.6600|308|36.0600|35.9115|35.8070|3.679807|

当前不能据此判定首轮与后续作用；等待固定R3和同环境B00-D，不再加其他方法。

## 启动

```bash
nohup /home/hh/miniconda3/envs/hh/bin/python -u tools/run_l04_final_pair.py \
  --execute --dataset-root /home/hh/ICML_finish/data \
  > runs/l04_final_pair_driver.stdout.log 2> runs/l04_final_pair_driver.stderr.log < /dev/null &
```

父脚本只启动B00-D和R3。GPU0空闲显存至少6000MiB时并行两组，否则串行，不修改训练配置。
每组输出独立，新根 `runs/e03_l04_final_pair400_seed1/`；禁止覆盖已有目录。
完整400轮后核验CSV、latest和best；不自动重试。后台运行，不实时监控。

## 完成后分析

自动汇总复用三组旧CSV，检查共同初始化、客户端调度和数值执行配置。
输出五组表、Accuracy/CE曲线及4策略的条件差异：

- 首轮Native影响：R3−R0，以及R1−R2。
- 后续Native影响：R2−R0，以及R1−R3。
- 非加性：R1−R2−R3+R0。
- 同环境方向门控：R0−B00-D。

以上是固定训练轨迹的描述性对照，不假设效应线性，也不提供单seed显著性检验。
“第1轮分类器保护”同时影响首次上传融合及首次SGD增量的权重，不能仅凭本对照将二者分离。
Peak为官方测试集test-oracle peak，最终科研决策同时考虑Final、Last20/50及CE。
依据结果保留E03或某个明确的分类器作用域策略；无一致收益则停止当前路线。
不得把classifier保护直接宣称为功能语义保护，不再新增Q、Memory、Gate、seed或搜索。
