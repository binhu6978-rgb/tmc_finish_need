# G03 GPU0固定1000轮结果

原G03能力层级均衡，seed1，GPU0，已实际完成1000轮，未改算法或追加其他方法。**Peak为38.58%，第774轮；Final为38.19%，Last20为38.0970%。**

| 运行范围 | test-oracle peak | 轮次 | Final | Last20 | Last50 | Final CE |
|---|---:|---:|---:|---:|---:|---:|
| 历史G03 400轮 | 38.45 | 339 | 37.78 | 37.6585 | 37.6056 | 3.587194 |
| 新运行前400轮 | 38.45 | 339 | 37.78 | 37.6585 | 37.6056 | 3.587194 |
| 新运行完整1000轮 | 38.58 | 774 | 38.19 | 38.0970 | 37.9922 | 3.937973 |

前400轮每一轮Accuracy和CE均与历史G03完全相等；原七档初始化、分区和前400轮客户端调度相同，训练配置除epochs预算外一致。当前未保存模型权重，因此只报告真实标量一致，不宣称参数逐位比较。

增加600轮后，Peak增加0.13个百分点、Final增加0.41、末20轮均值增加0.4385、末50轮均值增加0.3866。末20轮Accuracy标准差从0.2840降至0.1130。**收益主要是后段准确率趋稳，峰值提升很小。** 不同末段窗口来自不同训练阶段，1000轮还有更多test-oracle观测机会，不能把此差值当成同预算新算法增益。

Accuracy提高同时CE恶化：最终CE从3.587194升至3.937973；末20轮平均CE从3.561571升至3.939567。它说明类别是否预测正确与概率损失的趋势不同，不能将准确率趋稳解释成所有预测质量指标都改善。结果与错误预测置信度上升等解释相容，但当前没有逐样本置信度记录，无法确认具体原因；本次不为解释它另开实验。

与并行完成的P候选相比，P05在400轮取得38.64%的更高探索Peak，P04在400轮的最终CE为3.410200；这些不能据不同训练预算直接排成统一算法优劣。400轮候选的配对比较见[十组结果报告](G03_REFINEMENTS10_RESULTS.md)。

## 完成和执行证据

- 完整1000条有限Full评估和1000条聚合诊断，success=true，requested/completed均为1000，峰值summary与CSV一致。
- 原完整CIFAR-100训练50000/测试10000、原Dirichlet α=.3分区、七模型、100客户端、10slots、5local epochs、原SGD及确定性Full-only协议。实际Full训练slot为1470。
- 最终内存Full参数形状合计11,424,356，全部有限。按用户要求不保存任何.pt；checkpoint为空，只有峰值标量，没有峰值模型文件。
- 总墙钟约115.23分钟；服务器聚合累计63.36秒、Full评估598.82秒；allocator峰值allocated约1117.02MiB、reserved1252MiB。独占GPU0单任务与GPU1双任务的墙钟不可直接视作算法成本差异。
- 在相同服务器项目新增独立入口运行，没有修改正在运行的P训练文件。冻结入口提交为`d64fa99d43a2cc01918852b0918073204bd3be6a`。

服务器唯一项目：`/home/hh/TMC_re_e03_scope_components400_20261019`。
结果目录：`runs/g03_full1000_seed1_gpu0_nopt_20261021/g03/G03_full_only_1000_seed1_deterministic/`。

- [原始每轮Accuracy/CE](evidence/g03_completed_results/g1000/g03/G03_full_only_1000_seed1_deterministic/accuracy_each_round.csv)
- [完成标记](evidence/g03_completed_results/g1000/g03/G03_full_only_1000_seed1_deterministic/completed.json)、[最终内存模型检查](evidence/g03_completed_results/g1000/g03/G03_full_only_1000_seed1_deterministic/final_model_check.json)
- [1000轮Accuracy/CE对照图](evidence/g03_completed_results/analysis/g03_1000_accuracy_ce.png)、[数值表](evidence/g03_completed_results/analysis/g03_1000_comparison.csv)

单seed官方测试探索，Peak均标为test-oracle peak，不作为独立验证集选择后的无偏性能。本次不更改超参数、不延长轮次、不追加方法；完成自动任务已暂停。
