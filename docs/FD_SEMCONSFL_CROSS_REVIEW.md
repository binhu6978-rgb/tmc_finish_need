# FD-SemConsFL 交叉复核与最小诊断增强

复核基点：`ce7e2491cf70d04402816aaeccc1fe064d4cf9e3`。
工作分支：`feature/fd-semconsfl-cifar100`。

## 结论与修改边界

正式 A2 的算法、超参数和两类 Memory 更新规则保持不变。本轮增加汇总诊断、
原始分区的内存重生成核验及相应测试。复核与合成测试未发现需要修改正式决策逻辑的
确定性 Bug；置信度保护强度与 Gate 干预强度属于后续机制实验需要评价的研究问题。

以下 17 个文件与复核基点逐字节一致：原始基线的 10 个关键源码文件，
`semconsfl/measurement.py`、`semconsfl/semantic_memory.py`、
`semconsfl/aggregation.py`、`semconsfl/config.py`、`configs/fd_semconsfl.json`、
`experiments/baseline.py`、`experiments/run_semconsfl.py`。
原始 9 个数据分区 JSON 也逐字节一致。训练／测试集、数据增强、七档模型、
本地 CE／SGD 和调度保持原样。

`reliability_memory.py` 只额外返回原始 IQR 和下界触发标记；原有浮点计算顺序、
z/r/R/gamma 和历史写入逻辑保持一致。`trainer.py` 保存未经模式覆盖的决策原因，
在聚合、同步及 Memory 更新完成后调用纯诊断函数。诊断只读取已生成日志和旧 Anchor
快照，不增加模型前向，也不向参数、配置或 Memory 写回任何值。

## P0 正确性核验

| 环节 | 代码核验与测试证据 |
|---|---|
| 测量 | 相同去重索引、标签顺序、固定 batch；`no_grad`；恢复模块模式、buffer、Python/NumPy/CPU/全部 CUDA RNG；原始本地训练函数与 DataLoader 不变。CPU/CUDA 权重、梯度、训练 batch 顺序一致性测试通过。 |
| Semantic Memory | 所有槽位读取同一旧快照；round0 不写 Anchor；先决策聚合再更新；同一物理客户端一票；无 Anchor、支持不足、错误质量过低等情况是缺失证据。 |
| 数值范围 | Q 是非真实类别的条件概率分布；JS 位于 [0,ln2]，signed drift 位于 [-ln2,ln2]；正式 anomaly 位于 [0,ln2]。正负极值、置信度保护和缺失输入测试通过。 |
| Reliability | 旧同档中位数/IQR、旧客户端有效参与记录；重复槽位只写一次客户端事件；z 截断 [0,1]；缺失／过期历史中性；IQR=0、零异常保持有限值。对比基点原类，除新增遥测字段外的整个 decision 字典完全一致。 |
| 聚合 | 全一 Gate 与 round0 直接调用原始函数；Gate 仅乘更新分子；覆盖分母不变；未覆盖坐标保持原值；Full-only 单覆盖、不同前缀／样本权重测试通过。 |

round0 聚合后才完成首次统一同步；round1 的训练前签名可在该轮聚合后建立 Anchor，
round2 起才可能使用它评分。当前 signed drift 和旧历史共同产生决策，本轮新 z 仅供未来使用。

## P1 新增字段及统计口径

每轮新增 `gating_diagnostics`，保存计数、比例和固定五个分位数，不重复保存 B/A 向量。
每槽位只增加 `decision_reason`，以及完成 profile 校准时的 `profile_iqr_raw`、
`iqr_floor_hit`。新诊断耗时为 `timings.gate_diagnostics_seconds`，单独列出。

| 字段（位于 gating_diagnostics 下） | 含义与分母 |
|---|---|
| `available`, `observed_slots` | 是否执行过观察器、执行观察器的实际槽位数。A0 没有语义观测。 |
| `class_pairs.possible/observed/valid_ba` | 所有槽位类别位置数、有本地样本的 `(slot,class)` 数、B/A 均有效的 `(slot,class)` 数。 |
| `valid_ba_fraction_of_observed/possible` | 有效 B/A 对数分别除以有样本类别对数／全部类别位置数。 |
| `old_anchor_classes.valid/coverage` | 旧快照中未过期的 Anchor 类别数及其占 100 类的比例；不按重复槽位累计。 |
| `class_pairs.comparable_old_anchor` | B/A 有效且具有有效旧 Anchor 的类别对数；`comparable_fraction_of_valid_ba` 的分母是有效 B/A 对数。 |
| `positive_drift/negative_drift` 及对应 fraction | 正／负 signed drift 类别对数；比例的分母为具有可比较旧 Anchor 的类别对数。 |
| `positive_drift_confidence_increased` | 正漂移且真实类置信度增量 >0 的类别对数；同时输出相对正漂移／全部可比较类别对的比例。 |
| `confidence_protected` | 正漂移且置信度增量 >1e-7，因正式保护规则被清零的类别对数；同样输出条件比例和联合比例。 |
| `physical_clients` | 物理客户端去重计数。有任一有效槽位 anomaly>0 即归入 positive；所有有效槽位 anomaly=0 才归入 zero；没有有效槽位则 missing。比例分母为有有效 anomaly 的客户端数。 |
| `decision_reasons` | 未经 A1／round0 覆盖的原始决策原因直方图；`valid` 可以包含零风险。 |
| `insufficient_profile_slots/client_slots` | 同档／客户端历史不足的槽位数，按正式决策路径的首个阻塞原因计数。 |
| `iqr_floor` | 校准槽位数、原始 IQR<=1e-6 的次数和比例；不含无法校准的槽位。 |
| `z` | 有效／缺失校准数；z=0、0<z<1、z=1 各占已校准槽位的比例。 |
| `r`, `risk`, `gamma` | 正式返回值的分布；r/risk 包含正式中性处理返回的零值，gamma 是实际应用的 Gate，A1 固定为 1。 |
| `nonneutral_gate_slots` | 实际 gamma<1 的槽位数。 |
| `profiles["0".."6"]` | 各 profile 槽位数、实际非中性 Gate 数和 gamma 分布；`full_profile=6`。 |

分布摘要包括 `total/valid/missing/coverage`、精确零／正／负计数和比例、均值、
最小值、最大值、5%/25%/50%/75%/95% 分位数。缺失分母返回 JSON `null`；
它不会被伪装为零异常或零风险观测。

沿用已有 `coverage_update_diagnostics`，分别记录 `full_only`、`partial_profiles`、
`all_profiles` 三个静态实际覆盖区域的门控后／未门控更新 L2 范数及比值。
多个更新的抵消关系改变时，范数比可以大于 1，不能据此认定 Gate 被重复缩放或失效。

## P2 三种证据的只读比较

`anomaly_comparison` 中三个指标使用相同的有效旧 Anchor 类别及样本支持数：

1. `formal`：现有正式 anomaly，不重新定义。
2. `positive_without_confidence_guard`：`sum(n_c*max(d_c,0))/sum(n_c)`。
3. `signed_mean`：`sum(n_c*d_c)/sum(n_c)`，允许为负。

三个指标均提供有效覆盖率、零／正／负比例、均值和分位数。后两个分别通过
`unguarded_vs_formal_ranks`、`signed_vs_formal_ranks` 与正式异常比较：
报告共同有效槽位数、平均／最大绝对名次变化、Spearman 系数、严格排序逆转数、
发生并列变化的样本对数。并列采用平均名次；常数序列的 Spearman 返回 `null`。
这里的排行对象是实际参与槽位，物理客户端状态另外去重统计。

这两个对照指标完全不进入 profile 校准、Reliability、Gate 或 Memory。
本轮没有新增候选 Gate、候选超参数或新的训练决策分支。

## P3 原始数据分区复核

在原始 CIFAR100 标签顺序上，直接调用未修改的
`utils.get_datatset_new.dirichlet_split_unbalanced`。参数为
`num_clients=100, alpha=.3, seed=1, min_size=10, max_retry=500, min_classes=0`。
重生成只发生在内存；原始分区文件写入次数为零。

- 缓存真实元数据：`dirichlet_unbalanced`；文件名 `_bal` 不改变该事实。
- 客户端样本数范围：262--757，总训练样本 50,000。
- 精确索引序列：100/100 客户端一致。
- 无序样本集合：100/100 客户端一致。
- A0/A1/A2：经各自入口参数解析和原加载函数读取后，规范化索引哈希全部一致。

| 哈希口径 | 值 |
|---|---|
| 原 JSON 文件字节 SHA256（核验前后相同） | `41ab2d3486663ce0100e183ab00565de137f91af07632c236c2ddf60d890ba41` |
| 规范化客户端索引序列 SHA256（缓存／重生成／三个实验臂相同） | `ba7ee37f64a31c23cf9ce4fb55925807d78cf9d3d10884eeb835fcdd30cb2b7c` |
| 规范化无序客户端集合 SHA256（缓存／重生成相同） | `532e0a3954c4902444fe0c4f6fe634073f49d0ceeb2f15e2fdd8a05706efa3d5` |

逐客户端结果和生成器源码哈希见 `docs/cross_review_partition.json`。
规范化使用 `experiments.baseline.content_hash` 的 JSON 排序键及紧凑分隔符；
集合口径先对每个客户端内部索引排序。

复核命令（输出文件必须不存在）：

```powershell
python -m experiments.check_partition --dataset-root DATA --output NEW_REPORT.json
```

此工具只适用于本轮指定的原始 seed=1、alpha=.3 场景，不下载数据或执行训练。

## P4 已执行的检查

```powershell
python -m pytest -q tests/test_fd_semconsfl.py
```

49 项合成测试通过，无失败、无跳过；其中原有 36 项、此次新增 13 项。
原有 CPU／CUDA 测量隔离与 A0/A1 轨迹测试继续通过。新增检查覆盖：

- 类别／槽位／物理客户端分母、加权对照指标和 IQR／z／Gate 计数。
- 无观察器、无 Anchor、无样本支持和零异常均能输出有限的 JSON 摘要。
- 并列排名、严格排序逆转和常数序列的缺失相关系数。
- A0/A1/A2 各自开启／关闭诊断后，模型参数、两类 Memory、槽位决定、
  Python/NumPy/CPU/CUDA RNG 和模型前向次数完全一致；A2 案例确有非中性 Gate。
- 新 Reliability 遥测与复核基点原类预先记录的六组正式决策值逐项一致；
  测试固定这些原始数值，不依赖完整 Git 历史。
- 顺序变化与样本归属变化的分区哈希能够分别检出。
- JS 正负极值、置信度保护的正式行为和有限历史下 Gate 可达边界。

13 个 Python 文件通过 AST 检查，`git diff --check` 通过。
五条 warning 来自原有 thop distutils 版本比较及原始 options.py 的字符串转义。
未执行 CIFAR100 本地训练、正式联邦训练、短程性能实验或准确率／超参数搜索。
当前没有真实训练 B/A 轨迹，因此保护触发率、IQR 饱和率和真实 Gate 分布仍待后续采集。

## P5 对两项建议的独立科学判断

### 1. 置信度保护是否过度严格？

保护规则确实会把置信度提高超过 1e-7 的正漂移类别清零；这是一项强保护策略。
是否“过度”取决于被保护更新的真实效用，本轮证据不能给出经验判断。
新增正漂移保护比例和三种异常排行能够明确它清除了多少信号。

例如，真实类为 0，B=(.2,.8,0)、A=(.3,0,.7)，旧错误参考为 (1,0)。
错误条件分布的 JS 正漂移为 ln2，但真实类 CE 从 -ln(.2) 降至 -ln(.3)，改善约 .4055。
删除保护会对这种局部分类进步施加异常惩罚。全局效用仍需反事实评价。
另一方面，小幅真类概率提高也能掩盖大幅条件混淆变化；这正是需要用真实更新效用
判断的取舍，不能由“Gate 更活跃”直接裁决。

### 2. z 截断、历史收缩和乘积是否限制 Gate 动态范围？

是，而且界限可由当前配置直接推出。z 和历史均值均在 [0,1]，最多保留 5 个旧事件，
nu=2，因此 `r<=5/7`、`R<=5/7`，在 lambda=.25 时：

`gamma >= 1 - .25*(5/7) = 23/28 ≈ .82142857`。

这比公式名义下界 .75 更保守。若只有两个有效旧事件，可达下界为 .875；
旧事件不足两个时实际 Gate 为 1。这是已冻结的收缩、截断和历史中性原则的联合结果，
不是数值 Bug。截断同时会使足够大的相对异常都映射到 z=1；实际饱和程度由诊断确认。

### 3. 当前 z 加入当前 Reliability 会怎样？

它改变“当前证据 × 旧历史持久性”的定义，并使当前异常同时影响两项因子。
例如只有一个旧有效零值、本轮 z=1 时，现行规则因旧记录不足而中性；若把当前值也计数，
即可提前满足两条记录门槛并产生风险。这样会把新出现的异常更早当成持久性证据，
增加暂时波动和小样本噪声的影响。本轮保持仅使用旧历史。

### 4. gamma_min=.5 是否更适合 Full-only？

现有代码和合成测试没有这种效用证据。Full-only 坐标由少数 Full 槽位提供学习信号，
更强抑制可能减少有害漂移，也可能削弱有用更新。若只把 lambda 改为 .5，
在现有历史上限下可达下界仍为 9/14≈.642857，而不是 .5。

后续获授权后，应在预先固定的同一批上传上比较原 Gate 与候选干预，保持覆盖分母、
本地训练和评价协议一致，记录 Full-only／共享区域影响，并加入相同范数缩放和
同 profile 风险置换对照。用固定评价协议衡量更新效用和误抑制代价，再决定是否改算法。

### 5. 代码与合成测试能否证明漂移关联 Full 更新效用？

本轮证明的是实现时序、数值公式和诊断无副作用。真实关联需要后续缓存上传的
反事实机制实验：分别计算移除槽位后重算分母、固定分母置零贡献，以及按物理客户端
成组移除的影响；考察风险与效用的关系，并控制 profile、样本量、标签组成和更新范数。
之后才能评价时间可靠性、语义风险相对一般缩步的增益和闭环训练收益。
当前数据划分、训练／测试集和实验设定继续保持原样。

## 本轮实际修改文件

```text
semconsfl/diagnostics.py                 新增，只读汇总与排行
semconsfl/reliability_memory.py          只加 IQR 遥测
semconsfl/trainer.py                     接入只读汇总与原决策原因
tests/test_fd_semconsfl.py               新增 13 项参数化后测试
experiments/check_partition.py          新增，原始函数内存核验工具
docs/cross_review_partition.json        新增，真实分区核对结果
docs/FD_SEMCONSFL_CROSS_REVIEW.md         新增，本报告
docs/FD_SEMCONSFL.md                     链接本次复核，保留初始实现记录
```

初始 `docs/baseline_audit/baseline_manifest.json` 保留为原实现版本的历史记录，
本轮不改写其大体积内容。当前源码的日志 fingerprint 会包含新诊断模块；
原有 checkpoint 的源码一致性校验继续生效，跨源码版本恢复将明确拒绝。
