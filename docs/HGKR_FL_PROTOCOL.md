# HGKR-FL 三档协议 v1（实验前冻结）

本实现独立于旧 A0/A1/A2/A3。它是历史引导的参数刷新探索，不使用语义信息，不能将方向波动叫作已证实的语义冲突。原始 AdaptiveFL、旧 FD-SemConsFL 源文件均不修改。

## 模型及原始训练条件

调用原始 `get_model_list`，width=[0.40,0.66,1.0]、depth_saved=[2]。`slim_idx=2` 表示前两段不缩窄，所有残差层均保留。

|档位|参数量|单图卷积/线性 MAC|
|---|---:|---:|
|Small|2,120,985|392,635,056|
|Medium|5,159,621|504,859,732|
|Full|11,424,356|732,678,144|

静态参数区域：全部共享 2,120,985；部分共享 3,038,636；Full-only 6,264,735。每层实际形状和初始化哈希落盘 manifest。新三档覆盖区域与旧七档不同。

CIFAR-100 原始训练 50,000 / 测试 10,000；不增加验证集或增强。原始缓存 SHA256=`41ab2d3486663ce0100e183ab00565de137f91af07632c236c2ddf60d890ba41`，Dirichlet unbalanced α=.3，100 客户端，样本数 262–757；文件名中的 bal 不表示数量均衡。

训练调用原始 `LocalUpdate_AdaptiveFL.train`：5 epoch、batch50、drop_last、CE、SGD .01×.998^round、momentum .5、weight_decay 1e-4。聚合按完整客户端样本数 n 加权，计算预算按实际 drop_last 步数计算。官方测试原函数每轮评估三档，batch128，无改动。所有峰值标记 test-oracle peak；不依据测试集调参。

## client-first 与资源约束

新协议明确改变原始 model-first available 调度：用隔离的 Python RNG 预先无放回均匀抽取 10 个物理客户端。ID 0–39 仅能 Small；40–69 能 S/M；70–99 能 S/M/F。所有组完全相同的参与顺序，不依据标签、测试结果或训练结果筛选。

每轮双预算上限固定为该轮全部客户端运行能力上限的 80%，最低不低于全 Small。预算分别是：

- 训练 MAC 代理 = Σ 5 floor(n_c/50)×50×MAC(profile)。训练 FLOP 代理取该值×6（前向2 FLOP/MAC、反向近似2倍前向），不声称为精确算子级 FLOP。
- 下载+上传 = Σ 8×参数量 byte（float32，每方向一份；不含网络协议头）。

枚举可行分配。B1 优先最大化 Full 槽位数，再优先此前获得 Full 次数较少的客户端，再尽量使用计算预算；完全相同的 B1 分配用于 B2/B3。不排除低能力客户端。

B0 从相同可行集合中、实际两类成本均处于 B1 ±2% 的分配里均匀随机抽取。B1 本身在集合内，因此总有解；不能保证每轮有不同分配，实际相同轮数和可选数量均落盘。这是近似成本匹配，不是严格相等；B1−B0 的解释必须报告剩余成本差异。若后续要求严格匹配，可预先限定双成本完全相同的可行集合再随机抽取，但真实样本量加权下集合可能只剩同一分配，无法识别 Full 次数变化；需要独立预注册更长预算配对设计，不能用测试结果事后选对照。

初始100轮协议审计（不是训练结果）：B0 S/M/F=798/157/45，B1=854/73/73；39/100轮分配完全相同。总训练MAC代理 B0=1,007,033,662,447,000、B1=1,010,102,327,451,000（B1约多0.305%）；双向通信 B0=24,133,620,376、B1=24,175,612,088 byte（B1约多0.174%）。原始实验未训练的审计输出不是正式结果。

## 可比 kernel segment 与历史

21 个 Conv2d 权重。每个统计单位定义为 `W_l[o, :I_small, :, :]`，且 `o < O_small`。这是 Small 可见的输出核片段，不是完整 Full 输出滤波器。所有档位都完整覆盖这个固定片段，输入维度不会随客户端档位变化。重置仅写这些片段；同一 Full 输出通道中 Small 输入范围以外的坐标、额外输出通道、BN 和分类器均不重置。

每轮聚合完成后，将实际服务器变化 Δ_l,o 的共享片段拉平，范数大于 1e-12 且有限时保存单位方向 u=Δ/||Δ|| 和范数。最多保存最近5次覆盖记录、最大年龄20轮。读取只用过去记录；同轮只能写一次。round0 使用独立初始模型，故该轮不写方向，避免把初始化差异当学习历史。

至少3次有效方向时：instability = 1 − ||mean(u)||，范围[0,1]；越大表示观察到的方向越不稳定。零范数不伪造方向或有效证据。没有完整共享覆盖的记录跳过；固定三档10槽位协议下任意槽位都完整覆盖共享区域。

重置 warmup 按结构性覆盖记录数达到3决定，最早零基 round4，即第5轮。两组每层数量都是 floor(.01×O_small)，不因方向零范数或排序有效性而改变预算。B2 使用隔离随机排列；B3 优先从有效历史中按 instability 降序选，随机顺序打破同分；不足时从同一随机候选补足。两组都计算同一份历史统计/随机候选和排序，额外计算路径相同。不能在跨组历史不同的情况下伪称排名来自相同模型快照。

每个客户端、轮次、层单独使用 SHA256 派生的 CPU torch.Generator；选择与采样使用两个命名域。不会消耗 DataLoader、模型或全局 CPU/CUDA RNG。相同数量片段使用相同形状的标准正态抽样；不同轨迹上的均值/方差可以不同，但两组使用完全相同的分布定义。

## 共用 reset 算子与锚定增量

先深克隆下发子模型。采样 N(μ_l,σ_l²)，μ、σ取旧服务器层的 Small 共享块整体统计（population std），覆盖选中片段。只用共享块统计是为了避免受保护 Full-only 统计改变共享核重置尺度。不是全局模型原地修改，也不是整个 Full 输出滤波器重置。

算子参考 [FedPhoenix 原论文 §4.2.1](https://papers.neurips.cc/paper_files/paper/2025/file/16e71d1a24b98a02c17b1be1f634f979-Paper-Conference.pdf) 的层统计高斯重新采样思想。本实现不复现其全部动态层稳定策略；B2 应称 FedPhoenix 风格随机核片段重置对照。

保存实际下发起点 a_c 和训练后上传 w_c；真实学习增量 d_c=w_c−a_c。服务器：

`θ'_j = θ_j + Σ_c n_c M_cj d_cj / Σ_c n_c M_cj`。

分母零的坐标保留 θ_j。重置脉冲 a_c−θ 不聚合，重置后未学习的上传不改变全局 Full。B2/B3 调用同一聚合函数。这是新研究设计，与 FedPhoenix 的直接权重平均不同。

round0 不重置，调用原始绝对权重聚合处理三档独立初始化。之后全部无重置时也调用原始 golden 聚合，并断言起点等于服务器前缀，保证退化路径逐位一致。数学等价的增量公式另测浮点容差；不让浮点求和顺序成为 B1 与 warmup B2/B3 的隐藏消融。

## 执行和恢复

使用 conda fd 的 Python；单组命令：

```powershell
& 'D:\software\minicoonda\envs\fd\python.exe' -m experiments.run_hgkr --mode B3 --dataset-root 'C:\Users\admin\Desktop\submissioning\code\ori\AAAITMC_finish_need_code\data\cifar100' --output-dir runs/hgkr_B3_seed1_100 --epochs 100 --seed 1 --gpu 0 --hgkr-config configs/hgkr_three_tier.json
```

替换 B3 /输出目录可独立运行 B0/B1/B2。两轮 smoke 使用新目录和 `--epochs 2`。不将 smoke 改 epochs 后续训100。

一次性完整套件：

```powershell
& 'D:\software\minicoonda\envs\fd\python.exe' -m experiments.hgkr_suite --dataset-root 'C:\Users\admin\Desktop\submissioning\code\ori\AAAITMC_finish_need_code\data\cifar100' --output-root runs/hgkr_suite_seed1_20261018 --launch100
```

套件先 B0/B1 两轮，再 B2/B3 两轮。四组 smoke 完成且配对核验通过，才新开四组100轮；最多两个并行进程，均GPU0。源码/配置哈希冻结，失败终止该批仍运行的自有子进程并保留日志，不自动改参或重试。完成100轮自动输出 summary.json 和 report.md，无曲线。状态、实际argv、PID、stdout/stderr均保存。

两轮 CIFAR smoke 用于真实数据/训练/测试链路，warmup 内不会重置；成熟历史重置通过真实 ResNet+合成数据的独立测试验证，不把合成历史冒充真实 CIFAR 历史。

每轮 checkpoint 保存 Full、完整 Kernel History、RNG、next_round 和源代码/配置/数据指纹；同 epochs/config/source 和完整日志边界才允许 `--resume`。epochs 纳入指纹，不能把2轮/100轮改成1000轮直接恢复。恢复历史不重复累计。

CUDA 正式实验保持原始非确定性设置；CPU 配对路径逐位测试通过，跨GPU进程不同结果不自动归因机制。并行耗时受同卡竞争影响，不能当隔离性能基准。

## 仍待检验

B3−B2 才是历史选择的核心比较；B2−B1 检查随机干预，B1−B0 检查分配。方向不稳定可能来自 Non-IID、优化阶段、噪声及层尺度，未证明应重置，更未证明有害性或语义冲突。保护 Full-only 不代表 Full-only 学习一定改善。1%/5次/3次/80% 是事前工程选择，不经官方测试搜索；单seed100轮只有探索价值，不能声称稳健泛化或论文创新已成立。不自动启动1000轮。
