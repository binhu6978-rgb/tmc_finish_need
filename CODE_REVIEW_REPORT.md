# AdapterFL 六方法与 Full-only 400轮探索实验报告

最新开发审查：基于 bd64c2c 的 SemConsFL S00–S09 结构作用域与功能响应实验。设计与公式见 [SEMCONS_SCOPE10_DESIGN](docs/SEMCONS_SCOPE10_DESIGN.md)，100项快速检查及十组两轮CPU CIFAR子集smoke证据见 [SEMCONS_SCOPE10_VALIDATION](docs/SEMCONS_SCOPE10_VALIDATION.md)。S00/S02权重及训练RNG一致；早期语义Gate中性情况已如实记录。此次没有启动或排队正式400轮S实验，等待独立代码审查。下文保留此前阶段记录。

当前阶段基于 `0e3498c8b076b5df8d7d1ca9449d423346c5e369`，仅增加可选 Full-only 评估、RNG 隔离、日志/峰值快照和固定十组队列。七模型、分区、采样、本地训练、原聚合以及六方法数学公式保持不变。当前验证结果与执行方式见第 12 节；第 1–11 节保留此前实现及审查证据，其中“本轮”等表述属于当时阶段。

## 交付范围与状态

开发分支：`feature/adapterfl-six-methods`。本轮审查基准为 `17aa580`，继续在同一分支上修正 M6 warmup、M4 曲率参数化及 M1/M5 诊断。**没有启动 300 轮训练**。服务器使用 `hh` 环境，第三轮独立 checkout 为 `/home/hh/TMC_re_adapterfl_review3_20261018_src`；原 `/home/hh/TMC_re` 及上一轮实验结果均保留，不中断已有进程、不操作卡 1 的其他任务。没有打包，不把登录凭据、数据、权重或私人材料加入 Git。第 9、10 节保留上一轮的历史诊断，第 11 节记录本轮实际结果。

AdapterFL 是本任务的方法名称；仓库原始模块实际拼写是 `AdaptiveFL`，入口仍为 `main_fed_ori.py` 和 `Algorithm/Training_AdaptiveFL.py::AdaptiveFL`。保留这些真实文件名，不引入另一个框架。AdapterFL-5 的未提交文件及已有实验产物保持原状，不混入本分支提交。

## 1. 真实源码分析

|环节|原始实现|本次处理|
|---|---|---|
|网络|`models/resnet.py::ResNet` / `ResidualBlock`|不修改|
|七模型生成|`models/Fed.py::get_model_list`|不修改|
|模型配置|width `[.4,.66,1]`、slim_idx `[2,3,4]`，width=1 只生成一个模型|原顺序完全保留|
|采样|原循环先 `np.random.choice(range(7),m)`，再 `select_clients`|不修改顺序、分布及有放回特征|
|下发|`copy.deepcopy(net_glob_list[feedback_model_idx]).to(args.device)`|保留；M2/M5 额外记录该槽位实际源状态|
|本地训练|`models/Update.py::LocalUpdate_AdaptiveFL.train`|仅 M4 可选近端项，其余执行原 SGD/CE|
|上传|客户端直接返回 `state_dict`，原循环按槽位保存|保留|
|聚合|`models/Fed.py::Aggregation_AdaptiveFL`|原函数完全不修改，所有模式继续调用|
|重组/恢复|聚合后循环调用 `split_model(w_glob_param, net.state_dict())`|不修改原调用次序；逐模型同步后立即测试|
|测试|`models/test.py::test` 内调用 `test_img`|named 模式直接调用相同 `test_img` 一次，同时取 Accuracy/CE 并保留同样的 Accuracy 打印|
|原输出|`result_ori_resnet18_test/` 的 log，`result_sensitivity/` 的时间/精度 txt|未指定 method 的旧路径不变；named 实验另存独立目录|
|原 checkpoint|原训练函数没有模型导出或服务器优化器|named 模式添加逐轮 latest checkpoint；只有 M6 使用服务器速度历史|

### 网络与映射事实

这是项目自定义的 CIFAR ResNet，不是 torchvision ResNet。rate 的五段对应 stem 与四个残差 stage，基准通道 `[64,96,128,256,512]`，索引 `< slim_idx` 的 rate=1，其余 rate=scale。每个 stage 两个原残差 block，后面三个 stage 首 block stride=2，shortcut 由实际输入/输出通道及 stride 决定。分类器输出始终 100 类。

七档依次为 P0(.4,2)、P1(.4,3)、P2(.4,4)、P3(.66,2)、P4(.66,3)、P5(.66,4)、P6(1,2)。BN 在这七档中为 `track_running_stats=False`：只有可训练 affine 参数，没有 running_mean、running_var、num_batches_tracked。未修改 BN、shortcut、分类器或任何模型定义。

原 `split_model` 按同名张量取前缀：卷积/线性权重前两个维度；BN 和 bias 第一个维度。卷积空间核形状保持不变。原聚合在同一前缀位置按样本数累计分子与分母，对有覆盖坐标归一化，无覆盖位置保留旧 Full。客户端若在一轮被选中多次，原代码将其视为独立训练槽位；本次保留这种重复计数含义，不擅自去重。

原各档在初始化时独立随机构建，并非第一轮开始前就从 Full 同步。M2/M5 所以必须使用各槽位**实际下发快照**计算本地增量，尤其不能在首轮减去 Full 的前缀。M1/M3/M6 按设计作用于 `W_aggregated - W_previous_Full`，首轮该差值也包含初始独立模型的融合差异。本次不增加额外同步；仅 M6 默认将第一轮原聚合结果直接保留，并排除其历史动量，见第 11 节。

### 设计假设差异与兼容定义

源码中没有一对固定、命名的“原始两分区”，只有因 slim_idx/profile 而异的宽度边界及逐坐标前缀映射。**不能将以下兼容区域称为原生已有的两个网络分区。**

本次在不改变任何映射/拓扑的前提下，按真实覆盖定义两个互斥且完备的坐标区域：

- `shared`：全部七档同名张量支持的交集，由七份实际 state_dict 各维最小尺寸计算，合计 2,120,985 个参数。
- `remainder`：Full 中的补集，合计 9,303,371 个参数，包括部分共享与 Full-only 坐标。

M3/M5 的“分区”专指以上覆盖区域；M4 约束 shared 可训练参数，而非任意全模型 FedProx。分类器所有类别输出不变，其共同输入前缀属于 shared；BN affine 也按真实前缀归属，统计缓冲区不参与正则。边界没有重复更新。如果研究意图限定为另一对论文中特定的分区，需要先明确其到本仓库键/坐标的对应关系，再评审调整，不能把当前版本的结果冒充那个定义。

## 2. 模式和开关

`--method` 为单选：`baseline,m1,m2,m3,m4,m5,m6`，不支持组合。

- **不指定 `--method`**：默认 None，完全走原来的采样、本地训练、聚合、测试和旧输出路径，不创建方法状态，不进行新随机操作。
- **`--method baseline`**：原计算加结构化记录与 checkpoint，不启用任何六种方法。与冻结原版逐参数和 RNG 对照通过。
- 其他模式：只启用对应策略。M4 是唯一改变本地 loss 的模式；M2/M5 不改变本地训练；M1/M3/M6 仅在原聚合后处理服务器增量。

所有新计算不抽取随机数。未启用的近端项不保存目标快照、不计算额外梯度。旧默认模式与 named baseline 的差别只在 I/O 和记录，不能把新增记录开销当作算法计算收益。

## 3. 六种方法：公式、实现及默认值

记旧 Full 为 W，原函数聚合结果为 A，实际下发状态为 S_i，上传状态为 U_i，客户端样本数为 n_i；所有切片遵循原同名坐标映射。

### M1：覆盖感知服务器步长

调用路径：原训练循环 → `MethodHooks.aggregate` 的 m1 分支 → 原 `Aggregation_AdaptiveFL` → 服务器增量缩放。

每个张量 k 统计槽位覆盖次数 c_k(x)，不再次用样本数给覆盖计数加权；r_k 为该张量有覆盖坐标的平均槽位覆盖次数。

```text
delta = A - W
eta(x) = clip((r_k / (c_k(x) + 1e-12))^gamma, eta_min, eta_max)
new(x) = W(x) + eta(x)*delta(x)      if c_k(x)>0
new(x) = W(x)                       otherwise
```

默认 `--coverage-gamma .5 --coverage-eta-min .8 --coverage-eta-max 1.2`。原函数已经解决“只覆盖部分坐标时分母不能用总样本数”的问题；本方法额外改变 update strength，不重复做样本归一化，也不声称原算法没有覆盖处理。gamma=0 时精确返回原聚合值。记录每张量 coverage 直方图、参考量、有效比例、eta 最小/均值/最大及直方图。

### M2：实际下发增量的方向冲突

调用路径：原循环保存每槽 S_i → `MethodHooks.aggregate` → `conflict` → 原聚合（有效样本权重）。

在本轮上传客户端的共同支持交集，按同名参数键和正确坐标计算 d_i=U_i-S_i。不同形状先取共同支持，再按键累加 dot/norm，绝不拼接不对应的网络位置。

```text
reference_i = sum_{j!=i}(n_j*d_j) / sum_{j!=i}(n_j)
cos_i = cosine(d_i, reference_i)
g_i = exp(-lambda * max(0,-cos_i))
new = original_aggregate(U_i, effective_lens=n_i*g_i, W)
```

默认 `--conflict-lambda 1`，reference 使用留一法，避免自身范数主导。单槽、零增量/零 reference 范数退回权重 1；全部权重相同时仍是原聚合，不保证只要检测到冲突就必然改变结果。当前七档共享的 stem/坐标非空；不支持未知且键不匹配的模型，直接报错而非错误拼接。lambda=0 精确退回原聚合。

为浮点稳定，对 log-weight 减去共同最大值（不改变归一化后的数学权重）；极端 log 差异小于 -80 时作数值下限，避免 float32 的全零/独占区域分母下溢。默认 lambda=1 不触及该下限。记录 cosine、负向比例、有效冲突权重、原始 log-weight、共同参数量和退回槽位。权重最终仍由原函数按每个坐标有效样本权重重新归一化；不另换增量聚合公式。

### M3：固定双区域服务器步长

调用路径：`MethodHooks.aggregate` m3 分支；先原聚合，再对互斥覆盖 mask 分别更新。

```text
new_region = W_region + alpha_region*(A_region-W_region)
```

默认 `--partition-alpha-shared 1 --partition-alpha-remainder .9`。这里不是论文原始的两个模型重组分区，而是前述实际覆盖 shared/remainder 区域。只作用于本轮实际覆盖坐标，无覆盖位置不动；同一坐标恰好归属一个区域。当前只实现固定系数，**没有历史自适应系数**，也不根据官方测试集调 alpha。记录两区域的原始/实际应用更新 L2、相对上轮更新范数波动和实际 alpha；历史只用于诊断，两个标量随 checkpoint 保存。两 alpha=1 时精确返回原聚合值。

### M4：直接控制共同覆盖参数的近端曲率

调用路径：`configure_local` 仅给 M4 设置属性 → 原 `LocalUpdate_AdaptiveFL.train` 的可选 loss hook → `prepare_prox` / `prox_loss`。

```text
R_sum = 0.5 * sum_{shared named_parameter coords} (theta - actual_initial)^2
L = original_CE + mu*R_sum
grad_prox = mu*(theta_shared - actual_initial_shared)
```

当前默认 `--shared-prox-mu .01`，不再除以 N_shared。mu 是同一参数坐标上的直接近端曲率；目标仍从**实际训练副本**进入本地 SGD 前取 detached clone，仅约束原 shared 集合，不对任何统计 buffer 正则化，不把初始参数接入反向图。原 optimizer、lr decay、CE、loader、drop_last 不变。

保留互斥的旧接口 `--shared-prox-lambda`（默认 None）。显式传入时使用旧 `lambda*R_sum/N_shared`，不再叠加 mu；`--shared-prox-lambda 0` 和 `--shared-prox-mu 0` 都完全关闭 hook，与冻结原版本地训练逐参数/RNG 一致。两种 flag 不能同时提供。日志明确区分 sum/mean、mu/lambda、实际目标参数量和等效曲率，避免混淆旧 Prox Loss 与新 Prox Loss 的数值单位。Task/Prox 梯度诊断及前两步采样保留。

mu=.01 在原 lr=.01 下对应无量纲 `lr*mu=1e-4`：仅考虑近端梯度、无 momentum 的单步位移收缩为 0.9999。原 momentum 会改变多步响应，因此这只是保守尺度依据，不是完整优化稳定性证明。该默认值在真实检查前固定，未按测试 Accuracy 选择；实际梯度比例见第 11 节。旧 lambda=1 在 N=2,120,985 下极弱的实测留档见第 9 节。

### M5：相同支持集合内的分区增量裁剪

调用路径：原循环记录 S_i → `norm_control` → 原聚合（仍使用原 n_i）。

shared 区域在所有 profile 上支持一致，可以共同估计阈值；remainder 区域只在具有**完全相同参数支持/shape 签名**的槽位间估计阈值，避免把不同宽度范数直接混比。

```text
norm_i = ||(U_i-S_i)_region||_2
threshold = multiplier * median(norms in equal-support group)
factor_i = min(1, threshold/(norm_i+1e-12))
U'_i_region = S_i_region + factor_i*(U_i-S_i)_region
new = original_aggregate(U'_i, n_i, W)
```

默认 `--partition-norm-multiplier 2 --partition-norm-min-clients 3`。证据少于 3 槽时不裁剪，零范数不缩放，正常更新保持原 tensor 值，不因重构浮点数引入假变化。两区域不重叠，无覆盖坐标仍由原聚合保留。记录每组槽位、before/after 范数、阈值、factor、裁剪比例和证据不足退回。可能出现很多 remainder 分组证据不足，这是保守实现的限制，不自动扩大比较支持或降低阈值。

### M6：排除初始化融合的 warmup 与连续覆盖历史

调用路径：原聚合 → `MethodHooks.aggregate` m6 分支；原服务器没有 momentum，此方法不重复服务器已有优化器（原客户端 SGD momentum 不等同于服务器 momentum）。

```text
delta_t = A_t - W_previous
if t <= warmup_rounds:
    new = A_t                    # 原聚合结果逐值保留
    velocity = {}; active = {}  # 不将初始化融合增量送入后续历史
else:
    if covered_now and covered_previous_valid:
        v_t = beta*v_previous + (1-beta)*delta_t
    elif covered_now:             # 首个有效历史轮或覆盖间隔后重新参与
        v_t = delta_t
    else:
        v_t = 0
    new = W_previous + v_t        # 仅本轮覆盖坐标；无覆盖位置保持旧值
```

默认 `--history-beta .9 --history-warmup-rounds 1`。第一轮使用原聚合结果，velocity 和 active 均为空；第二轮用当前真实聚合增量初始化有效历史，应用结果仍与原聚合逐值一致；第三轮才开始对连续覆盖坐标平滑。未覆盖位置不变，覆盖中断清空相应历史，重新覆盖只使用当前增量。warmup=0 显式复现旧初始化策略，warmup=2 等更长配置也可保存/恢复，不按精度自动选择。

checkpoint 同时保存 `history_rounds_seen`、warmup 配置、有效 velocity、previous_active 与诊断阶段。warmup 期间不允许包含 velocity；旧污染历史不会静默迁移到新配置。首轮巨大 delta 的范数可以保留为标量诊断，但其张量不作为后续动量。beta=0 精确返回原聚合。保留原始/平滑更新范数、贡献量、覆盖和间隔后重新初始化诊断。

## 4. 修改清单

下列 8 个文件为首版实现清单。本轮相对 `17aa580` **修改 6 个文件**：

|文件|本次修正|
|---|---|
|`adapterfl_methods.py`|M6 可恢复 warmup；M4 sum/mean 参数化；M1 相对原聚合的逐层/全局实际差异 L2；M5 尾部比值与不触发原因|
|`models/Update.py`|M4 mu 对应未归一化 penalty/梯度，保留旧 lambda 和零强度路径|
|`utils/options.py`|互斥 mu/lambda 参数及 history-warmup-rounds，原训练默认参数不变|
|`tests/test_adapterfl_methods.py`|warmup 前后恢复、巨大初始化排除、覆盖间隔、mu 曲率及零强度等价等测试|
|`CODE_REVIEW_REPORT.md`|本轮数学依据、已有真实 M5 范数分析及集成证据|
|`tools/probe_adapterfl_m4.py`|配对探针跟随新 mu 默认配置，保留零正则同诊断控制|

本轮不修改 `main_fed_ori.py`、`Algorithm/Training_AdaptiveFL.py`、原聚合/切片函数、客户端采样、真实分区或七模型配置。

**修改 4 个现有文件：**

1. `main_fed_ori.py`：named 方法的独立输出目录与异常上报；保持既有入口、数据加载及 unused Full 初始化顺序。方法实验要求 `--generate_data 0` 且已有原分区，防止原 loader 在缺少记录时生成新分区。原默认模式仍保留原行为。
2. `utils/options.py`：新增互斥 method 单选、各方法参数、显式恢复选项，原参数默认值不动。
3. `Algorithm/Training_AdaptiveFL.py`：同一个原循环增加 hooks、实际下发快照、CE 记录与 checkpoint；原 sampler、下发、训练、逐档 split/test 顺序不动。
4. `models/Update.py`：仅原 `LocalUpdate_AdaptiveFL.train` 添加 M4 可选近端分支，其他训练类和旧 FD 方法不改。

**新增 4 个文件：**

5. `adapterfl_methods.py`：六个互斥策略的数学辅助函数与记录/恢复 hook；没有新模型、第二套训练循环、独立 server/client 或调度平台。
6. `tests/test_adapterfl_methods.py`：CPU 专项检查。
7. `tests/fixtures/adapterfl_original_training.py`：从修改前 c51c166 冻结的原训练函数和本地 train 方法，作为 golden 对照，避免只用修改后实现自证。
8. `CODE_REVIEW_REPORT.md`：本报告。

原 `models/Fed.py`（含聚合/切片/模型创建/采样）、`models/resnet.py`、`models/test.py`、`utils/get_datatset_new.py` 均未修改；开发前 SHA256 保存于本地 `runs/adapterfl_methods_checks/original_source_sha256.json`。私人材料与实验权重/数据不加入提交。

## 5. 输出与恢复

统一路径：`runs/adapterfl_methods/<method>/<run-name>/`，与原服务器结果目录和 AdapterFL-5 隔离；新实验拒绝覆盖已有同名目录。

|文件|含义|
|---|---|
|`console.log`|原 logging 输出；后台 stdout/stderr 仍可单独重定向|
|`manifest.json`|方法、七档配置、原训练参数、源码与实际分区 SHA256、环境与调用命令|
|`accuracy_each_round.csv`|Full 及全部七档每轮 Accuracy/CE；原最大客户端耗时与累计值；额外 round wall、聚合与评估耗时|
|`accuracy_summary.csv`|各档最终 Accuracy/CE、末最多 20 轮均值及总体标准差、实际轮数|
|`method_diagnostics.csv`|每轮 method 和结构化 metrics_json（层级覆盖、客户端正则、clip 分组、history 等）|
|`latest.pt`|已完成边界的 Full 权重、方法历史、参数/源码/分区身份、RNG、轮数和已记录结果|
|`completed.json`|只有实际完成训练循环后生成；只有确实完成 300 轮才标 formal_300_success|
|`failed.json`|named 模式异常 traceback，进程返回失败，不像旧入口静默 continue|

时间含义：原时间是每轮**最大单客户端时长**及其累加，绝非整轮串行墙钟，本次原样保留。另报 `round_wall_seconds`（不含该轮日志/checkpoint 写入）、`server_seconds`（聚合 hook）、`evaluation_seconds`。不把它们混称端到端成本。latest 每轮原子替换，需额外 I/O；本次未跑真实吞吐测量。

checkpoint 仅按最新/最终轮保存，不根据官方测试 peak 选模。恢复是显式用户操作，无自动重试：同一 epochs/config/source/原分区身份、日志与 checkpoint 轮数对齐后恢复；先恢复 Full 并按原 split 同步七档，再恢复 M6 velocity/覆盖 mask 与训练 RNG。断在“日志写完但 checkpoint 未写完”时拒绝不一致恢复，保留证据，不自动截断日志。不是通用训练恢复平台。

## 6. 验证证据

环境为本地 conda fd：Python 3.12.0 / PyTorch 2.7.1+cu128。所有本次专项计算均在 CPU，数据为合成张量；没有真实 CIFAR 正式训练或精度收益结论。

专项命令：

```bash
python -m pytest -q tests/test_adapterfl_methods.py
```

首版 43 项专项检查的证据为 `runs/adapterfl_methods_checks/pytest_final.log`，第二轮为 47 项。本轮六方法专项 **55 passed**，Linux hh 日志为新目录 `runs/third_review_checks/pytest_linux.log`（8.37 秒）；本地 fd 同样 55 项通过，并额外对现有 AdapterFL-5 作 6 项兼容回归，合计 **61 passed**，日志 `runs/adapterfl_methods_checks/third_review_all_local.log`。没有修改或提交五模型代码。

本轮增加 warmup=0/1/2 的状态恢复、默认第2轮巨大初始化排除与覆盖间隔、mu=0/旧lambda=0 冻结原版等价、直接曲率梯度、M1 实际差异范数和 M5 正常更新不强制裁剪。原来的 M6 无 warmup 代数测试显式设置 warmup=0，不以修改预期值掩盖旧策略。其余原检查包括：

- 冻结原循环 vs 新的默认旧路径 / named baseline：真实七档 ResNet、固定输入、同样采样、本地 SGD、重组与七次测试，最终所有参数 **bitwise equal**，Python/NumPy/Torch RNG 状态相同。
- 冻结原本地 train vs 新 train 在 M4=0 时 bitwise equal；未额外设置 proximal snapshot。
- M1 单次样本权重归一化及 coverage 缩放；M2 实际下发基准、留一参考、负向衰减、单槽/零范数/不同 shape/极端 lambda；M3 两区域互斥、更新系数与诊断；M4 detached 目标、正确梯度支持、非零正则实际改变更新；M5 相同支持比较、正常更新不变、异常更新裁剪；M6 不覆盖不漂移、间隔后重初始化。
- 六方法和 baseline 的参数有限/shape 合法、旧服务器输入不被原地污染、未覆盖尾部不变；所有方法状态导出/读取后后续聚合一致。
- 真实七档配置的 Conv/BN/shortcut/classifier 映射，以及六个 hook 的真实参数形状 CPU 聚合与 strict load。
- M6 在同一个原始训练循环中，三轮合成训练“不中断” vs “完成第一轮后中断、从 checkpoint 恢复”：最终七档权重与 RNG **bitwise equal**，CSV 轮数为 1/2/3 无重复。

首次专项检查的唯一失败来自测试替身的 profile tuple 漏填原代码要求的第三项参数量，补正测试 fixture 后通过，没有改训练算法来迁就测试。

### 首版扩展旧回归的历史限制：本地原生异常

额外检查了 `test_fd_semconsfl.py`、`test_hgkr.py`、`test_hgkr7.py`、`test_adapterfl5.py`，使用 `CUDA_VISIBLE_DEVICES=-1`，没有 GPU 工作负载。原日志 `regression_cpu.log` 在旧 HGKR batch50 CPU 训练的 BatchNorm 中出现 Windows `access violation`（退出码 -1073741819）。限制线程为 2 后，仍在旧训练 backward 路径发生同类异常，日志 `regression_cpu_threads2.log`。

单独隔离该旧测试，当前源码一次干净通过，日志 `isolated_current.log`；将本地 train 替换回本次冻结的原函数再隔离，也出现原生访问异常日志（虽随后 pytest 打印 1 passed，不能视作干净通过），见 `isolated_frozen.log`。因此不能仅凭这个现象把问题归因于新方法，根因尚未确定。

最后仅在测试进程关闭 MKLDNN、限制 2 线程核验，旧回归推进到 66% 后仍在旧七档 HGKR 的 CPU 卷积中遇到访问异常，见 `regression_cpu_native_backend.log`。没有将这些后端设置写入项目或正式训练，也没有修改旧测试以隐藏问题；停止继续尝试后端排列。

**首版结论：43 项专项检查通过，扩展旧回归未完整通过，不能声明全项目回归通过。** 第二轮完成目标 Linux 的 47 项专项检查与七模式各 3 轮真实集成，见第 10 节；第三轮的 55 项专项及五模式三轮检查见第 11 节。这不等于此前旧 FD/HGKR 全套回归已重新完整通过。没有用正式训练来代替检查，没有精度收益结论。

## 7. 后续正式命令（仅供审查，不执行）

请在**独立服务器 checkout** 切换到审查并批准后的分支，不在正在运行的旧实验目录中替换公共训练文件。保留原数据路径 `/home/hh/ICML_finish/data` 和项目内原始分区。服务器沿用已经可以跑原版的环境；本地用 conda fd。训练依赖没有增加。

每个 method 分别执行，以下以 m1 为例；baseline/m2…m6 只替换 method 和 stdout 文件名。七档、100 客户端、10 槽位、5 local epoch、batch50、原 SGD 超参数和逐轮七档测试保持不变：

```bash
nohup python -u main_fed_ori.py \
  --method m1 --method-run-name seed1_300_reviewed \
  --epochs 300 --seed 1 --gpu 0 \
  --generate_data 0 --data_beta 0.3 \
  > m1_seed1_300.stdout.log 2>&1 < /dev/null &
```

不要同时启动全部七组；此命令不包含后台调度器，也不授权现在启动。独立 baseline 需显式 `--method baseline`。没有 `--method` 的命令仍执行原版及其旧输出路径。

显式恢复示例（原 epochs、方法参数与源码不得改；仅展示接口）：

```bash
python -u main_fed_ori.py --method m6 --epochs 300 --seed 1 --gpu 0 \
  --generate_data 0 --data_beta .3 \
  --method-resume runs/adapterfl_methods/m6/seed1_300_reviewed/latest.pt
```

## 8. 科学与使用限制

1. 六方法是独立待验证的候选机制，不是已证实改善 CIFAR Accuracy 的算法；没有调参、试种子、测试峰值选模或新增损失以外的组合模块。
2. M3/M4/M5 的目标是本报告明确的实际覆盖分区，不是原代码中不存在的固定命名“两分区”。这个兼容定义应在正式实验前重点审查。
3. M2 使用所有当前槽位共同支持上的一个客户端权重，可能受大张量方向主导；没有另加分层权重或额外模块。M5 对 remainder 的同支持组可能频繁证据不足而不裁剪，日志会如实显示。
4. 旧 M4 mean-normalized lambda=1 很弱、旧 M6 首轮初始化融合污染历史的问题已有实测。本轮默认改为直接 mu 和 warmup=1；旧行为只能显式选择对应兼容参数复现，未按测试精度调参。
5. 只支持本仓库经核验的 float32、BN 无统计缓冲区的七档 ResNet；未知标量 buffer 或其他网络不做猜测性兼容。原聚合对标量 BN buffer 的潜在问题没有在 baseline 中修补。
6. 原最大客户端时间指标不能替代真实墙钟。M2/M5 下发快照、M6 历史状态和逐轮 checkpoint 会增加内存/时间/I/O，尚未测 GPU 吞吐。M6 持久化速度和 bool 覆盖 mask 不会更新无本轮覆盖的参数。
7. 共享参数代码只在新分支更改；不要在服务器运行目录原地换文件。旧 FD/HGKR 的已有 checkpoint 指纹含源码/参数，本分支不得用于恢复旧实验，旧实验继续使用其原 checkout。
8. 现有默认入口捕获异常后静默 continue、`get_final_acc` 会基于测试序列挑一个显示轮次：原旧路径保留；named 模式改为失败明确上报、最终轮汇总，不把该显示规则当作验证集选择。

## 9. 上一轮 17aa580 诊断留档（不是当前默认配置）

### 9.1 M1/M2/M3：原公式和作用范围

M1 数学公式未变。c 是**训练槽位覆盖次数**，原聚合的分母是该坐标覆盖槽位的**样本数之和**；二者不能混为一谈。本方法是原聚合之后的服务器更新步长控制，不替代也不重复原覆盖归一化。

M2 留一法公式未变。新增共同支持/Full 的比例、共同支持/各上传的比例、有效 cosine 的负向比例、冲突权重 min/median/mean/max/std、单槽独占坐标数及其归属。对一个独占坐标 x，只有槽位 i 覆盖时，原聚合为 `(n_i*g_i*U_i(x))/(n_i*g_i)=U_i(x)`。因此**独占区域上的冲突权重会约消**，这是保留原逐坐标归一化后的正确语义，不能声称该区域也受到冲突衰减。共享区域仍按有效权重归一化。新增合成测试特意让 Full 槽位发生负冲突，验证共享部分改变而独占尾部维持上传值（浮点容差）。没有额外加入独占区域 damping。

M3 的准确名称为**固定双区域服务器步长**：shared/remainder 是实际覆盖区域，不是论文的原始两个模型重组分区。历史范数只作诊断，当前没有历史自适应系数，也没有改变边界。

### 9.2 M4：默认正则确实极弱，保持 lambda=1

目标参数数 N=2,120,985。当前 mean-normalized 正则的梯度与曲率为：

```text
R = ||theta_shared - theta_initial||² / (2*N)
grad(lambda*R) = (lambda/N)*(theta_shared-theta_initial)
effective unnormalized proximal curvature = lambda/N
```

默认 lambda=1 的等效未归一化系数为约 `4.71479e-7`。在首个 SGD 步后，若位移主要来自 lr*Task 梯度，则下一步的近端/Task 梯度比例约为 `lambda*lr/N`（还会受当前 Task 方向、momentum、weight decay 影响）。lr=.01 对应约 `4.71479e-9`，而不是与 Task 梯度同量级的约束。

新增 `prox_gradient_diagnostics`：每个 M4 客户端采样前两个 SGD 步，直接用 `autograd.grad` 取得 Task 梯度、用解析式计算近端梯度；只读诊断，不累积 `.grad`、不增加 forward、随机抽样或 optimizer step。**额外最多两次 Task backward/客户端**会增加诊断开销，已计入日志，不能把它当成纯算法成本。记录 Task CE、Prox Loss、加权与未加权损失比、全参数/目标共享 Task 梯度范数、加权与未加权近端梯度、二者比值。非零采样梯度始终低于 Task 的 1e-4 时记录弱约束警告；该阈值仅用于诊断，不改变更新。

Linux CPU 的配对合成探针使用真实原七档模型构建、Full、相同初始权重与 minibatch RNG、3 个本地 SGD 步、batch2、32×32 图像，不是 CIFAR 性能实验，也未以探针调整 lambda：

|本地步|Task CE|Prox Loss|lambda*Prox / Task|Task 全参数梯度 L2|Task shared 梯度 L2|lambda*Prox 梯度 L2|近端/shared Task 梯度比|
|---|---:|---:|---:|---:|---:|---:|---:|
|1|4.570661|0|0|50.4403|37.8542|0|0|
|2|4.641078|3.37801e-8|7.27850e-9|50.3529|37.5859|1.78475e-7|4.74844e-9|
|3|5.143319|1.08794e-7|2.11524e-8|52.1738|38.7503|3.20293e-7|8.26557e-9|

三步 baseline 更新范数为 1.264611；M4 与 baseline 最终参数差异 L2 为 **2.115714e-6**，相对 baseline 更新为 **1.673016e-6**。零正则、同诊断图的控制与 baseline 参数差异为 **0**；M4 与该控制仍差 2.115714e-6。该控制支持“代码有很小但非零的参数作用”，不能据此宣称有实际精度收益。近端直接梯度只作用于 shared，但后续 CE 可以将先前 shared 变化间接传到其他参数；不能以最终变化坐标数冒充正则目标数量。

**有量纲依据的后续方案，未自动应用：**如果研究意图是未归一化近端曲率 mu，等价参数化应为 `lambda_mean=N*mu`，而不是随意乘固定倍数。也可先明确一个训练梯度目标比例 rho，在预先规定的训练数据探针位置计算 `lambda_rho=rho*||g_task_shared||/||g_R||`。以上合成点若仅用 rho=1% 作尺度参照，第 2/3 步分别对应约 2,105,954 和 1,209,839；**这些不是可直接推荐到正式训练的最佳值**，更不是依据测试集调出来的参数。本轮 lambda 始终是 1，没有引入自动标定或新超参数搜索。

### 9.3 M5：不生效也必须可见

裁剪公式、multiplier=2、min_clients=3 未变。新增：每个支持签名的槽位数、有效区域大小、阈值触发与真正发生 float32 上传变化的槽位数、上传变化 L2、按区域实际裁剪率、因槽位不足回退的数量/比例。空 remainder 不进入非空区域分母。

逐轮完全无实际裁剪时，结构化日志和控制台均写 `NO_ACTUAL_CLIPPING_THIS_ROUND`；连续至少 3 轮再写 `NO_ACTUAL_CLIPPING_FOR_AT_LEAST_3_ROUNDS`。连续未生效轮数及累计生效轮数随 checkpoint 保存/恢复，不改变阈值，不自动降低最少槽位数。新增合成 outlier 检查证明裁剪可触发、正常范围不变；真实三轮未触发也如实报告。

### 9.4 M6：初始化量与历史贡献可分辨

公式、beta=.9、velocity 及覆盖 mask 不变。新增首个聚合增量范数、首次/间隔后重新初始化坐标、连续覆盖坐标数、历史分量 `beta*v_previous` 范数、当前分量范数、二者比例、应用更新/原聚合增量比例，以及与原聚合结果的差异范数。诊断状态随 checkpoint 保留。

原七档初始化独立，因此首轮 delta 不是纯本地学习位移，还包含独立模型初始化融合。该事实不通过额外同步或修改 warmup 隐藏。

## 10. 上一轮 17aa580 的 Linux 三轮集成留档

### 环境、执行与配对

- 独立目录：`/home/hh/TMC_re_adapterfl_review_20261018`，从 GitHub 的现有开发分支克隆，再应用待提交修正；实际测试的全部训练源码 SHA256 与本地提交源码逐项匹配。
- 系统：Ubuntu 24.04.3 LTS；环境 `hh`，Python 3.12.0、torch 2.7.1+cu118、torchvision 0.22.1+cu118；GPU0 RTX4090，24,564 MiB。卡1已有任务，未使用或中断。
- `hh` 原来缺 pytest，仅安装 pytest 及其纯 Python 测试依赖；未安装/升级 torch、torchvision 或 CUDA。
- 本地 fd 专项 **47 passed**；Linux hh、GPU 隐藏的 CPU 专项 **47 passed**（8.90 秒）。含无方法/named baseline 的冻结原版权重/RNG一致性、各方法独立性、真实七档映射、未覆盖参数保护和 M6 恢复一致性。
- 真实数据为 `/home/hh/ICML_finish/data`。实际已有分区 SHA256 仍为 `41ab2d3486663ce0100e183ab00565de137f91af07632c236c2ddf60d890ba41`。
- baseline、m1～m6 **各真实完成 3 轮**，顺序使用 GPU0。每轮仍是原 100 客户端、10 槽位、5 local epoch、batch50、SGD/CE、原学习率与衰减、原全部七档测试；只缩短轮数用于代码检查。
- 所有模式的七档初始权重哈希、实际三轮客户端/模型槽位、实际分区与训练源码均一致；CSV、diagnostics、checkpoint next_round 均为 3。global 权重有限，所有 `formal_300_success=false`。没有超时推断完成、重试、调参、换 seed 或追加长训。

实际每组命令（method 依次为 baseline m1 m2 m3 m4 m5 m6）：

```bash
/home/hh/miniconda3/envs/hh/bin/python -u main_fed_ori.py \
  --method "$method" --method-run-name linux3_seed1 \
  --epochs 3 --seed 1 --gpu 0 --generate_data 0 --data_beta 0.3
```

执行脚本仅是该次检查的简单顺序 Shell，位于 `runs/review_revision_checks/run_linux_checks.sh`，任一失败即停，不是新增训练框架。

### 真实机制作用

|模式|实际观察|解释|
|---|---|---|
|M1|三轮覆盖参数中非中性 eta 占比 95.49%、92.30%、97.26%|更新步长控制确实启用，不是替换原归一化|
|M2|共同支持占 Full 18.57%、23.16%、20.19%；负冲突比例 30%、0、0；首轮权重 min=.986489、mean=.997254，后两轮全1|首轮有很小的冲突干预，后两轮中性；第3轮有 4,478,775 个独占坐标，不能宣称它们也被衰减|
|M3|remainder 原/应用更新 L2：37.0373/33.3336、.698134/.628320、.762998/.686699；shared alpha=1|固定 .9 的 remainder 步长确实产生差异；不是历史自适应|
|M4|30 个非零梯度采样的近端/shared Task 比为 4.60e-9～5.35e-9；30 个客户端均有弱约束警告|数学项存在且探针可测出参数差异，但默认约束在实际尺度上几乎不起作用|
|M5|三轮实际裁剪率均0；shared 每轮10槽有足够证据；remainder 非空组回退比例100%、70%、60%|实际这批没有算法裁剪作用；不因 Accuracy 不同就声称 M5 生效。逐轮及连续不触发警告已触发并保存|
|M6|首轮 delta L2=44.4507；第2/3轮 raw delta=.811491/.991575，应用更新=40.0095/36.0083；历史/当前分量比73.88/363.14|首轮初始化融合量显著支配后续 velocity，应在下一轮审查确认研究意图；本轮不改公式|

M5 remainder 各轮支持分组槽位数分别为 `[2,2,1,2]`、`[3,2,2,1,2]`、`[4,1,2,1,1,1]`；shared 均为 `[10]`。前述 fallback 分母只计非空区域槽位，分别为7、10、10。其余满足最少槽位的组也未出现超过中位数两倍的异常更新，故无裁剪不是分组代码失效的证据。

M4 实际客户端的平均 Task CE 范围为 2.77965～3.49761，平均 Prox Loss 为 9.16630e-8～2.57670e-7，平均加权 Prox/Task 比为 3.52716e-8～9.44195e-8。梯度采样发生在每个客户端的前两个 SGD 步，不能将其写成整个本地训练的完整梯度统计。

### 精度仅作有限性检查，不能比较性能

|模式|第1/2/3轮 Full Accuracy (%)|第3轮 Full CE|
|---|---|---:|
|baseline|1.00 / 2.55 / 5.02|4.63542|
|M1|0.98 / 2.82 / 5.00|4.68133|
|M2|1.02 / 3.02 / 5.10|4.63757|
|M3|1.09 / 2.48 / 4.72|4.59306|
|M4|1.00 / 3.01 / 5.16|4.63633|
|M5|1.05 / 2.72 / 4.92|4.64005|
|M6|1.03 / 0.99 / 1.12|5.26036|

不同 GPU 进程保留原 CUDA 非确定性，不能把 M5 完全没有裁剪时的 baseline 精度差异解释为方法作用，也不能把 M4 的 5.16% 解释为极弱正则带来的可信收益。这是三轮实现检查，不是正式性能实验、不作排名、不用官方测试结果选 lambda、beta、裁剪阈值或 checkpoint。

### 保留的结果与结论

服务器各组：`runs/adapterfl_methods/{baseline,m1,m2,m3,m4,m5,m6}/linux3_seed1/`，均含独立 CSV、console log、manifest、latest.pt 和 completed.json；baseline/M1～M5 checkpoint 约45.7MB，M6约102.9MB。未覆盖区域保护与历史恢复由专项测试核验，不以精度变化替代这些检查。

汇总：`runs/review_revision_checks/integration_summary.json`；Linux 单测：同目录 `pytest_linux.log`；原 M4 探针：`m4_default_probe/`；含零正则控制的最终探针：`m4_control_probe/`；stdout 分别为 `{method}_linux3.stdout.log`。所有服务器日志与 checkpoint 留存，未覆盖原实验。

本地已保存证据副本：`runs/adapterfl_methods_checks/review_revision_linux/{integration_summary_linux.json,pytest_linux.log,m4_control_probe_linux.json}`，并核对汇总中的所有训练源文件 SHA256 与本地文件相同。权重留在服务器，不加入 Git。

结论：本轮修正与轻量集成已完成。M1/M2/M3 有对应机制作用；M4 默认梯度强度极弱；M5 当前三轮没有实际裁剪且已有明确警告；M6 有强历史作用但首轮初始化融合支配明显。这些观察不等于任何方法已取得科学性能收益。**不启动 300 轮，等待下一次代码/研究设定审查。**

## 11. 第三轮修订：数学依据、验证和真实三轮结果

本节是当前配置的结论；第 9、10 节为修改前 `17aa580` 的证据留档。没有以任何一轮官方测试 Accuracy 选择配置。

### 11.1 M6：初始化 warmup 与可恢复时序

默认 warmup=1，第一轮精确返回原聚合结果，同时不建立 velocity 或 previous_active。第2轮所有当前覆盖坐标以本轮 delta 建立有效历史，仍直接保留原聚合结果；第3轮才开始 beta=.9 的原 EMA。覆盖中断后 velocity 清零，重新参与时直接取当前 delta。`history_rounds_seen` 和配置、velocity、mask 一起保存；warmup 期间 checkpoint 不允许携带有效动量。首轮范数只作标量留档，不参与公式。

专项测试用 100 倍的大首轮增量验证默认第2轮只使用新的单位增量。另对 warmup=0/1/2、warmup 完成后的 checkpoint、首个有效增量之后的 checkpoint、未覆盖/重新覆盖作恢复检查。原训练循环的“第一轮保存后中断→恢复”与不中断轨迹，最终七档权重/RNG 仍逐值一致。

### 11.2 M4：预先固定的直接 mu

默认 mu=.01，以直接曲率而非参数均值的系数控制约束；`grad_prox=mu*drift`，不再随约212万个目标参数被除小。原 lr=.01 给出 `lr*mu=1e-4`，作为保守的单步位移尺度。此值在真实检查前固定，真实结果仅作量级核验，没有反过来调整 mu。

旧 lambda 接口显式选用时仍对应 `lambda/N`，两个接口互斥，均不叠加；mu=0 和 lambda=0 都不取快照、不计算附加项，恢复原训练。实际下发 detached 快照和 shared 参数集合原样保留。日志的 `prox_normalization=sum`、`mu_prox=.01` 明确标识新量纲。

配对 CPU 合成代码探针（同一原始 Full、3 个本地步、batch2、相同初始状态/RNG）的第2/3步加权近端/shared Task 梯度比为 **1.00714e-4 / 1.75391e-4**；M4 与 baseline 的参数差异 L2=**.0117858**，相对 baseline 更新范数约 **.93197%**，零正则同诊断控制与 baseline 差异仍为0。这证明新参数化有可观测作用；多步网络非线性与浮点变化不受单步梯度比例的简单上界约束，不能把该探针当作精度收益。

### 11.3 M5：为何不改成“保证触发”的阈值

重新读取上一轮真实 `norms_before`，不使用测试分数：

|区域/轮次|同支持槽位数|中位数 L2|最大 L2|max/median|原阈值|
|---|---:|---:|---:|---:|---:|
|shared/1|10|1.29801|1.50682|1.16088|2.59601|
|shared/2|10|1.05758|1.24302|1.17535|2.11516|
|shared/3|10|1.01301|1.20516|1.18968|2.02603|
|remainder/2 合格组|3|1.01585|1.07852|1.06169|2.03170|
|remainder/3 合格组|4|1.00450|1.05011|1.04541|2.00901|

其余非空 remainder 组不足3槽；空区域单独排除。因此未触发来自两种可解释情形：证据不足，或正常更新远未超过固定阈值。现有 median×2 本身是预先固定的稳健规则，当前范数没有支持降低阈值的异常证据，本轮**不更改 multiplier=2、min_clients=3**，也不为制造触发率加入另一套阈值候选。

新增每组 median/max、max/median、nonactivation_reason（空区域、槽位不足、阈值内）。保留真实上传变化、裁剪率、连续不生效警告和 checkpoint 计数。人工异常 `[1,1,100]` 单测会裁剪并改变聚合；正常 `[1,1.1,1.2]` 不裁剪且逐值返回原聚合，分别验证预期条件下的有效性与正常更新保护。

### 11.4 M1/M2/M3

M1 公式未变，新增每层及 `server.__summary__.difference_from_native_aggregate_l2`，直接度量 `||W_M1 - W_native_aggregate||2`，不以 eta 活跃比例替代实际参数差异。三轮实测分别为 **6.92534 / .135756 / .157105**。

M2 的留一法、共同支持/独占区域和冲突权重公式/诊断均未改。M3 仍称**固定双区域服务器步长**，shared/remainder 不是论文原始两重组分区，没有历史自适应系数。两者仅运行全套专项回归，没有追加真实训练。

### 11.5 Linux 真实检查与产物

目录：`/home/hh/TMC_re_adapterfl_review3_20261018_src`；使用 `hh`，torch2.7.1+cu118 / torchvision0.22.1+cu118，RTX4090 GPU0。原 TMC_re 和上一轮结果均不覆盖。远端 GitHub 拉取曾遇 TLS 断开，改从服务器上已核验的 `17aa580` Git checkout 克隆源码（不复制旧 runs、不给 Git 对象创建硬链接），再应用本轮修正，origin 仍指向同一 GitHub 仓库。

先完成 Linux 55 项 CPU 专项，再顺序执行 baseline、m1、m4、m5、m6 **各3轮**真实 CIFAR-100；只缩短轮数，其余七档、100客户端、10槽、5 local epoch、batch50、SGD/CE、学习率、分区和全部七档逐轮测试不变。每组实际命令：

```bash
/home/hh/miniconda3/envs/hh/bin/python -u main_fed_ori.py \
  --method "$method" --method-run-name linux3_seed1 --epochs 3 \
  --seed 1 --gpu 0 --generate_data 0 --data_beta 0.3
```

默认参数已在 manifest 中核对：mu=.01、旧lambda=None、warmup=1。五组七档初始化哈希、实际三轮客户端/模型槽位、分区和训练源码均一致。CSV/diagnostics/checkpoint 均记录3轮，Full 权重有限，`formal_300_success=false`；M6 checkpoint 的 history_rounds_seen=3。所有测试训练源文件 SHA256 与本地待提交代码逐项一致。

M6 的实际结果：

|轮次|阶段|原始 delta L2|实际更新 L2|历史分量 L2|相对原聚合结果差异 L2|
|---|---|---:|---:|---:|---:|
|1|warmup，未建历史|44.44994|44.44994|0|0|
|2|首个有效历史|.808814|.808814|0|0|
|3|连续覆盖 EMA|.870758|.743278|.727933|1.00454|

第2轮不再继承初始化增量；第3轮历史/当前分量比约8.36，来自正常 beta=.9 的有效历史，与上一轮约40的污染历史不同。这证明初始化问题已按时序修正，不证明 EMA 改善精度。

M4 真数据的30个非零梯度采样（各客户端第2个 SGD 步）近端/shared Task 梯度比 **9.75532e-5～1.14355e-4**。客户端平均 Task CE 为2.76250～3.50706，未加权 `R_sum` 为.195910～.551015，加权 penalty 为.00195910～.00551015，加权 penalty/Task 比为 **.00075436～.00201327**。这些是前两步梯度采样及全本地步的损失均值，不冒充全程梯度统计。

M5 三轮裁剪率仍为0，逐轮与连续3轮不触发警告完整；这符合本批没有超阈值异常的范数数据，不能把不同进程的精度差异归因于 M5。

Full 测试仅作有限性/轨迹检查：

|模式|第1/2/3轮 Accuracy (%)|第3轮 CE|
|---|---|---:|
|baseline|1.07 / 2.48 / 5.09|4.62479|
|M1|1.01 / 3.34 / 4.70|4.69593|
|M4|1.04 / 2.60 / 5.20|4.63435|
|M5|1.05 / 2.63 / 5.15|4.62642|
|M6|1.04 / 2.82 / 3.96|5.04279|

不按三轮精度排名或调参；保留原 CUDA 非确定性，尤其不能将 M5 零裁剪时的差异解释为作用，也不能把 M4 5.20% 解释为可靠收益。M6 修复的是历史初始化污染，其三轮精度并不足以评价长期性能。

结果路径（均相对于新 checkout）：

- `runs/adapterfl_methods/{baseline,m1,m4,m5,m6}/linux3_seed1/`：独立 manifest、三个 CSV、console.log、latest.pt、completed.json。
- `runs/third_review_checks/`：Linux 单测、各组 stdout、`integration_summary.json`、`m5_prior_norm_analysis.json`、`m4_mu_probe/` 和本次顺序执行脚本/完成状态。
- 本地证据副本：`runs/adapterfl_methods_checks/third_review_linux/`，含汇总、Linux单测、mu探针与原 M5 范数分析；权重保留服务器，不上传 Git。

本轮结论：M6 warmup/恢复和覆盖中断处理通过；M4 直接 mu 在真实数据上有预期量级的非零梯度作用；M5 在异常单测下生效，对本批正常范数保持不裁剪并明确说明原因；M1 实际参数差异已度量，M2/M3 算法不变。**停止于代码与最多三轮检查，等待最终审查，不启动300轮。**

## 12. Full-only 评估与固定十组400轮探索（2026-10-19）

### 12.1 评估行为及随机数边界

`--eval-profiles all` 为默认，继续原来的七次测试顺序和训练 RNG 消耗；旧路径和 named baseline 都通过冻结原版的逐参数/RNG 对照。`--eval-profiles full` 先按原 `split_model` 同步全部七档，然后只对 `net_glob_list[-1]` 调用一次原 `test_img`。`models/test.py`、测试集、batch size、eval/no_grad 均未改。没有 method 的显式新评估选项会选择 named baseline，以便保存完整日志；没有任何新选项的旧入口保持原路径。

Full-only 隐式隔离评估的 Python、NumPy、Torch CPU 和已初始化 CUDA RNG，finally 恢复，即使评估抛出异常也恢复。新 all 模式可显式添加 `--isolate-eval-rng` 使用同一隔离规则，供机制核验。DataLoader 的迭代器即使 shuffle=False 仍可抽取 Torch base seed，因此新 full/isolated-all 与历史 legacy-all 的后续训练 RNG 轨迹不同；**不能称为与旧300轮日志 bitwise identical**。十组新实验包含新的 B00，全部统一使用 Full-only 的隔离规则；400轮预算也不同于历史300轮。

`main_fed_ori.py` 改为 `os.environ.setdefault('CUDA_VISIBLE_DEVICES','0')`，外部可见卡设置不被覆盖。此次最终用户要求仅物理 GPU0，两个槽位均设置 `CUDA_VISIBLE_DEVICES=0`，各自内部 `--gpu 0`；不使用卡1。

### 12.2 日志、峰值和恢复

评估结果显式带 profile ID。Full-only 仅保存 full_accuracy/full_ce 及 P6 列，不写 P0–P5 分数；`accuracy_summary.csv` 只有 P6。每轮仍有 wall/server/evaluation 时间和原 method diagnostics。

`summary.json` 每轮原子更新：best_full_accuracy/best_round、final_full_accuracy/final_full_ce、末20轮均值/总体标准差及样本数、末50轮均值、完成轮数、耗时及方法/全部方法超参数和训练配置指纹。manifest 另存命令、原七档、实际评估档位、可见 GPU、完整 argparse 配置与 RNG 兼容说明。

`best_full.pt` 保存严格超过此前峰值时的 Full 权重、轮次、Accuracy/CE 和配置；相同峰值保留最早轮次。它明确标为官方测试集 **test-oracle peak** 的探索性快照，不是独立验证集选模。`latest.pt` 保留完整恢复状态和 RNG，恢复时要求评估模式、RNG策略、方法参数、epochs、数据与源码指纹不变，并核对 peak snapshot。新代码不会恢复旧源码的300轮 checkpoint；旧实验继续使用冻结的旧 checkout。

### 12.3 固定配置和轻量队列

固定配置：`configs/adapterfl_full_only_400.json`。公共参数固定为400轮、seed1、原保存分区、alpha=.3、Full-only；十个 run-name 均包含 ID/full_only/400/seed1。

|ID|方法|固定差异参数|
|---|---|---|
|B00|baseline|无方法改进|
|E01|m2|conflict-lambda=.5|
|E02|m2|conflict-lambda=2|
|E03|m2|conflict-lambda=4|
|E04|m3|shared=1，remainder=.95|
|E05|m3|shared=1，remainder=1.05|
|E06|m3|shared=1.05，remainder=.95|
|E07|m4|mu=.003|
|E08|m4|mu=.03|
|E09|m6|beta=.5，warmup=1|

`tools/run_full_only_sweep.sh` 调用标准库队列 `tools/run_full_only_sweep.py`，队列只启动原 `main_fed_ori.py`，不实现训练。批次严格为 B00/E02 → E01/E03 → E04/E05 → E06/E07 → E08/E09；两组400轮均成功才进入下一批。无 early stopping、自动恢复、重试、调参或按测试结果改顺序。每组独立 stdout/stderr 合并日志、manifest、每轮CSV、summary、best/latest；整批完成生成 comparison.csv 和 report.md。

默认和 `--dry-run` 只打印十条实际命令，不创建实验目录。`--execute` 才执行，根目录 `mkdir(exist_ok=False)` 防止重复覆盖。状态文件记录父/子PID、实际 argv、GPU、批次与完成状态。失败保留原错误/checkpoint，停止本批自己的另一个子进程和后续调度，绝不操作已有训练进程。

执行前与每批边界调用 nvidia-smi；有其他 GPU0 compute 进程或显存不足时停止，不改 batch/model。共享单卡必须显式 `--allow-shared-gpu`。本次用户已明确授权卡0每次两个；现场旧 M4/M5 两进程实际使用约3.6GB/24GB，前阶段 baseline/M1 双进程也已验证。`--after-suite` 仅等待旧七组队列完成状态（60秒一次检查批次状态，不读取或播报逐轮准确率）；旧队列失败或异常退出时不启动新实验。

### 12.4 验证与服务器交付

本地 fd：**65 passed**（9.83秒）；目标 Linux hh：**65 passed**（11.38秒），Linux 单测通过 `CUDA_VISIBLE_DEVICES=''` 在CPU执行，未占用训练GPU。证据为本地 `runs/adapterfl_methods_checks/full_only_pytest_local.log` 与新服务器目录 `runs/full_only_checks/pytest_linux.log`。第一次本地运行有一个测试读取中文源码时误用 Windows 默认 GBK，修为显式 UTF-8 后全部通过；不涉及算法修正。

新增检查覆盖：Full每轮仅一次原测试函数调用、全七档同步、合成小数据上三轮实际SGD的 full/isolated-all 最终权重及采样/RNG一致、旧 legacy-all RNG 差异、默认 baseline 冻结原版兼容、Full-only CSV/summary、峰值权重/轮次/最早并列、M6中断恢复及配置变更拒绝、异常恢复RNG、十组 hook 参数注入/路径唯一、失败停止同批自身进程/错误保留、GPU映射、显式共享授权、重复目录拒绝和旧队列失败停止。这些是实现检查，不是新一批真实 CIFAR-100 性能结果。

服务器独立 checkout：`/home/hh/TMC_re_adapterfl_full_only_400_20261019`，环境 `/home/hh/miniconda3/envs/hh/bin/python`。源数据仍为 `/home/hh/ICML_finish/data`，原始分区不变。现场确认旧 M4（PID1750465）/M5（PID1750467）仍在卡0，父队列PID1645598之后还有旧M6，因此不立即抢占卡0。旧源码、目录、日志及checkpoint均未修改。

最终执行命令（后台启动时将 stdout 重定向至新 checkout 的 runs 下）：

```bash
PYTHON=/home/hh/miniconda3/envs/hh/bin/python bash tools/run_full_only_sweep.sh \
  --execute --gpus 0,0 --allow-shared-gpu \
  --after-suite /home/hh/TMC_re_adapterfl_formal_0e3498c/runs/formal_300_20261018/status.json \
  --output-root runs/full_only_400_seed1
```

十条完整命令可用同脚本 `--dry-run --gpus 0,0` 生成，已在服务器保存 `runs/full_only_checks/commands_dry_run.txt`。新队列结果根目录为 `runs/full_only_400_seed1/`；排队等待不表示十组已开始或已完成。此次不做实时逐轮监控，不依据官方测试峰值在线修改任何设置。

## SemConsFL 完整分区机制激活预检（2026-10-19）

审查基准 `5d7c1522269b69c6ab08475dd58d343b000f98de` 的训练、聚合、Gate与历史公式全部保留。新增独立诊断工具，目标Linux上S03/S04/S06/S07/S08/S09各完成10轮完整CIFAR-100训练分区预检：100客户端、10slots、5本地epoch、seed1、原七档、Full-only。没有启动或排队正式400轮。

Linux专项测试105 passed。六组初始模型、真实调度、分区和源码指纹配对一致，latest实际完成10轮，229个旧B/E/N结果文件哈希保持不变。用户明确授权共用GPU0后，两进程三批全部成功且无OOM，单进程PyTorch峰值allocated约1506–1579MiB，reserved约1748–1838MiB。

完整数据上的语义分数约80%非零；S04的386/490个区域-slot Gate非中性，S06仅3/490且修改强度很弱；S07/S08/S09为0/490。S07的5次、S08的4次、S09的5次联合正风险均发生在对应客户端/区域置信度为0时。S04三个双客户端活跃区域均因共同Gate被归一化精确抵消。S08有385/444个分数数值实际改变，但未改变Gate；S07真实输入逐轮重放S09的差异均为0，不能声称已验证非中性真实分母效应。

建议暂缓正式队列，先审查方向比较粒度及成熟阶段诊断的必要性；没有为了制造激活而改公式或调整系数。详细数学解释、统计分母、置信度变化、GPU与执行证据见 `docs/SEMCONS_SCOPE_ACTIVATION_PREFLIGHT.md` 和 `docs/evidence/semcons_activation_full10_seed1.json`。

## SemConsFL 聚焦机制诊断（2026-10-19）

基于21fb7eeb115afae6a9e775bcf4835fae3b823434，离线分析E03完整400轮及六组10轮训练侧日志；仅因历史未保存上传张量，额外执行一次S07原路径3轮完整分区只读张量分解。训练/聚合/Gate/历史/正式配置均未改，没有启动正式S系列400轮。

E03负余弦426/4000，非均匀权重221/400轮；前10轮2%，第301–400轮15.1%，阶段效应明显。M2的全交集、slot级LOO、全上传降权及首轮融合锚点和S03不完全相同，不能把差别归因于区域划分。3轮诊断排除初始化轮后，124个区域比较全为非负；按同参数张量合并支持矩形后，52个区域含局部负向，107/2737个张量比较为负，但隐藏负片段仅占目标更新能量1.988%，负点积为正点积的0.437%。

S04的355个正语义风险记录全部伴随自身真类概率增加；分组相关性对轮次/模型规模敏感，目前缺接收端功能影响证据。S07/S08/S09历史重建正确，490个有效区域-slot中356个冷启动、134个已具历史，联合正风险全部落在冷启动项。唯一优先候选是先在固定接收端训练探针上观测有符号CE影响并校准语义证据，再审查替换响应分歧Q；尚未实现任何新评分或门控。

Linux114测试通过，新增9项只读诊断检查。229个旧B/E/N文件及56个上一轮预检证据哈希均未变，3轮checkpoint有限且实际完成。详细证据、已证实/待检验结论见docs/SEMCONS_FOCUSED_MECHANISM_DIAGNOSIS.md及docs/evidence/semcons_focused_mechanisms_20261019.json。

## 固定接收端训练探针CE诊断（2026-10-19）

基于0cc42156bd7add87d53525f1646d1655b55b7a33，一次5轮完整分区S07原路径，仅新增只读接收端CE观察。100个客户端的样本/增广batch在训练前固定，每客户端16类×2张；按原结构区和既有LOO方向正负哈希分层抽样，原生覆盖系数n_i/Σn注入实际区域学习增量。正式方向、Q、历史、聚合及配置均未改，未启动S00–S09的400轮队列。

共156对、2496个类别观测，零控制CE差为0。主要第2–5轮122对中86改善、36恶化，Q均>0；Q AUC=.5598，方向+Q逐轮留出MSE比方向模型增加1.00%，相同幅度/自身收益控制下增加0.39%。23/58个多接收端上下文中同一发送区域对不同接收者产生相反CE符号。真正LOO冲突仅首轮6对、一个发送者，对冲突场景Q的增量价值标为无法判断。

固定接收端训练探针的CE不进入Gate或SGD。真实第2轮原始快照CPU复算证明原聚合逐坐标一致，历史、RNG、上传/下发和模板均不变。Linux完整回归125 passed，最终11项专项复核通过；原数据保护检查通过。若用于在线决策，5轮抽样对应约636.46MB额外区域增量下行及类别CE反馈，需客户端本地探针评估和额外往返，不能默认服务器读取私有数据。

详见docs/RECEIVER_CE_FIXED_PROBE_DIAGNOSIS.md、docs/evidence/receiver_ce_fixed5_seed1_20261019.json及配对/逐类CSV。新代码仅为诊断和离线分析/复算，原始权重与图像留在服务器，不提交GitHub。

## 当前Q的最终成熟状态只读验证（2026-10-19，本地fd）

基于已审核0d57b6c，在本地从E03最终第400轮latest恢复原七档，所有50槽的实际下发均逐键验证为同一成熟Full的原生前缀。固定5批、每批10 slots、完整原分区、每槽5本地epoch、seed1，原SGD的round索引固定400，不聚合、不更新锚点、不加载官方测试集；不恢复M2历史或假设S07历史。只取回服务器最终checkpoint及manifest，没有服务器训练，也没有正式S00–S09队列。

200条配对均有效：真正负LOO冲突66对（14发送者、26接收者），35改善/29恶化/2中性；非冲突134对，73改善/53恶化/8中性。全部来自成熟独立采样，没有首轮初始化融合。冲突层Q AUC=.5793，方向=.6650；加入Q后按批次、发送者、接收者和区域留出的方向模型MSE分别增加0.39%、4.22%、1.67%、11.79%；控制注入幅度及自身收益后仍无稳定改善。Q比较类别敏感性也未形成稳定增量证据。16/30个多接收端冲突上下文呈相反CE符号，不能将Q直接解释为客户端整体有害程度。

全部50槽、200条记录离线重放支持范围/物理排除/原生alpha与来源，Q复算差0，方向最大差2.08e-16；训练RNG、七档模板、锚点和历史逐批保护通过。129项专项测试通过；3项旧报告绘图测试因本地NumPy/Matplotlib LAPACK DLL问题未完成，没有将其计为通过。分析OLS改用等价CPU Torch float64 gelsd后端，正式训练源码及数学规则不变。

本项预算用完后结束当前Q机制搜索；不新增评分、Memory或Gate，不追加采样，不启动正式队列。弱边际关联、没有稳定增量证据、仍无法外推的范围已分别报告。详见docs/MATURE_Q_FINAL_READONLY_VALIDATION.md、docs/evidence/mature_q_final5_seed1_20261019.json和原始配对/逐类CSV及JSONL；大权重/图像只保留本地。

## 结构约束聚合离线反事实（2026-10-19）

基于8f730c5，复用同一E03最终Full、5批真实上传/下发及100个客户端已物化训练探针。没有客户端SGD、全局训练、官方测试或Q/历史研究。A0/M2λ4/M3(1,.98)/S03直接调用既有聚合；A3仅为离线整体方向控制，复用S03的同份区域风险，按区域参数量汇总为物理客户端风险并保留原生覆盖归一化。正式方法文件未修改。

5批Full探针CE均值：A0 .334086937、A1 .334095483、A2 .333984930、A3 .334218787、A4 .334172013。A4真实改变22个区域-批次，3个双物理客户端冲突区域因共同Gate被归一化精确抵消；A3出现119个无本区域正风险的外推Gate。A4原尺度比A3平均改善.00004677，但同A0更新L2下反而恶化.00006640；原始平均落后M2和M3，对M3五批全部落后。不能将其归因为干预边界优越，更不支持Full Accuracy或峰值优势。

7项必要检查通过；真实调用保护原始上传/下发及锚点，保存25份反事实Full模型仅供本地复查。报告明确A3额外存在风险汇总及重复slot池化差异，不把A3/A4当作只改变作用掩码的严格单因素实验。当前不建议优先启动A4正式400轮，任务到此结束。完整对比、尺度分析、容量/参与端分组与证据见docs/STRUCTURE_COUNTERFACTUAL5.md及docs/evidence/structure_counterfactual5/。

## E03固定十组机制对照与C2按指令重跑（2026-10-19，启动记录）

训练源码固定于 `7577f422e92f3b7490785b10c210b6515d098156`。服务器继续使用 `/home/hh/TMC_re_e03_scope_components400_20261019`，没有新建项目checkout。原七档、本地SGD、采样/原分区、聚合函数、Full-only和RNG隔离不变。新增 `TonightHooks` 直接复用原M2和原覆盖聚合；T01/T03使用原M3.region，分类器/特征实验调用原M2.conflict。数学定义及对照局限见 `docs/E03_TONIGHT10_DESIGN.md`。

C1实际完成400轮，test-oracle peak为36.0200%（387轮），Final35.5300%，Last20 35.5650%，Last50 35.4244%，Final CE3.74354。用户随后明确要求停C2、更新代码后重跑；已核对PID2628652的实际argv后，仅停止该进程，停止时114条逐轮记录，原日志/checkpoint未覆盖。新C2从fresh seed1重跑，不冒充114轮恢复；新结果名为 `C2_full_only_400_seed1_after_update`。

本地89项训练/聚合回归及3项报告完成性检查通过；目标Linux合并 **92 passed**（9.32秒）。新增十种方法均经过原训练循环两轮合成小数据检查，调度一致并正常保存，不能作为真实CIFAR100性能结论。GPU0真实七档权重形状的合成上传聚合检查T05/T06/T09/C2均成功，最高allocated1,533,502,464字节、reserved1,763,704,832字节；这是显存与几何检查，没有额外客户端SGD实验。既有真实训练内存检查保留原batch50/5ep。

`tools/run_e03_tonight10.sh --dry-run` 输出11条固定命令：C2重跑加十组T实验。`--execute` 严格一次一个，物理GPU0、hh环境，顺序C2→T01→T04→T05→T06→T09→T02→T03→T07→T08→T10；套件PID2643515，启动C2 PID2643520。新C2启动manifest证明七档初始化与C1完全一致、分区相同、只评P6。独立方法失败保留日志、不重试，继续其余既定方法；既有结果和其他用户进程均不动。

结果根为 `runs/e03_tonight10_400_seed1/`。完成后自动验证每组真实400轮、best/latest和逐轮文件对齐，生成report/comparison.csv、summary.json、Accuracy/CE曲线及固定对照表。当前仅为启动交付，不能声称十组已经完成或超越E03。证据在 `docs/evidence/e03_tonight10_launch/`；训练期间不修改冻结算法、不根据测试峰值调参、不追加实验。

## 本地REF及L01–L11：六批双进程400轮（启动交付）

用户新增授权本地12组：新REF为原M2 lambda4，所有L组首先与此本地控制比较，不按服务器37.29%判断优劣。服务器7577f42版本C2/T队列未修改、未停止；本地冻结训练源码为 `975ccb1ac030cf9a937a60b589fe8539c97f7f36`。

本地新增Local12Hooks及参考只读包装、固定JSON、原入口批量调用器和报告工具。原七模型/聚合函数/采样/SGD未改；模块仅在选定新组时启用。REF直接返回原M2计算，额外只读记录更新L2。L01分离下发融合与实际SGD；L02首轮保留E03、同步后与REF代数等价，作为中性数值对照，不声称新机制。L03–L05按真实模块保护BN、分类器或二者；L06按物理能力ID0–39保护，L07按实际Full profile6；保护时改原始logit再统一稳定化。L08排除BN证据，L09等权平均有效特征/分类器余弦再计算负风险；L10对不同物理客户端的内部成对中位数作外层中位数；L11用用户规定的幅度有界负点积风险，无新增可调参数。公式和限制见docs/E03_LOCAL12_DESIGN.md。

本地fd专项及旧baseline/Full-only回归 **93 passed**（13.70秒），另有完整12组构造日志的报告/实际PNG绘图检查 **1 passed**（4.13秒），合计94。含12组原训练循环两轮合成数据、REF与原M2逐键一致、L02与REF最终权重相同、物理去重、参数保护范围、零参考、原覆盖分母、容量保护、固定队列/失败等待同批/禁止覆盖与显存仅影响调度。这些均是代码核验，不冒充CIFAR性能实验。

正式六批已在本地4080 SUPER启动，conda fd为D:/software/minicoonda/envs/fd/python.exe，PyTorch2.7.1+cu128。数据根C:/Users/admin/Desktop/submissioning/code/ori/AAAITMC_finish_need_code/data/cifar100，完整50000/10000，原有分区。套件PID18468，第一批REF PID40452、L01 PID50084，实际双进程使用GPU0且初始七档、分区一致，只评P6。启动后GPU总占约5882MiB/16GB，剩余10166MiB；没有改batch或训练协议。Windows使用fd激活所需Library/bin和Scripts路径，并保留原main已有KMP_DUPLICATE_LIB_OK设置，报告绘图已验证可用。

输出根为runs/e03_local12_full400_seed1；按REF/L01、L02/L03、L04/L05、L06/L07、L08/L09、L10/L11自动执行。每批两组真实完成400轮后才进入下一批；失败保存错误/checkpoint并等待同批另一组结束后停止后续批次，不自动重试。结束自动生成以本地REF为对照的表格、全部400轮Accuracy/CE曲线及冲突/权重/更新L2统计。启动证据在docs/evidence/e03_local12_launch/。本节仅确认启动，不预报任何精度收益；完成后再按真实结果给出最多两个建议机制或保留REF，并停止本批扩展。

## 服务器C2重跑与T01–T10最终交付（2026-10-20）

冻结7577f422e92f3b7490785b10c210b6515d098156的11组新训练在北京时间06:43:21全部成功完成400轮，status finished、failed为空；无重试或临时改参数。checkpoint完成性与首次峰值best对齐已在服务器逐组检查。下载原始标量日志、manifest、完成与资源记录和图，不下载或提交权重。C1/C2/T全部12条实际初始化与400轮slot调度一致，原初始化和models/Fed.py也与历史E03一致。

没有新组超过E03的37.29% test-oracle peak。T04追平37.29%，Final36.6300%、Last20 36.8410%、Last50 36.7704%；相对E03，Final低.2100pp，末段高.0525/.0758pp，未全面占优。T04只有首轮相对同输入E03差异L2=.302450，第2–400轮差异严格0；原首轮加权融合不是达到该峰值的必要条件。其他T组峰值36.42–36.83%，全部Last20/50低于E03。C1/C2峰值36.02/36.32%，同样没有提升。

T09在215轮实际恢复373个负冲突槽位剩余分量，C2在209轮调整423个负投影槽位，不能将失败解释为未激活。修正离线报告的口径：原protected_components仅是T09字段，新增C2独立投影计数；训练源码和输出数据不变。T09后续L2比均值1.001329，同输入差异主要含方向成分，却未带来性能收益。T07负余弦增至33.20%、权重最小.3378，更多冲突也未改善精度。T05物理去重和T06成对支持规则改变评分/聚合，但仍落后；T01/T02/T03/T10均未显示可保留的尺度协同。

新增只读tools/analyze_tonight_results.py，从日志范数/内积分解同输入更新差异的平行与垂直成分，并生成对照图与成本统计；不读取训练图像或测试集作决策、不创建新模型/SGD、无算法修改。报告与分析专项检查通过。保留E03为最强参考，T04仅作初始化控制，当前无新机制足以作为优于E03的SemConsFL核心；本服务器任务停止，不追加实验。

完整15组对比、固定10组设定、首轮/后续诊断、因果限制和曲线见docs/E03_TONIGHT10_RESULTS.md，原始400轮CSV及诊断见docs/evidence/e03_tonight10_results/。所有峰值明确为官方test-oracle peak，单种子探索不构成独立泛化证据。本地REF/L系列保持独立，以本地REF另行判断，不混用服务器峰值。

## 本地REF/L01–L11最终交付（2026-10-20）

六批12组全部真实完成400轮，于北京时间08:28:18结束，status completed、failed为空。冻结训练源码975ccb1ac030cf9a937a60b589fe8539c97f7f36保持不变，无重试、调参、额外种子或模型/SGD协议改动。全组初始化、400轮slot调度一致，每组实际Full训练587槽；latest与best/逐轮文件完成性已由原队列检查。完成自动任务ref-l01-l11已暂停，后续不追加训练。

主要本地REF峰值36.9500%（352轮），Final36.3200、Last20 36.3545、Last50 36.2316。L04分类器Native聚合峰值37.2400%（308轮），Final36.6300、Last20 36.7505、Last50 36.6288；相对REF四项分别+.2900/+.3100/+.3960/+.3972pp，Final CE3.631186，低于REF3.646787。排除等价中性L02后，L04是唯一在四项Accuracy指标均超过REF的新机制。

L02峰37.3300%，但其全部400轮相对自身同输入原M2的聚合差异严格为0，第2–400轮399次下发均同步；不能将其高于REF的.38pp当作新算法收益。相同数学算法、相同初始/调度却有不同数值轨迹，表明运行间非复现性仍需认真对待，现有证据不能定位具体CUDA算子或仅据一个中性重复估计波动分布。L04的峰值增量小于该中性峰差，末段均值则同时高于REF和L02，仅作为初步保留理由，不宣称稳定泛化优势。

L04只保护51,300个classifier参数（Full的.44904%），实际221轮改变，变化范围验证未超限；后续相对同输入M2的L2比均值.9999921，差异能量约99.85%垂直于原更新，不能等价解释成全局缩步长，但首轮分类器融合与后续局部权重变化尚未完全拆开。L03 BN保护和L05联合保护均落后；L06低能力保护87个负方向槽位、L07 Full保护中仅35个原负方向槽位，均未改善；L09分块等权使负方向增至29.8%，也无收益。L10中位数参考增加计算成本、精度仍低于REF；L11幅度风险平均为原负余弦风险的.6421，Last20低1.6595pp，停止该固定版本。

本次只补充离线分析和报告：tools/analyze_local12_results.py读取保存的诊断/参数形状，验证作用范围并绘图；tools/report_local12.py将代数等价L02从新机制排名中排除，保留完整表格数据。报告/几何回归3 passed，未改变训练代码或启动新SGD。原始CSV/diagnostics/manifest/完成及资源记录、图和完整科研解释见docs/E03_LOCAL12_RESULTS.md及docs/evidence/e03_local12_results/；权重保留本地runs，不提交大checkpoint或私人资料。

仅推荐L04作为后续研究候选，保留原M2/REF及L02数值对照，不推荐第二个新机制，不继续本轮BN/容量/评分/幅度固定版本的无目的调参。全部峰值标明test-oracle peak，以本地REF为主，不按服务器37.29%评判。当前只是参数类别分域聚合的探索结果，不是已验证的功能语义可靠性机制。本任务结束，等待后续授权。

## 确定性五组分类器对照最终交付（2026-10-20）

本节为后续受控验证，更新上节对L04的初步保留判断。R0/R1/R2以及最后并行的B00-D/R3均实际完成400轮，无失败或重试。原三组不重跑，所有服务器代码与结果均在/home/hh/TMC_re_e03_scope_components400_20261019。原三组训练冻结c183962，最后两组冻结115c7f9；只新增默认关闭的classifier end-round1选项、原baseline入口支持和轻量双进程/报告工具，原M2评分、lambda4、七模型、SGD和覆盖聚合函数均保持不变。启动本地与Linux96tests通过。

此前真实GPU复核已定位旧数值分叉到同输入首次forward相同但backward梯度不同；严格确定性策略下本地REF/L02三轮逐张量权重、评价、RNG和调度完全相同。五组正式实验统一strict deterministic、cuDNN deterministic、benchmarkFalse、TF32off、CUBLAS4096:8，Torch2.7.1cu118/RTX4090；初始化、原分区及400轮slot调度一致。该数值环境与旧E03 37.29%的结果不同，不能直接跨批严格配对。

B00-D原生AdaptiveFL的test-oracle peak36.6800%r350，Final36.0300、Last20 36.3190、Last50 36.1154、CE3.672783。R0原M2 peak36.5600r308、Final35.9100、Last20 35.8475、Last50 35.7176：相对Native的四项Accuracy全部下降。R1全程classifier Native peak36.4000、Final35.8400、Last20 35.6910、Last50 35.6068；R2第2轮起保护 peak36.6600、Final36.0600、Last20 35.9115、Last50 35.8070；R3仅首轮保护 peak36.5700、Final35.8500、Last20 36.0835、Last50 35.8768。R2的Final比Native多0.03pp，但Peak和后期均值落后；R3的Final CE3.621463最低，也没有转化为更高Accuracy。

R1/R2实际212/228轮改变聚合，classifier范围不超过Full的0.44904%；R3只改变第1轮，差异L2=.04329591，第2–400轮相对各自同输入原M2差异严格0。R0有214轮非均匀权重，不能以门控未激活解释结果。首轮Native的条件Last20效应R3-R0=+.2360pp而R1-R2=-.2205；后续Native效应R2-R0=+.0640而R1-R3=-.3925。不存在普遍独立的classifier保护收益，首轮与后续存在非加性轨迹差异；首轮作用同时包含独立下发融合与第一次SGD权重，不宣称已将二者分离。R1/R2的同输入L2比约.999989且差异大部分非平行，几何变化真实但未胜过Native。

新增纯离线tools/analyze_l04_results.py核对五组原始标量、checkpoint核验证据、调度/执行设置和2x2差值，不读取训练图像或测试集生成评分，不创建新SGD。实际结果与相关聚合回归4tests通过；完整证据、曲线和中文报告见docs/evidence/e03_l04_final_results及docs/E03_L04_FINAL_RESULTS.md。所有best/latest在服务器验证实际400轮/首次峰值对应，Full参数形状一致且有限，权重未上传。完成检查自动任务已暂停。

最终以Native B00-D为该确定性环境的优先参考，保留原E03作为历史参考和对照代码；不将M2或任何classifier保护策略选为下一版核心，停止当前路线的无目的调参。后续核心创新应重新审视问题与功能影响证据，本轮不实施新算法或追加训练。单seed官方test-oracle探索不证明所有环境下M2均无效，也不证明独立泛化收益。

## Native来源/覆盖/离散度十组实现与启动前检查（2026-10-20）

本节为用户后续明确授权的新固定研究范围：G01–G10全部从Native AdaptiveFL起步，不组合M2/M3/Q/S03决策或分类器保护。原七模型、抽样、SGD、同步、分区及确定性Full-only协议不改。原训练循环仅增加opt-in hook工厂、必要的服务器下发快照和真实槽位传递；原Aggregation_AdaptiveFL和models/Update.py未改。build_regions仅用于真实九区域几何。

G01模型档位组内样本加权、组间等权；G02在每个区域按物理客户端合并覆盖slot并将原数据质量计一次；G03按ID0–39/40–69/70–99能力组内Native、组间等权；G04/G05为G01的静态七模型共同区与补集；G06为覆盖slot等权。中性成比例权重直接保持Native，避免把额外舍入误差当作机制。

G07只存九个累计整数，每轮每区域物理身份去重计数，第1轮照计数但不干预；以九区域累计次数算术均值Cbar计算clip(sqrt(Cbar/C_r),.8,1.2)。G09将重复槽位汇为物理统计块、保留Native总slot质量，以加权跨客户端离散度估计均值噪声N，使用非负信号S=max(norm(mu)^2-N,0)的固定Wiener系数S/(S+N)；单客户端和全零信号/噪声直接Native。该交换性噪声近似不能被称为已验证的Non-IID更新有害性。G08/G10仅在自身相同上传上匹配虚拟G07/G09的全局更新L2，不声称两条后续训练轨迹始终同范数。

全部方法第一轮严格Native，第2轮开始新方法，覆盖为空绝不更新。G07–G10显式核对同步后的实际下发与原Full prefix一致，再作用于Native锚定学习变化，不更改首轮初始化融合。本地93项检查通过，含十种hook两轮小模型原循环与旧baseline/all/full路径回归；真实七ResNet形状合成GPU检查通过，9区域、全参数有限，G08/G10尺度相对误差约3.25e-8/2.84e-8，allocated1246.8MiB、reserved1580MiB。这是合成上传正确性/内存检查，不是CIFAR性能实验。

固定配置configs/native_groups10_400.json及完整公式docs/NATIVE_GROUPS10_DESIGN.md，轻量启动tools/run_native_groups10.py。指定五批G01/G02、G03/G06、G04/G05、G07/G08、G09/G10，GPU0每批最多两进程；只有调度可因显存改为串行，训练协议不变。复用已完成B00-D，不重复基线。全部400后自动汇总原始日志、Peak/Final/Last20/50/CE、区域权重/覆盖/增益、L2控制误差与成本；阶段最高值标test-oracle，最终最多保留1–2候选或全部否定，不据结果改公式/追种子。

服务器同一checkout冻结训练代码36b84b858b6a03c038ad8a142ededf870c5b5ce6，Linux93tests通过；实际七形状GPU合成聚合检查通过，reserved1598MiB。正式父PID3243806，首批G01/G02子PID3243837/3243838，GPU0实际concurrency2后台启动。首次manifest核对初始化七档、原分区、训练及确定性策略均与B00-D相同，真实50k/10k，仅P6测试。证据docs/evidence/native_groups10_launch/server_startup.json含真实argv、预检和资源检查，不含逐轮性能结果。此处仅交付启动，不宣称400轮已完成或方法有效；完成后再基于真实十组结果作最终研究判断。

## Native十组400轮最终交付（2026-10-20）

G01–G10五批各两进程在同一服务器checkout与GPU0全部完成400轮，于北京时间22:31结束，status completed、failed为空，无重试、调参或追加种子。冻结训练提交36b84b858b6a03c038ad8a142ededf870c5b5ce6保持不变，B00-D未重跑。服务器直接核验全部latest实际400轮、best首次峰值对应CSV、Full参数有限且形状一致；离线核对初始化、分区/确定性协议、400轮slot调度及首轮Native评价均与B00-D一致。所有组Full训练587槽，165轮有重复物理客户端；完成自动检查已暂停。

G03能力层级均衡最强：test-oracle peak38.4500%r339、Final37.7800、Last20 37.6585、Last50 37.6056、Final CE3.587194，相对Native B00-D四项Accuracy分别+1.7700/+1.7500/+1.3395/+1.4902pp。G05非共同区域档位均衡peak36.9900、Final36.3000、Last20 36.3890、Last50 36.2874，四项小幅改善；G02虽peak37.1100，Last20低.0765pp，不因峰值单独入选。

G03将公共区低/中/高能力来源的平均Native贡献16.5565%/35.5676%/47.8759%改为27.9449%/35.5890%/36.4662%，与原训练数据质量39.862%/30.746%/29.392%并不相同。来源均衡改善的是资源约束下的实际贡献，不能把层级当成天然更新质量。G01/G06没有修正该物理来源失衡。G03实际398轮改变聚合，Full-only单来源区域直接保持Native。G05的实际变化仅发生于部分共享额外参数，不能误报为Full-only保护或补偿。

G07/G08的九区域累计机会完全相同且逐轮去重正确；公共区G07一直使用.8、最低覆盖区域有上传时1.2，方法确实生效，但两组Accuracy均低于B00-D，结构补偿对尺度对照没有一致优势。G09公共区平均收缩系数.0771，83轮为0，独立客户端不足时正确回退；全参数覆盖加权零系数比例均值16.57%。G09峰27.96、Last20 27.0710，G10峰32.34、Last20 31.9830，均明显退化，当前非IID离散度作为噪声的固定近似不支持继续。

G03同输入Native反事实的更新L2比均值1.18635，差异能量约95.78%为非平行分量；G05对应1.03352和93.46%。变化不是单纯统一缩放，但没有G03自身尺度匹配对照，不能声称性能收益已排除有效步长贡献。G08/G10只匹配自身轨迹同输入虚拟方案，区域gain是虚拟系数、实际用global_scale。G10最大相对匹配误差.0111%来自第127轮目标L2约1.0264e-4、绝对误差1.14e-8；保留真实浮点误差，不报严格等范数。

本次只新增tools/analyze_native_groups_results.py、实际证据回归tests/test_native_groups_results.py、中文docs/NATIVE_GROUPS10_RESULTS.md和原始标量证据。原聚合/客户端/采样/训练源码均未改。报告与相关算法/队列31tests通过；四张结果图已视觉检查。通信相同，每组float32状态上行88,387,695,816字节、下行同量，无额外客户端计算/通信；模型权重仅留原服务器目录，不上传私人资料。

最终最多保留G03为主候选、G05为低优先级次选。停止本轮G01/G04公共区档位均衡、G07/G08固定机会补偿、G09/G10固定离散度收缩的继续调参，不追加新实验。全部峰值为单seed官方test-oracle探索，来源/能力/Non-IID效应尚未完全分离，不能宣称统计泛化验证或功能语义可靠性创新已经成立。完整表、400轮曲线、对照差值及证据见docs/NATIVE_GROUPS10_RESULTS.md和docs/evidence/native_groups10_results/。

## G03固定十组候选实现与启动前检查（2026-10-20）

本节为用户后续授权P01–P10，围绕G03来源均衡，不重跑B00-D/G03，不新增评分或客户端机制。复用native_group_aggregation.py及原双进程/报告脚本，仅增加可选config、十个固定分支和诊断；原聚合、客户端SGD、模型、采样、同步与评估函数不改。G01–G10规则保持兼容。

P01只用Native方向匹配同输入虚拟G03整体L2；P02用G03方向匹配Native L2。P03/P04组间质量分别sqrt(M)/inverse sqrt(M)，组内原样本；P05为参与能力层4:3:3先验；P06组内物理数据质量去重；P07能力层/实际档位两级等权；P08/P09只限静态共同区/补集；P10只让classifier.weight/bias Native，BN和特征仍G03。全部首轮严格Native。按明确的单能力层中性要求，P06/P07也在单能力层区域返回Native；P01/P02最后的全局尺度可能缩放已覆盖的单层区域，保持其预定范数对照定义。

组质量均正、P03/P04等质量直接复用G03；P06按当前区域的覆盖slot去重，不合并真实SGD；P07不生成缺席组。P01/P02零源与零目标中性，零源非零目标无可缩放方向时明确失败，不用epsilon造更新。每候选只读计算同输入G03作为诊断，记录实际/Native/G03的L2、差异和更新余弦；余弦不进入决策，不消耗RNG。P01实际来源份额Native、虚拟G03份额单列；P10来源份额注明classifier排除，避免把日志误读为全参数统一权重。

本地111 tests通过，覆盖十候选首轮/作用范围/重复客户端/尺度/分类器、原训练循环两轮小模型Full-only及旧AdaptiveFL/M2和评估路径兼容。真实七ResNet形状合成GPU检查通过，九区域/参数有限、P01/P02范数相对误差约4.03e-8/4.32e-8。这些是正确性和内存检查，不是CIFAR性能结果。固定配置configs/g03_refinements10_400.json，完整公式docs/G03_REFINEMENTS10_DESIGN.md。启动后另存Linux测试、真实argv/PID与初始化配对证据，实际400轮完成前不宣称方法有效。

### P01/P02首批存储配额失败，未完成正式十组

同一服务器checkout冻结0da1ed2cb38953476bf3400cf025876485f5bf7e，Linux111 tests及七形状GPU检查通过，reserved1076MiB。父PID3399638、子P01=3399672/P02=3399673在GPU0并行启动。两组第1/2轮真实记录已写，但第2轮写best_full.pending时hh账户配额用尽，出现PytorchStreamWriter写文件错误及Errno122 Disk quota exceeded；队列按规则失败停止，P03–P10未启动。无OOM、自动重试或修改数学规则。

直接quota显示hh用户102400M已用、100G硬限制，虽df还有123G物理空间，账户无法继续写入。直接加载两组latest/best仍有效但只对应第1轮，日志已到第2轮，不能宣称已完整保存2轮或直接续训。两组初始化、训练/确定性配置与G03/B00-D一致，首轮严格Native；原CSV、诊断、manifest及完整stderr证据已下载。没有删除或覆盖旧结果。自动完成检查已暂停，等待用户解决存储及重启安排。

当前只交付算法、测试和诚实失败状态，不作400轮性能结论，不按2轮测试数据改参数。完整证据与后续存储条件见docs/G03_REFINEMENTS10_RUN_STATUS.md。

### 用户授权清理权重及原定十组重启准备（2026-10-21）

用户明确要求删除runs下全部.pt。仅在服务器绝对目录/home/hh/TMC_re_e03_scope_components400_20261019/runs内处理普通.pt文件，先核对根路径和每个目标，跳过符号链接；实际无符号链接，删除59个文件、逻辑总量2.51426GiB。全部非.pt文件和历史标量保留，没有删除目录、训练数据或其他项目。删除前直接验证B00-D/G03实际400轮latest/best与峰值记录一致，删除清单及quota前后证据保留服务器。

仅修改历史参考检查：B00-D/G03读取真实completed标记与完整1–400有限Accuracy/CE CSV，不再依赖被用户删除的权重；本批新训练仍必须通过latest/best实际400轮核验。追加回归确认无.pt的真实G03历史可比，2轮或未完成参考仍拒绝。本地112 tests通过；聚合、SGD、模型、配置公式及RNG策略均不改。失败目录保留，原定十组fresh seed1使用新输出根runs/g03_refinements10_full400_seed1_after_cleanup_20261021，避免日志/权重轮次不一致的错误续训；不重跑B00-D/G03。

清理后quota为99782M/100G，约2.56GiB可用；Linux112 tests通过。服务器同一checkout冻结7d686b6993e4d81306508f75788733ffdc879cd4，父PID3586425、首批子3586431/3586432在GPU0后台concurrency2启动。首批七模型初始化、训练/原分区与确定性配置均匹配保留的B00-D/G03 manifest，完整数据且仅Full测试。真实删除清单、历史删除前checkpoint核验、Linux测试和启动证据见docs/evidence/g03_refinements10_launch/after_cleanup_startup.json；当前仅确认启动，不报400轮结果。完成后检查安排2026-10-21 18:00，非实时逐轮监控；未再授权自动清理新checkpoint。

用户随后指定物理GPU1。确认GPU1空闲；对本套件父/子PID先验证cwd和命令身份，暂停后核对两组latest/best、CSV和diagnostics均对应38轮、方法状态轮次一致，再仅停止本套件GPU0进程，保留模型与RNG。外层队列新增显式physical-gpu及仅限migration_paused的continue-suite，使用既有method-resume继续400轮，不改训练源指纹、不重跑38轮、不自动重试失败；后续批次GPU1。checkpoint迁移状态与原GPU0环境单独记录，保留日志追加和资源快照。局部未保存计算被丢弃，正式总轮数仍400。新增调度回归及原专项本地113tests通过；新进程资源wall/peak只代表恢复阶段，最终成本需结合完整逐轮与累计日志。
