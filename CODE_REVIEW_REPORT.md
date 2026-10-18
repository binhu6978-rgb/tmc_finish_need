# AdapterFL 七模型六种独立方法：第三轮修订报告

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
