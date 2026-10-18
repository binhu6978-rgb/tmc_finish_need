# AdapterFL 七模型六种独立方法：代码审查报告

## 交付范围与状态

开发分支：`feature/adapterfl-six-methods`，从 `c51c166` 建立。只做实现和 CPU 检查，没有运行正式 300 轮、没有启动 GPU 实验、没有访问或中断服务器训练，也没有打包。按任务第 9 节要求，先本地提交；**等待明确要求后才推送 GitHub**。因此本次不声称已经存在可在线审查的新分支链接。

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

原各档在初始化时独立随机构建，并非第一轮开始前就从 Full 同步。M2/M5 所以必须使用各槽位**实际下发快照**计算本地增量，尤其不能在首轮减去 Full 的前缀。M1/M3/M6 按设计作用于 `W_aggregated - W_previous_Full`，首轮该差值也包含初始独立模型的融合差异，本次没有增加额外同步或 warmup。

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

### M3：两种覆盖区域的固定服务器融合系数

调用路径：`MethodHooks.aggregate` m3 分支；先原聚合，再对互斥覆盖 mask 分别更新。

```text
new_region = W_region + alpha_region*(A_region-W_region)
```

默认 `--partition-alpha-shared 1 --partition-alpha-remainder .9`。只作用于本轮实际覆盖坐标，无覆盖位置不动；同一坐标恰好归属一个区域。当前只实现固定系数，**没有**根据官方测试集或历史统计自动调 alpha。记录两区域的原始/实际应用更新 L2、相对上轮更新范数波动和实际 alpha；历史只用于诊断，两个标量随 checkpoint 保存。两 alpha=1 时精确返回原聚合值。

### M4：共同覆盖参数的近端约束

调用路径：`configure_local` 仅给 M4 设置属性 → 原 `LocalUpdate_AdaptiveFL.train` 的可选 loss hook → `prepare_prox` / `prox_loss`。

```text
R = 0.5 * sum_{shared named_parameter coords} (theta - actual_initial)^2 / N_shared
L = original_CE + lambda_prox*R
```

默认 `--shared-prox-lambda 1`。目标从**实际训练副本**进入本地 SGD 前取 detached clone，仅对 named_parameters 对应 shared 前缀计算梯度；不对任何统计 buffer 正则化，不把服务器/初始参数接入反向图。其他原 optimizer、lr decay、CE、loader、drop_last 不变。lambda=0 完全关闭 hook、与原单步/多步更新逐值一致。记录 task CE、归一化 R、lambda、实际目标参数量、lambda/N 和 SGD 步数。

由于 N_shared=2,120,985，默认平均正则可能很弱，不能仅凭开关存在宣称它已改善真实训练；没有为追求精度调整 lambda。测试用小模型的非零正则证明其梯度和第二步更新确实不同，未将该合成结论当作 CIFAR 收益。

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

### M6：只在连续覆盖坐标平滑历史

调用路径：原聚合 → `MethodHooks.aggregate` m6 分支；原服务器没有 momentum，此方法不重复服务器已有优化器（原客户端 SGD momentum 不等同于服务器 momentum）。

```text
delta_t = A_t - W_previous
if covered_now and covered_previous:
    v_t = beta*v_previous + (1-beta)*delta_t
elif covered_now:                 # 首次或覆盖间隔后重新参与
    v_t = delta_t
else:
    v_t = 0
new = W_previous + v_t            # 仅本轮覆盖坐标
```

默认 `--history-beta .9`。首轮和间隔后再次覆盖的坐标用当前增量初始化；本轮未覆盖坐标完全保持旧值，同时清空该坐标历史，下一次不沿用陈旧方向。beta=0 时精确返回原聚合值。记录原始/平滑更新 L2、当前 delta 与上一轮 velocity 的方向 cosine、覆盖与重新初始化坐标数；该 cosine 不是两轮 raw delta 的 cosine，不混淆日志含义。

## 4. 修改清单

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

**43 passed**，证据 `runs/adapterfl_methods_checks/pytest_final.log`，包括：

- 冻结原循环 vs 新的默认旧路径 / named baseline：真实七档 ResNet、固定输入、同样采样、本地 SGD、重组与七次测试，最终所有参数 **bitwise equal**，Python/NumPy/Torch RNG 状态相同。
- 冻结原本地 train vs 新 train 在 M4=0 时 bitwise equal；未额外设置 proximal snapshot。
- M1 单次样本权重归一化及 coverage 缩放；M2 实际下发基准、留一参考、负向衰减、单槽/零范数/不同 shape/极端 lambda；M3 两区域互斥、更新系数与诊断；M4 detached 目标、正确梯度支持、非零正则实际改变更新；M5 相同支持比较、正常更新不变、异常更新裁剪；M6 不覆盖不漂移、间隔后重初始化。
- 六方法和 baseline 的参数有限/shape 合法、旧服务器输入不被原地污染、未覆盖尾部不变；所有方法状态导出/读取后后续聚合一致。
- 真实七档配置的 Conv/BN/shortcut/classifier 映射，以及六个 hook 的真实参数形状 CPU 聚合与 strict load。
- M6 在同一个原始训练循环中，三轮合成训练“不中断” vs “完成第一轮后中断、从 checkpoint 恢复”：最终七档权重与 RNG **bitwise equal**，CSV 轮数为 1/2/3 无重复。

首次专项检查的唯一失败来自测试替身的 profile tuple 漏填原代码要求的第三项参数量，补正测试 fixture 后通过，没有改训练算法来迁就测试。

### 扩展旧回归：未完整通过的本地原生异常

额外检查了 `test_fd_semconsfl.py`、`test_hgkr.py`、`test_hgkr7.py`、`test_adapterfl5.py`，使用 `CUDA_VISIBLE_DEVICES=-1`，没有 GPU 工作负载。原日志 `regression_cpu.log` 在旧 HGKR batch50 CPU 训练的 BatchNorm 中出现 Windows `access violation`（退出码 -1073741819）。限制线程为 2 后，仍在旧训练 backward 路径发生同类异常，日志 `regression_cpu_threads2.log`。

单独隔离该旧测试，当前源码一次干净通过，日志 `isolated_current.log`；将本地 train 替换回本次冻结的原函数再隔离，也出现原生访问异常日志（虽随后 pytest 打印 1 passed，不能视作干净通过），见 `isolated_frozen.log`。因此不能仅凭这个现象把问题归因于新方法，根因尚未确定。

最后仅在测试进程关闭 MKLDNN、限制 2 线程核验，旧回归推进到 66% 后仍在旧七档 HGKR 的 CPU 卷积中遇到访问异常，见 `regression_cpu_native_backend.log`。没有将这些后端设置写入项目或正式训练，也没有修改旧测试以隐藏问题；停止继续尝试后端排列。

**交付结论：新功能的 43 项专项检查已通过；扩展旧回归未完整通过，保留该限制，不能声明全项目回归通过。** 建议代码审查后，在目标 Linux/PyTorch 环境复核这些旧回归与获准的短程测试。没有用正式训练来代替检查，没有精度收益结论。

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
4. M4 的 mean-normalized penalty 在真实百万参数上可能很弱；M6 首轮按当前真实更新初始化，未额外避开独立模型初始化融合。均不据测试精度自动改参数。
5. 只支持本仓库经核验的 float32、BN 无统计缓冲区的七档 ResNet；未知标量 buffer 或其他网络不做猜测性兼容。原聚合对标量 BN buffer 的潜在问题没有在 baseline 中修补。
6. 原最大客户端时间指标不能替代真实墙钟。M2/M5 下发快照、M6 历史状态和逐轮 checkpoint 会增加内存/时间/I/O，尚未测 GPU 吞吐。M6 持久化速度和 bool 覆盖 mask 不会更新无本轮覆盖的参数。
7. 共享参数代码只在新分支更改；不要在服务器运行目录原地换文件。旧 FD/HGKR 的已有 checkpoint 指纹含源码/参数，本分支不得用于恢复旧实验，旧实验继续使用其原 checkout。
8. 现有默认入口捕获异常后静默 continue、`get_final_acc` 会基于测试序列挑一个显示轮次：原旧路径保留；named 模式改为失败明确上报、最终轮汇总，不把该显示规则当作验证集选择。
