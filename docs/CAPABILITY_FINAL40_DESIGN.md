# 最后一轮固定预算：40组×400轮

本次恰好40组，全部seed1、严格确定性CUDA、原七ResNet、本地SGD、400轮Full-only、无.pt。GPU0二十组A，GPU1五条件各四方法。两卡各自独立十批、每批最多两组；不运行第41组，不根据中途测试表现改参数或顺序。所有方法第1轮完整Native。

## 不变的基础与新条件

原`Aggregation_AdaptiveFL`、`Algorithm/Training_AdaptiveFL.py`、客户端SGD、模型切片/同步、采样与评估函数均不修改。新增服务器hook复用`build_regions`、`write_weighted_region`及原G03/P04/P05权重函数。原权重函数只新增默认仍为4:3:3的可选人口参数；旧实验入口继续使用旧定义。新40组仅通过独立薄入口调用同一训练循环。

P04在真实区域覆盖的实际能力层之间分配`π_g ∝ M_g^(-1/2)`，层内按真实样本-slot质量加权；P05使用实际物理人口先验。上传是原训练后完整状态，不另行改变下发融合；第2轮后子模型已原生同步，保留原Native覆盖归一化。未覆盖坐标保留旧Full。

单能力层的来源重加权均返回Native。A01–A04最后是全局位移L2控制，可能缩放已覆盖的单层区域；如果整轮只有一个能力层，虚拟方法本身等于Native，最终仍精确Native。不能同时冻结所有单层区域并声称匹配完整更新L2。

| 条件 | 数据集 | α | 物理能力人口 | 边界 | 类别数 |
|---|---|---:|---|---|---:|
| A | CIFAR-100 | .3 | 40/30/30 | 40,70 | 100 |
| S1 | CIFAR-100 | .2 | 40/30/30 | 40,70 | 100 |
| S2 | CIFAR-100 | .5 | 40/30/30 | 40,70 | 100 |
| S3 | CIFAR-100 | 1.0 | 40/30/30 | 40,70 | 100 |
| S4 | CIFAR-100 | .3 | 20/30/50 | 20,50 | 100 |
| S5 | CIFAR-10 | .3 | 40/30/30 | 40,70 | 10 |

原数据函数实际强制`balanced=False`，即原`dirichlet_split_unbalanced`；旧文件名含bal不代表实际样本量均衡。A/S4逐字复用原α=.3记录，SHA仍41ab2d34…；S1/S2/S3/S5在启动前以该原函数、seed1、min_size10、max_retry500生成一次，完整50000训练样本无重复无遗漏。新分区独立保存于本次prepared，不读错旧α=.3，也不覆盖任何旧记录。

数据准备在独立进程进行，分区生成使用函数内部局部RandomState(1)，不扰乱训练RNG。每个正式进程在原未使用Full初始化之前重新seed1。CIFAR-100保留原ToTensor/Normalize；CIFAR-10沿用项目原训练RandomHorizontalFlip/ToTensor/Normalize与测试无Flip流程，真实10类网络、真实50k/10k数据。每条件四方法的初始化、分区、slot调度及首轮Native结果配对验证；不同数据集不按绝对Accuracy混排。

## A01–A20的固定公式

记W为聚合前Full，B/H/J为同输入Native/P04/P05；以下只写实际覆盖坐标。

| ID | 与P04的预定差别 |
|---|---|
| A01 | `W + ||H-W||/||B-W|| * (B-W)` |
| A02 | `W + ||B-W||/||H-W|| * (H-W)` |
| A03 | `W + ||J-W||/||B-W|| * (B-W)` |
| A04 | `W + ||B-W||/||J-W|| * (J-W)` |
| A05 | 仅静态七模型公共区P04 |
| A06 | 仅静态部分共享区P04，支持档位数严格大于1且小于7 |
| A07 | 分类器Native，其余P04 |
| A08 | 仅分类器P04 |
| A09 | BN affine Native，其余P04 |
| A10 | 仅非BN特征参数P04，BN和分类器Native |
| A11 | 区域内同物理客户端上传先平均，其样本质量层内计一次；层间以去重样本质量反平方根分配 |
| A12 | 区域内不同物理客户端数量作为层间质量，层内仍Native样本-slot加权 |
| A13 | 区域内slot数量作为层间质量，层内仍Native样本-slot加权 |
| A14 | 整轮每层全部slot样本量作为层间质量；区域只对实际出现层归一化 |
| A15 | 整轮每层去重物理客户端样本量作为层间质量；区域层内仍原样本-slot加权 |
| A16 | 层间目标`population_g/sqrt(M_g)` |
| A17 | 层间目标`training_data_mass_g/sqrt(M_g)`，先验只由训练分区统计 |
| A18 | `.5*B + .5*H`，同输入完整状态等比例插值 |
| A19 | 层间目标`M_g^(-.25)` |
| A20 | 层间目标`M_g^(-.75)` |

BN参数键来自真实BatchNorm模块类型，不依赖易混淆的名字片段。A11只改变服务器权重，不合并独立SGD；A15只改全局质量定义，不叠加A11的层内去重。P04/P05的原−.5指数和人口规则不重新训练。L2控制以float64累积，目标/源均零中性，零源且非零目标没有可缩放方向时明确失败，不用epsilon伪造更新。

所有A方法只读计算同输入P04作参照，记录实际/Native/P04的L2、位移余弦、差异、真实覆盖和来源份额。这些余弦不用于决策。A18来源份额按Native/P04各半记录；部分参数作用域的份额与weighted_parameter_numel一并保存，报告按真实参数范围合成，不把classifier/BN例外误写成整区域统一权重。

## 运行与交付

唯一服务器项目仍为`/home/hh/TMC_re_e03_scope_components400_20261019`。固定根`runs/capability_final40_400_seed1_20261021/`，预备分区在prepared，两卡结果分别gpu0/gpu1；不覆盖任何旧目录。正式单个结果为`gpuN/<method>/<ID>_full_only_400_seed1/`。

```bash
/home/hh/miniconda3/envs/hh/bin/python tools/run_final40.py --prepare --dataset-root /home/hh/ICML_finish/data
/home/hh/miniconda3/envs/hh/bin/python tools/run_final40.py --dry-run --dataset-root /home/hh/ICML_finish/data
nohup /home/hh/miniconda3/envs/hh/bin/python -u tools/run_final40.py --execute --dataset-root /home/hh/ICML_finish/data > runs/capability_final40_400_seed1_20261021/launcher.stdout.log 2> runs/capability_final40_400_seed1_20261021/launcher.stderr.log < /dev/null &
```

仅批次边界检查显存；充足时每卡两任务，资源不足仅降低并发或记录resource_blocked，不改训练参数、不杀其他进程。每卡任一失败等待本批同伴结束后停止该卡后续批次，另一卡独立继续；不重试、不静默跳过。父队列和每卡status记录真实PID/argv，stdout/stderr逐组独立。已有目录拒绝重启覆盖。

成功必须有实际400条连续有限评估、400条诊断、completed成功、峰值summary一致和最终内存Full形状/有限性检查；不依赖不存在的.pt，也不支持续训。两卡均真实完成20组后自动生成report/comparison40.csv、summary.json、四张400轮曲线、来源/覆盖/更新几何、服务器时间和各条件相对Native的变化。只统计各方法在五条件的胜负与条件内差值，不混排跨数据集绝对精度。

最终最多一个主方法、一个次方法，区分来源重加权与更新尺度。所有Peak标为test-oracle peak，seed1探索不作独立泛化保证。若没有实质超过P04，保留P04并停止指数搜索；完成后由离线报告作最终科学判断，不自动追加训练。
