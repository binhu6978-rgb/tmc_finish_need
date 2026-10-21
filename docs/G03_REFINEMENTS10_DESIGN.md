# G03固定十组400轮：来源、尺度与作用范围

本批仅P01–P10，每组400轮、seed1。复用已完成B00-D和G03作为参考，不重跑旧方法。
原七模型、完整CIFAR-100与缓存Dirichlet α=.3分区、100客户端、10 slots、5 local epochs、batch50、原SGD及确定性CUDA/Full-only隔离RNG协议不变。
**全部第1轮严格Native，第2轮开始对应候选。** 不新增本地训练、反馈、通信、方向风险评分、Q、Memory或模型模块。

## 共同记号与覆盖

复用`native_group_aggregation.py`、原`Aggregation_AdaptiveFL`、真实九区域几何及原prefix切片。
对每个区域r，只使用实际覆盖slot I_r。slot i上传U_i、实际样本量n_i、物理客户端c_i、实际模型档位p_i。
原能力层为物理ID [0,40)、[40,70)、[70,100)，不根据上传模型大小猜测能力。
原生结果B先完整调用原聚合；无上传坐标保持原Full W。

令实际参与能力层集合为G_r，质量M_g=sum_{i in tier g}n_i；这是样本-slot质量，不是更新质量评分。
G03在组内按n_i平均，层间等权。为复用相同浮点计算顺序，服务器有效权重总质量保持与Native相同。
G03只有一个参与能力层时与Native权重成比例，直接返回Native，不额外制造舍入差异。

本批遵循用户要求的**单能力层中性回退**：P03–P10的来源重加权在只有一个实际能力层时保留Native，包含P06/P07；P06去重和P07层内档位均衡仅在至少两个能力层参与该区域时应用。
P01/P02先遵守这一来源权重规则，但最终是全局范数控制，可能缩放已覆盖的单能力层区域。这是全局尺度对照的预定作用，不能再把这些区域冻结，否则无法整体匹配L2。

## 十个固定规则

### P01：Native方向、虚拟G03尺度

同一批上传、同一Full锚点W计算原Native B和原G03 H：

`V = W + (||H-W|| / ||B-W||) * (B-W)`。

只写本轮实际覆盖坐标。最终不使用G03的来源权重；诊断中`virtual_g03_slot_shares`是虚拟证据，`applied_slot_shares`仍是Native。

### P02：G03方向、Native尺度

`V = W + (||B-W|| / ||H-W||) * (H-W)`。

保留G03的来源贡献与更新方向，只匹配同输入Native整体更新L2。
P01/P02使用float64累积范数，结果按原float32权重存储。源与目标均零时保持Native；源为零而目标非零时没有可缩放方向，明确报错并保留日志，不用任意方向或epsilon制造更新。

### P03/P04/P05：只改变层间目标份额

对实际存在层的目标份额π_g归一化：

- P03：π_g ∝ sqrt(M_g)。
- P04：π_g ∝ 1/sqrt(M_g)。
- P05：π_g ∝ [4,3,3]_g，缺席层不参与归一化。

组内仍为n_i/M_g，有效slot权重与`π_g*n_i/M_g`成比例。正样本量保证M_g>0，不用伪计数；不存在组不参与。相同组质量时P03/P04直接复用G03权重，保证同输入退化正确。

### P06：只在能力层内部去重物理客户端数据质量

在当前区域内，同一物理客户端c有k_c个实际覆盖slot，先平均这些独立上传，不合并SGD训练；该客户端样本量n_c在其能力层内计一次。
组内slot权重与`n_c/(k_c*sum_unique_clients_in_tier n_c)`成比例，实际参与能力层间仍等权。
某重复客户端的其他slot不覆盖该区域时不计入k_c。保留单能力层Native回退。

### P07：能力层→实际模型档位→样本的分层均衡

层间等权；每层内部对实际覆盖该区域的模型档位等权；同一能力层/档位内部保持n_i加权。
slot份额为`1/(实际能力层数*该层实际档位数) * n_i/(同层同档位样本-slot质量)`。
不叠加P06去重，不生成缺席层/档位；单能力层按预定中性要求回退Native。

### P08/P09：只改变固定区域作用范围

P08只在**静态全部七模型共同覆盖区**采用G03，其他区域Native。
P09反之。静态共同区与当轮所有上传的交集不同，不动态扩大/缩小定义。
对于完全相同输入，P08共同区=P09区外=G03对应部分，其余严格Native；单层/无覆盖中性。

### P10：只让classifier保持Native

所有卷积、特征及BN按G03；仅`classifier.weight`和`classifier.bias`完整采用Native。
层分组与权重不变，训练和分类器可训练性不变。
区域中混有classifier与其他cell时只跳过classifier cell，不能将整个区域回退Native。
日志`applied_slot_shares`指非classifier cell，classifier单独为Native，不将其解释为语义保护。

## 日志与比较

每个候选都只读计算同输入虚拟G03，记录Actual/Native/G03更新L2、相对Native/G03差异L2、更新余弦、各区域实际权重份额、能力层/物理身份/模型档位、无覆盖及首轮状态。
余弦只作最终几何诊断，不进入权重或训练决策；全部诊断不消耗RNG。

自动报告同时保留历史B00-D和G03，每组400轮Full Accuracy/CE、test-oracle peak及轮次、Final、Last20/50、相对两参考的差值、成本和能力来源贡献。
P01/P02只匹配各自当前上传上的反事实目标；不同训练轨迹后续上传不同，不要求P01和历史G03逐轮同范数。
不能仅凭来源方向变化或峰值提高就断言泛化收益，优先判断峰值与末段均值是否一起改善。

## 运行与边界

仅使用服务器既有项目`/home/hh/TMC_re_e03_scope_components400_20261019`。
输出`runs/g03_refinements10_full400_seed1/`，各组`pXX/PXX_full_only_400_seed1_deterministic/`，不覆盖历史结果。

复用现有双进程启动脚本，新增可选固定config参数；原G01–G10默认不变。

```bash
conda activate hh
cd /home/hh/TMC_re_e03_scope_components400_20261019
nohup python -u tools/run_native_groups10.py \
  --execute --config configs/g03_refinements10_400.json \
  --dataset-root /home/hh/ICML_finish/data \
  --output-root runs/g03_refinements10_full400_seed1 \
  > runs/g03_refinements10_launcher.log 2>&1 < /dev/null &
```

批次固定P01/P02、P03/P04、P05/P06、P07/P08、P09/P10，GPU0最多两进程。只在批次边界查可用显存；资源不足仅改变调度或停止，绝不改训练协议。任一失败保留错误/权重，等同批另一独立进程结束后停止后续批次，无重试。输出目录存在即拒绝重复覆盖。

不实时逐轮监控，不读取中间Accuracy来改顺序、参数或追加训练。全部完成后离线汇总，最多保留1–2候选，单seed官方test-oracle探索，不声称独立验证。完成本次十组后停止。

## 用户授权清理旧权重后的重启（2026-10-21）

首批因账户配额失败后，用户明确要求删除本项目runs下全部.pt。已在严格限定的绝对目录内删除59个.pt，释放逻辑大小约2.514GiB；日志、CSV、manifest和完成记录保留。删除前直接核验B00-D/G03最新与峰值checkpoint均对应真实400轮，保留核验及删除清单。

历史B00-D/G03比较仅依赖已有完成标记与400轮有限值CSV，不再要求被授权删除的权重文件。**本批新实验的400轮完成检查仍要求实际latest/best checkpoint，不降低成功条件。** 聚合、模型及训练代码不变。

失败目录保持原状，新输出根固定为`runs/g03_refinements10_full400_seed1_after_cleanup_20261021/`，十组全部从seed1初始状态开始，不从不对齐的2轮日志恢复，也不重跑历史方法。命令同上，仅替换`--output-root`为这个新根并使用独立launcher日志。
