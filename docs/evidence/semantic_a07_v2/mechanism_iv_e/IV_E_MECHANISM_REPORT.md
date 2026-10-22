# IV-E Mechanism Analysis：真实短程结果

**结论：存在有限的类别匹配描述性关联，但没有得到稳定的增量预测证据，也未观察到V2相对同输入A07的聚合效用优势。两张图是机制分析候选图，不是完整语义算法有效性的证明。**

## A—E直接回答

|问题|本次数据支持的回答|
|---|---|
|A. U与真实更新效用是否有意义地关联？|匹配共同类别口径有正向关联：Pearson0.754、Spearman0.791；但只有11个来源—轮次单元、5轮。全probe口径Spearman降至0.327，控制混杂因素后的分组留出预测不优于简单控制变量。不能称稳定、可迁移的增量信息。|
|B. Semantic Memory提供当前观测之外的信息？|从D到M，单变量Spearman由0.264升至0.700；但留出接收者后MSE反而增加，未得到一致的独立信息证据。严格移除Memory会中性退化A07，这只证明工程依赖。|
|C. Reliability改善证据判断？|M到U的单变量排序继续提高，去掉rho的聚合对照均值更差，但区间宽、跨留出方式不一致，且rho同时减弱了调整幅度。尚不能区分“可靠性有信息”与“更保守的调整更安全”。|
|D. V2相对同输入A07有益？|未观察到优势。5个非中性轮、15个独立接收probe的平均CE改善为−2.290×10^-5，95%探索性分组区间跨0；5个改善、10个恶化。|
|E. 已知/未知？|已验证真实只读数据流、固定U、实际份额变化、接收端反事实及原V2轨迹不变。未知成熟阶段、未见客户端/分区、多seed、400轮Full Accuracy收益及双Memory各自的科研必要性。|

## 实际源码、运行与保护

开始时本地和服务器HEAD均为 `6f7a60a700a11f12de2460e5bbe0e99a40e775a8`，分支 `feature/semconsfl-semantic-a07-v2`。原A07/V1/V2核心、SGD、分区、模型、历史实验和报告均未修改；只增加诊断、统计、绘图及测试。旧未跟踪文件保留。没有新建服务器项目。

真实诊断源码：`5ca91c90fb15d494ad81444c8696ae4fc873c959`；后续统计/绘图与测试提交 `19f3cb00296dd5975eecc33fc2765434967a634e` 没有改变已执行的诊断或V2公式。原始manifest记录实际SHA和argv，不能用最终文档提交代替训练SHA。

现有20轮日志有U、类别、rho和份额，却没有上传/下发张量或独立接收端CE；没有据此推测效用。实际补做**一次2轮pilot和一次20轮V2观测重放**，共22轮、220个真实本地训练slot，完整CIFAR100、原100客户端/4:3:3/七模型/10slot/5 local epochs/batch50/seed1/原分区/原确定性协议，Full-only原评价仅用于轨迹核对。没有400轮，没有额外seed，没有PT。

服务器原目录 `/home/hh/TMC_re_e03_scope_components400_20261019`；conda hh、PyTorch2.7.1+cu118、物理GPU0单进程共享，不触碰其他训练。数据 `/home/hh/ICML_finish/data`，原分区SHA `41ab2d3486663ce0100e183ab00565de137f91af07632c236c2ddf60d890ba41`。

真实命令（已执行，不需重复）：

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
/home/hh/miniconda3/envs/hh/bin/python -u tools/run_semantic_a07_iv_e.py \
  --rounds 20 \
  --dataset-root /home/hh/ICML_finish/data \
  --prepared runs/capability_final40_400_seed1_20261021/prepared \
  --output-root runs/semantic_a07_iv_e_20261022/diagnostic_20 \
  --reference runs/semantic_a07_v2_short_20261022/a07/active_20_seed1
```

pilot使用同一命令的2轮预算和独立 `diagnostic_2` 输出，先通过才进入20轮。两条均exit0；七模型初始/逐轮权重、每slot上传、Python/NumPy/Torch CPU/CUDA RNG、Full Accuracy/CE与原V2相同预算轨迹**逐字段完全一致**，完成标记真实。官方测试集未进入probe、U、统计规则或对照构造。

## Figure7：固定U与真实跨来源学习增量

复用现有 `ce_vector`、固定训练侧视图、区域前缀和原A07系数。对每轮2..20、每个多来源共享Feature区域：

`delta(g,r) = sum_{i in tier g, covers r} alpha_A07(i,r) * (upload_i - actual_dispatch_i)`。

将此**来源级实际SGD贡献**注入其他能力层接收者的真实profile下发模型副本，正效用为CE_before−CE_after。重复slot保留为实际训练更新，不能伪造成独立客户端。接收者按固定hash选取，不按U、方向或CE筛选。首轮不进Figure7，避免独立初始化混杂。

probe取接收者自己的训练分区，每类2个、最多16类，排除V2语义测量的全部样本索引，固定视图、batch50且前后相同。**它独立于语义测量，但仍被原SGD训练使用过，不是训练外验证集。** 模型BN无运行统计，逐类CE来自同一整批前向，不拆类重新组成BN batch。

- 原始436个来源—区域—接收者配对均保存；348个没有有效U。
- 88个有U；其中4个与接收probe没有共同类别，主分析记missing，不填0。
- 主分析84个配对：54个改善、30个恶化；37个不同物理来源客户端、29个不同接收者。
- U仅有**11个独立取值单元（来源、轮次）**，来自第7、12、13、15、19轮，不把84配对当84个独立语义分数。
- 主散点先在区域内平均接收者，再在同来源—轮次间等权平均区域。共同类别不是重算U：仍使用原始来源级U，仅将效用限定到实际可评价的共同类别。21/84配对只含一个共同probe类别，存在较高测量噪声。
- 主分析中没有负向来源LOO余弦配对，不能据此验证高方向冲突情况下的效用。

固定类别匹配分析：

|分数|定义|Pearson|Spearman|benefit AUC|
|---|---|---:|---:|---:|
|D|当前正向task概率变化宏平均|0.001|0.264|0.708|
|M|加入过去其他能力层优势，mean(min(d,a))|0.668|0.700|0.917|
|U|原V2，mean(rho×min(d,a))|0.754|0.791|0.958|
|置换U|同轮层间循环置换，保留分布和有效性|0.295|0.536|0.667|

U的探索性多维cluster bootstrap区间为[0.250,1.000]。**5个轮次不足以保证95%名义覆盖率；不能因区间为正或AUC较高宣称有效。** 使用全部固定probe类别后，U Pearson仅0.161、Spearman0.327、AUC0.611，Spearman区间[−1.000,0.884]。Figure7同时展示这两个口径，避免只展示较强关系。

固定U区间[0,.1)、[.1,.2)、[.2,1]分别只有2/5/4个来源—轮次单元，平均共同类别效用为−0.023955、0.006315、0.025640。误差条为1000次固定seed的round/physical-source/receiver多维Poisson分组bootstrap探索性区间；不是FL随机种子置信区间。

### 混杂因素及留出分析

固定ridge=1，无超参数选择。控制实际注入L2、方向余弦及缺失、能力层、接收profile、来源profile构成均值、样本质量、区域大小、训练标签分布JS、来源自身学习收益和接收基线CE。每个来源—轮次的重复配对总权重为1，标准化仅在训练fold完成。

|预测器|留整轮MSE|留物理接收者分组MSE|
|---|---:|---:|
|简单控制变量|0.00378630|0.00415933|
|控制+D|0.00416635|0.00415978|
|控制+D+M|0.00413587|0.00421816|
|控制+D+M+U|0.00397086|0.00426822|
|控制+D+M+置换U|0.00457121|0.00429925|

完整分数相对控制变量MSE分别**增加4.87%和2.62%**。M/U有些逐级改善只出现在整轮留出，不在接收者留出复现。后者允许同来源—轮次的其他接收者进入训练，不能冒充未见更新验证。

低层只有1个来源—轮次单元且效用为负；中/高层各5个，层内Spearman分别0.5/0.9，但样本不足。能力、profile和类别难度仍可能混杂；JS只描述标签频率，不消除输入难度或概率校准差异。当前统计不支持“Semantic Memory与Reliability分别提供稳定增量价值”。

## Figure8：同输入聚合与独立Full任务评价

每轮同一组真实上传计算原A07和V2。V2公式、lambda0.2、Feature作用域、BN原A07和classifier Native均不改。先验选择每层一个未参与当前轮的物理接收者，共3个Full探针；20轮涉及7个不同接收者。各反事实加载同一Full架构、同样probe和BN batch，记录相对旧Full效用与相对A07的配对效用。

控制分为两类，不能混称：

1. **严格去掉SM或RM**：原V2缺少必要证据时精确回退A07。两种控制复用同一已实测A07模型的CE，明确标记 `evaluation_reused_from`，没有假称新增GPU评价；这只验证功能依赖。
2. **匹配有效性mask的数值消融**：去掉历史优势项用mean(rho×d)；去掉rho用mean(min(d,a))。保留完整V2同一有效类别与历史合格mask，包括已有rho的来源选择。因此前者只隔离anchor优势项，**并非彻底删除所有历史数据流**，不能用它证明整套Memory的必要性。

另有固定层间置换U和不使用任何语义、仅匹配完整V2全局更新L2的A07方向控制。置换在4个双来源轮与V2的差异L2相差不超过0.31%，但第19轮三来源时为V2的1.60倍；不声称所有轮严格等干预。纯尺度控制的全局更新L2与V2最大绝对误差7.9994×10^-8，但它会缩放更广参数作用域，不是完全相同方向的语义消融。

共有360条真实反事实CE记录（20×3×6）和120条明确标注的严格中性复用记录。V2仅在7/12/13/15/19轮调整，75%轮次中性。区域份额最大变化仍约0.513个百分点；Figure8(a)用同一Feature面积分母汇总各层份额变化，缺席层贡献变化为0，三层曲线逐轮和为0。

以下为5个实际非中性轮的15个接收probe；正值表示比A07降低CE：

|反事实|平均CE改善×10^-5|探索性95%区间×10^-5|改善/恶化probe|平均参数差异L2|
|---|---:|---|---:|---:|
|A07|0|[0,0]|0/0|0|
|V2|−2.290|[−35.609,25.921]|5/10|0.002272|
|去anchor优势项（匹配mask）|5.254|[−47.777,53.458]|7/8|0.002445|
|去rho（匹配mask）|−6.567|[−40.623,23.209]|5/10|0.002603|
|置换语义|−2.077|[−57.459,32.178]|11/4|0.003062|
|A07全局L2匹配|−1.173|[−15.707,22.148]|4/11|0.000375|
|严格无SM / 严格无RM|0|[0,0]|0/0|0（既有中性路径）|

全部20轮V2相对A07平均改善为−5.725×10^-6。V2没有优于A07或置换语义的明确证据。去rho更差可能只是更大幅度干预；去anchor项均值略好也不足以证明其优越。**份额变化和参数修改数量不能替代效用证据。**

## 工程核验、成本与复现

- 本地conda fd **280 passed（18.69s）**，Linux conda hh **280 passed（16.78s）**：10项IV-E、25项原V2、245项旧专项。
- 全部2/20轮精确轨迹回归通过。单元检查涵盖探针索引分离、固定视图与RNG、真实前缀注入、重复身份、上传/下发/主模型不变、CE符号、原分数不变、归一化、严格Memory缺失回退、分组统计。
- Linux从服务器原始目录独立重算，与本地221个统计数值字段最大绝对差9.16×10^-16，见 `verification.json`。
- PDF各1页，300dpi PNG与矢量PDF均生成，已用Poppler重新渲染检查，无遮挡数据点或裁切标题。图脚本不会挑选样本或拟合评分。
- 20轮诊断计时块合计21.158s，接收端额外前向样本37,056；这个时间不含前置A07影子重算、保存dispatch等全部开销。进程墙钟313.868s，峰值allocated/reserved为1,395,918,848/1,583,349,760字节；共享GPU时不能把跨运行墙钟直接比较为算法开销。
- 若真实在线实施这些效用查询，额外区域贡献下行float32代理量约1.620GB，逐类反馈代理83,712字节，还需要请求、回传与隐私协议。这里只是服务器模拟器的离线观测，未产生这些网络消息，接收探针从未反馈聚合或Memory。

原始记录在服务器 `runs/semantic_a07_iv_e_20261022/`；只读归档 `raw_evidence.tar.gz` 含36份标量/manifest/完成记录，无模型PT、图像或凭据。原始上传只在内存中存在，不能从CSV重建过去的更新张量；复核数值统计不需要重训。

从项目根目录复现统计和图（不训练）：

```bash
python tools/analyze_semantic_a07_iv_e.py <已解压>/runs/semantic_a07_iv_e_20261022/diagnostic_20 \
  --output docs/evidence/semantic_a07_v2/mechanism_iv_e
python docs/evidence/semantic_a07_v2/mechanism_iv_e/plot_figures.py \
  docs/evidence/semantic_a07_v2/mechanism_iv_e
```

环境依赖沿用项目PyTorch/NumPy/pytest，绘图使用Matplotlib；不需要新联邦学习框架。

## 论文候选图说明与使用决定

**Figure7 caption (candidate).** Relationship between fixed source-level V2 potential complementarity and recipient training-probe CE utility along one early trajectory. Each point represents one source-tier/round unit, averaging regional interventions. Matched-class and all-probe-class views are both reported. Fixed-bin intervals use exploratory multiway cluster bootstrap with only five effective rounds. These associations do not establish incremental predictive value or global performance gains.

**Figure8 caption (candidate).** Source-share changes and paired Full-model counterfactual utility from identical real uploads. Feature-area-weighted share changes sum to zero. Utility contrasts use the same nonparticipating recipients and fixed batches. Memory-term ablations retain the full-V2 validity mask; shuffled scores preserve score distributions, while the L2-only control preserves the A07 direction. Error bars represent exploratory cluster intervals, and per-round ranges show recipient heterogeneity.

两图都获得真实数据支持，可以用于诚实呈现“工程激活与关联/效用的区别”。**当前不支持将其写成语义互补已经提高全局聚合、双Memory分别有效或完整SemConsFL优于A07的正面机制证据。** 本次没有启动400轮，也不据此修改V2公式或追加机制搜索。
