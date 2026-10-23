# Semantic-A07 V2：1000轮机制运行报告

本次单条 fresh seed1 运行已完成1000轮。Semantic Memory、Reliability Memory和来源级语义融合均实际参与计算，602轮产生非零聚合差异，398轮按固定规则中性退化。Figure 7/8适合展示机制的数据流与执行行为。

## 运行与性能记录

| 项目 | 实际记录 |
|---|---:|
| 完成轮数 | 1000 |
| Full test-oracle peak | 39.3300%，第567轮 |
| Final Full Accuracy | 39.2300% |
| Last20 / Last50 | 38.9725% / 38.8936% |
| Final Full CE | 3.750442 |
| 训练、评价及观测总墙钟 | 20668.36秒（约5.74小时） |
| 峰值显存 allocated / reserved | 1.299 / 1.477 GiB |

服务器根目录为 `/home/hh/TMC_re_e03_scope_components400_20261019/runs/semantic_a07_v2_activation1000_seed1_20261023`，正式记录在 `active/training/a07/active_1000_seed1`。仅运行这一条1000轮轨迹，GPU0单进程，没有新增A07完整训练。

训练源码冻结于 `149c6713c339204ad23fe43952350ae2832b3cd0`；运行配置SHA256为 `0452a08a6fb90e1bd95644c8c10dc1177e89ecdbcbe7c8f7ee13af3a12d0d441`；原分区SHA256为 `41ab2d3486663ce0100e183ab00565de137f91af07632c236c2ddf60d890ba41`。完整真实命令保存在 `results/status.json` 和 `results/worker_manifest.json`。

原七模型、CIFAR-100全量分区、100客户端、4:3:3能力层、10 slots、5本地epoch、batch50和原SGD保持不变；确定性CUDA、Full-only每轮评价、lambda=0.2、首轮Native均按冻结协议执行。CRD按用户要求省略。

## 核验

- 1000条连续有限Full Accuracy/CE、1000条语义记录、1000条来源权重记录和1001条七模型/RNG记录齐全。
- 53轮预先固定反事实诊断完整：1、2、8–24、492–508、984–1000。正、负和中性记录均保留。
- 从空Memory离线逐轮重放1000次决策、历史提交和来源份额，与日志精确一致；所有训练依赖源码文件保持启动时指纹。
- 与旧500轮V2的前500轮在七模型参数、客户端slots、训练RNG、Full Accuracy/CE上完全一致。1000轮是独立fresh运行，不冒充500轮checkpoint续训。
- 最终滚动恢复状态确为1000轮，Full有限且与末轮模型指纹一致，保存Python/NumPy/Torch CPU/CUDA RNG、Adapter和双Memory。只读核验，没有执行恢复训练。模型文件留服务器。
- 启动前：Linux295测试通过（15.98秒），本地295通过（18.64秒）；真实2轮smoke与原V2前两轮完全一致。交付阶段：7项报告/CSV测试通过（4.16秒）。

`completion_audit.json`保存逐项结果。通用训练summary沿用“checkpoint disabled”的文字，指原通用checkpoint/best保存分支；本入口另有每50轮/末轮的单个滚动恢复文件，已实际读取核验，不能根据该旧字段误判未保存。

## Figure 7：Task-grounded evidence and memory calibration

[PNG](figures/figure7_score_formation.png) / [PDF](figures/figure7_score_formation.pdf)

(a) 同一固定训练侧样本的本地SGD前后正确类别概率。每条来源—轮次先对实际参与决策的共同类别作宏平均，再显示各能力层的均值和四分位范围。(b) 实际分数链：当前正向变化 d，历史优势限制 m=min(d,a)，可靠性校准 U=mean(rho*m)。(c) 实际使用的历史年龄和rho分布。(d) 完整1000轮的U轨迹；缺失为间断，真实零值保留，symlog线性区间明确标注。

共有1626条来源—轮次记录、27085条实际使用的能力层—类别观测，来自602个有效轮次。它们包含跨轮复用及共享历史，不能当作1626个独立随机实验。Semantic Memory在20790条类别观测中将m数值限制到d以下；Reliability在16225条观测中进一步改变数值。这证明两个Memory对现有评分计算有实际作用，其预测效用的增量价值需由独立效用实验判断。

Adapter及冻结头继续记录辅助Prototype，**没有进入U或来源份额决策**。图中task probability来自原任务分类输出，不能把这张图描述成Adapter嵌入空间已得到跨客户端效用验证。

建议英文图注：

> **Task-grounded evidence and memory calibration over 1,000 communication rounds.** (a) Fixed-sample task responses before and after local SGD, summarized over eligible source-round observations; markers denote means and horizontal bars interquartile ranges. (b) The recorded progression from current positive change d to historical-advantage-limited evidence m and reliability-calibrated U. (c) History age and evidence-use coefficients for eligible tier-class observations. (d) The complete source-score trajectory, retaining zero scores and gaps for missing evidence. Adapter prototypes are auxiliary observations; task probabilities drive the displayed score.

## Figure 8：Semantic-guided source contribution and realized intervention

[PNG](figures/figure8_aggregation_activation.png) / [PDF](figures/figure8_aggregation_activation.pdf)

(a) 展示U减去区域加权均值与来源份额相对变化的对应关系。这是冻结融合公式的数值执行验证，不能将近线性关系当成语义与效用的经验相关证据。(b) 展示全部1000轮，602轮实际调整、398轮中性退化。(c) 每轮相对同输入融合前状态实际改变的Full坐标比例；(d) 对应参数差异L2。后两项保留中性零值，未以阈值删除小幅后期干预。

中性原因：396轮缺少公平共同类别支持、首轮Native 1轮、只有一个当前能力层1轮。最大来源份额绝对变化为0.630656个百分点；最大实际变化坐标为6902454 / 11424356（约60.42%）；最大融合差异L2为0.00832305。坐标数是数值差异计数，不代表同等规模的功能影响。

| 固定诊断窗口 | 实际调整轮数 | 窗口内平均融合差异L2 |
|---|---:|---:|
| 早期8–24 | 6/17 | 0.00130970 |
| 中期492–508 | 12/17 | 0.00000854 |
| 后期984–1000 | 15/17 | 0.00000315 |

后期调整更常出现，但幅度明显小于早期。图片保留这一真实尺度变化。说明案例按冻结规则取各窗口首次实际激活轮及最小七模型共同区域，分别为12、492、984轮；未按U、Accuracy或CE效用选取。

建议英文图注：

> **Source-contribution adjustment and realized aggregation intervention.** (a) Relative source-share changes against region-centered U under the fixed semantic blending rule. (b) Every communication round, including 602 adjusted and 398 neutral rounds. (c) The fraction of Full-model coordinates numerically changed by fusion. (d) The L2 difference induced by fusion on the same uploads. Neutral rounds and small late-stage interventions are retained. These measurements describe the execution and magnitude of source reallocation.

## 开销和数据

日志累计前测量207.60秒、后测量1199.16秒，共1406.76秒，约为记录本地SGD时间13172.63秒的10.68%；该比例来自代码计时区间，不等于独立对照下的端到端减速。记录额外Backbone前向1918152个样本，Adapter反向2877228个样本，逻辑上传约2.801 GiB；决策消息序列化约37.89 MiB。逻辑字节数为模拟通信载荷计算，不是实测网络流量。

最终1530个Adapter状态共201.75 MiB；能力层Anchor JSON约50.13 KiB，Reliability历史JSON约1.891 MiB。53轮只读反事实诊断累计175.67秒、350096个接收端前向样本。完整恢复文件约248.34 MiB，仅留服务器。

`results/`提供五份机制CSV、完整Accuracy/CE、实际完成标记、manifest、资源、测试日志、原始标量文件索引，以及1286条来源—区域—接收者效用记录和1908条Full反事实记录。共同类别与全部probe类别口径均保留在原始效用表中。probe排除语义评分样本，但仍来自原SGD训练分区，属于训练侧评价。

下载归档SHA256：`5723e2445f67406aa3da618d93d8931c3df824f7dc135a8e142d8aa2340f6b99`。48份下载原文件逐一校验通过，52份原文件被索引；大体量原始逐轮日志留服务器。GitHub不包含模型、原始图像、分区索引或凭据。

## 科研状态与停止

- **已证实：** 冻结V2在真实1000轮轨迹上使用past-only双Memory、生成来源级U，并在合格共享特征区域产生可复核的来源份额和参数变化。
- **仍需区分：** 分数是否预测真实跨客户端效用、同输入调整是否有益、完整训练是否优于独立A07，是三个不同问题。新图覆盖第一段计算链的执行情况；本次没有新增1000轮A07配对参考。
- **保留结论：** 原500轮配对实验、效用图和No-Go报告完整保留。本次激活展示不改写其结果，也不证明Semantic/ Reliability Memory具有性能收益。

本次1条1000轮任务完成；完成检查自动化已删除。没有追加A07训练、随机种子、lambda搜索或其他算法。

## 复现离线图和审计

```bash
python tools/report_semantic_activation.py docs/evidence/semantic_a07_v2/activation1000/results --redraw-csv --output /path/to/new_figures
python tools/audit_semantic_activation1000.py runs/semantic_a07_v2_activation1000_seed1_20261023 --reference500 runs/semantic_a07_pair500_seed1_20261022
```

两个命令均只读训练记录；输出目录须尚不存在。审计命令需要服务器本地最终恢复状态和原始日志，不运行SGD。
