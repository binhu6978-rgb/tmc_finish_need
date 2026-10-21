# P01–P10启动与存储处理记录

当前不是完成后的性能报告。**最新用户要求为GPU1从头训练、不保存.pt；之前38轮已停止并保留日志，不用于本次正式比较。新一轮十组采用完全fresh seed1、仅保存标量与最终内存有效性检查。** 以下按时间保留首次失败、清理与早期启动事实；最新运行目录和版本以文末记录为准。

## 已完成的准备

- 训练源码已提交推送：`0da1ed2cb38953476bf3400cf025876485f5bf7e`，分支`feature/adapterfl-six-methods`。
- 十个固定算法与公式：[G03_REFINEMENTS10_DESIGN.md](G03_REFINEMENTS10_DESIGN.md)，配置：[g03_refinements10_400.json](../configs/g03_refinements10_400.json)。
- 本地及服务器hh环境各111 tests通过；真实七ResNet形状合成GPU检查通过，服务器峰值reserved1076MiB，P01/P02范数相对匹配误差约3.26e−8/4.43e−8。这是正确性检查，不是性能实验。
- 十条真实Linux命令保存在[evidence/g03_refinements10_launch/commands.txt](evidence/g03_refinements10_launch/commands.txt)。

## 实际启动及失败

唯一服务器代码目录：`/home/hh/TMC_re_e03_scope_components400_20261019`。
输出根：`runs/g03_refinements10_full400_seed1/`。

父PID3399638启动首批P01/P02，子PID3399672/3399673，物理GPU0并行两进程；GPU1其他训练未干扰。
两组都实际评估并写入第1、2轮CSV和diagnostics，但第2轮保存新best checkpoint时出现：

```text
PytorchStreamWriter failed writing file
OSError: [Errno 122] Disk quota exceeded
```

原错误完整保存在[失败证据目录](evidence/g03_refinements10_launch/failure/)。主队列state=`failed`，failed=`P01,P02`，completed为空；P03–P10没有启动。没有OOM、自动重试或更改公式。

直接SSH核对：底层文件系统还有123G可用空间，但**hh用户100G硬配额已用满，quota显示102400M/100G**。因此仅查看`df`空闲空间不能解决该账户写入失败。

## 已有文件的边界

服务器直接加载两组latest.pt和best_full.pt均有效，分别仍对应第1轮；第2轮日志已写但best_full.pending写到中途。因此当前日志和checkpoint不处于同一个轮次，不能冒充已经完整保存2轮，也不能直接按原恢复检查续训。

本地只读核对两组初始化七模型、训练/确定性协议与B00-D/G03一致，完整50000训练样本、GPU0、仅Full评估，首轮聚合相对Native差异为0。第二轮诊断中的P01/P02范数匹配符合既定规则；这不足以评价400轮Accuracy收益。

全部旧B00-D、G03及其他历史结果、当前失败文件均保持原状。没有删除旧权重、移动数据、覆盖失败目录或再次启动队列。完成检查自动任务已暂停。

## 首次失败后的存储处理

用户于2026年10月21日明确授权删除当前项目runs下全部.pt。删除前在服务器直接核验B00-D/G03真实400轮latest/best；严格限定绝对runs目录、不跟随符号链接，删除59个.pt，逻辑总大小2.51426GiB，全部标量保留。quota从102357M降为99782M，100G硬限制下约有2.56GiB可用。

只修改历史对比的读取条件：B00-D/G03使用完成标记与完整400轮有限值CSV，不再要求已授权删除的权重。新实验仍须实际400轮latest/best与日志一致才成功。新回归及原算法/评估检查在本地和Linux各112tests通过，聚合、SGD、模型、数据及确定性协议没有变化。

## 清理后实际后台重启

- 冻结源码：`7d686b6993e4d81306508f75788733ffdc879cd4`。
- 新输出根：`runs/g03_refinements10_full400_seed1_after_cleanup_20261021/`。
- 父PID：3586425；首批P01/P02子PID：3586431/3586432，GPU0实际并行2进程。
- 已核对首批全部七档初始化、原数据分区和训练/确定性配置与B00-D/G03一致，完整50000训练样本、仅Full评估。
- 五批仍为P01/P02、P03/P04、P05/P06、P07/P08、P09/P10；不重跑历史方法，不从错位2轮checkpoint恢复，不根据该2轮测试数据改参数。
- 首次失败目录原样保留（其中.pt按用户要求删除）；新的权重正常保存，未安排继续删除新checkpoint。

[清理清单、删除前历史核验及真实启动证据](evidence/g03_refinements10_launch/after_cleanup_startup.json)。用户要求无实时监控；仅安排2026年10月21日18:00完成后检查，仍运行则静默顺延，全部完成后再制作真实400轮统一对比与研究决策。

## 最新覆盖要求：GPU1全部从头、不保存.pt

用户在GPU迁移准备后明确要求“从头训练，不需要保存pt文件”，因此不执行第39轮恢复。先前P01/P02各38轮由用户取消，日志保留；其4个已无用途的.pt删除。新输出根为`runs/g03_refinements10_full400_seed1_gpu1_nopt_20261021/`，十组均从原seed1初始化开始，GPU1每批两组，不再使用continue-suite或method-resume。

新增默认关闭的`--no-save-pt`，固定P配置启用；旧baseline/其他配置仍默认保存。只跳过模型/RNG checkpoint复制及写入，CSV、Full CE/Accuracy、test-oracle peak、诊断、manifest、summary和完成标记正常保留。第400轮直接检查内存Full参数形状/数量和有限性，写`final_model_check.json`，不保存任何模型权重或.pt/.pending checkpoint。

正式成功依据改为：实际1–400轮有限Accuracy/CE、400条聚合诊断、completed成功、标量summary与峰值轮次一致、最终Full内存检查通过。无checkpoint模式不支持断点续训，不能伪造或声称存在best/latest模型；峰值只有标量记录。

115项专项检查通过，额外验证关闭保存后两轮最终Full权重、Accuracy/CE与训练RNG均与正常保存模式完全一致。没有改算法公式、SGD、模型、原分区或400轮协议。

GPU1 fresh/no-PT已实际后台启动：冻结源码`d3ced897482ff34062d422a47efadeda22612ae9`，父PID3592112，首批子3592118/3592119。Linux115tests通过；首批manifest确认save_pt=false、CUDA_VISIBLE_DEVICES=1、无method-resume、七初始化与原分区/训练配置匹配参考、完整数据且仅P6测试。启动证据见[evidence/g03_refinements10_launch/fresh_gpu1_nopt_startup.json](evidence/g03_refinements10_launch/fresh_gpu1_nopt_startup.json)。

全部新候选使用GPU1、旧参考使用同型号GPU0，记录这一物理设备差异，不声称已完成跨卡bitwise验证；与旧方法的墙钟成本差异还包含取消checkpoint I/O，不能将其全部归因于聚合算法加速。按用户要求不发送本次启动回复，完成后再交付原任务要求的真实结果。
