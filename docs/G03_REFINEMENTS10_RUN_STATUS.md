# P01–P10启动与存储处理记录

当前不是完成后的性能报告。**用户授权清理旧.pt后，原定十组已在新输出根fresh启动，首批P01/P02在GPU0并行运行。尚未完成400轮，不能作候选排序。** 以下保留首次配额失败的完整事实。

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
