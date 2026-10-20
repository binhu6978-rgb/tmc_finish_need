# P01–P10启动状态：hh账户磁盘配额阻塞

当前不是完成后的性能报告。**十组400轮尚未完成，队列已失败停止，不能作候选排序。**

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

## 等待解决

需要用户扩容hh配额，或明确指定允许清理/迁移的旧文件，或提供另一处可写结果存储路径。建议为本批至少准备2GB账户可用额度，并考虑其他正在运行任务的增长。

存储及重启安排明确后，保留本次失败证据，再按原十组固定配置进行独立完整运行；不利用这2轮测试结果改参数。当前不能报告是否突破G03、不能推荐新候选，也没有生成伪造的400轮曲线。
