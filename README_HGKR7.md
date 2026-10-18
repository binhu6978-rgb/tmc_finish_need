# HGKR-FL 七档六组300轮 Linux GPU 运行包

本包准备六组固定实验，不含已完成的300轮性能结果。没有服务器地址或可用连接时，只交付本包；不能把本地正确性测试说成已经在服务器部署。

## 上传和安装（首次一次）

将 `hgkr7_300_server.tar.gz` 上传到目标Linux机器，解压：

```bash
scp hgkr7_300_server.tar.gz USER@HOST:~/
# 在服务器上：
tar -xzf hgkr7_300_server.tar.gz
cd hgkr7_300
```

USER/HOST替换成自己的SSH账号/地址。包内附原始分区JSON，不附数据集、历史runs、论文、审稿材料、凭据或Git元数据。

使用conda `fd` 环境（没有时创建），参考本机已核验Python3.12 / torch2.7.1 / torchvision0.22.1：

```bash
conda create -n fd python=3.12 -y   # 已有fd时跳过
conda activate fd
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-hgkr7.txt
```

上述CUDA轮子命令来自 [PyTorch官方2.7.1安装说明](https://pytorch.org/get-started/previous-versions/#v271)。根据服务器驱动选同版本支持的CUDA构建，六组共用同一环境。`thop/torchinfo`是原始models包导入需要的依赖，不接入新算法。NumPy只需兼容已缓存分区；实际版本和环境在输出中记录。

把已有CIFAR-100解压目录放在：

```text
data/cifar100/cifar-100-python/train
data/cifar100/cifar-100-python/test
data/cifar100/cifar-100-python/meta
```

若数据已在服务器其他位置，启动前 `export DATA_ROOT=/absolute/path/to/cifar100`，该目录下面直接有 `cifar-100-python`。代码使用download=False，不生成新分区，不覆盖数据。包内分区文件名虽然含_bal，实际是Dirichlet unbalanced α=.3，原始50000索引完整保留，100客户端样本数262–757。

## 一条命令运行

```bash
bash run_7tier_300.sh
```

默认使用当前环境的python3；需要时先 `export PYTHON=python`。默认有两个可见GPU时各占一张（0/1），只有一张时两进程共用GPU0；可事先设置 `export GPUS=0,0` 或 `export GPUS=0,1`。索引相对于CUDA_VISIBLE_DEVICES。

脚本首先做一次真实双进程显存/短程SGD检查：每个进程创建七档模型，模拟10个Full槽位，batch50、本地5epoch和锚定聚合，另检查七档batch128前向。两个进程保留分配器缓存直到对方也完成，测试当前设备能否共存。失败（含OOM）立即停止并保留stderr，不降低batch或模型，不串行替代、不自动重试。

预检查通过后只准备一次客户端/模型分配，然后依次运行：

1. C0 + C1，每组300轮；
2. C3 + C4，每组300轮；
3. C2 + C5，每组300轮。

任何一组失败都会停止同批尚在运行的本脚本子进程，退出整个脚本，不启动后续批次。所有错误日志保留。默认输出 `runs/hgkr7_300_seed1/`；目录已存在时拒绝覆盖。只有六组真实完成300轮且日志核验通过才生成 `SUCCESS.json`。不提供resume、500/1000轮、自动扩展或调参入口。

## 模式与固定协议

|模式|分配|卷积片段选择|同轮客户端位置|
|---|---|---|---|
|C0|近似成本匹配的随机可行分配|无重置|无|
|C1|Full-priority|无重置|无|
|C2|同C1|随机|独立|
|C3|同C1|随机|共享|
|C4|同C1|过去方向不稳定性高者优先|共享|
|C5|同C1|过去方向不稳定性低者优先|共享|

七档实际调用原始get_model_list/ResNet18_cifar：P0(.4,2)、P1(.4,3)、P2(.4,4)、P3(.66,2)、P4(.66,3)、P5(.66,4)、P6(1,2)。slim_idx控制开始缩窄的段，不删除层。共享片段逐层按七份真实形状求共同输入/输出前缀；本模型实测交集与P0形状一致，代码不以该巧合作为假定。Full-only区域由七档实际覆盖计数识别，不能沿用三档区域统计。

原始协议：seed1、100客户端、4:3:3能力、每轮10个不重复参与客户端，5本地epoch、batch50/drop_last、原始CE、SGD .01×.998^round、momentum.5、weight_decay1e-4；原始ToTensor+Normalize，不增强或增设验证集；原始官方test每轮测试七档，batch128。新调度是client-first，明确不同于旧AdaptiveFL的model-first available。设备ID0–39能P0–P2，40–69能P0–P5，70–99能P0–P6。

复用三档已冻结的80%计算/通信双预算及±2%成本匹配容差。依据真实profile MAC和样本数计算成本，不假定profile编号与MAC单调：P2的MAC高于P3。Full-priority只枚举本轮高能力客户端的Full子集（至多2^10），优先Full次数，再优先过去Full机会较少者；随后用剩余预算作简单贪心可行分配。C0用固定128次可行随机分配抽样，保留实际双成本接近C1的候选，再随机选；没有匹配候选时透明回退同一分配。所有选择均在训练前冻结，不使用标签偏好、损失或test成绩。实际差额、退化回退和相同分配轮数记录protocol.json；不将近似预算匹配说成严格相等。

重置定义保持不变：每层floor(.01×共同输出通道数)，每个单位仅包含该输出通道的共同输入前缀；round0不写历史，至少3条旧方向、最近5条、最大年龄20、范数eps1e-12；不稳定性=1−||mean(unit_direction)||。结构性warmup统一，最早第5轮。零方向不提供有效排序证据，不足时共用随机顺序补足，保持每组数量相同。实际每槽位15片段，16320参数；前部小通道层floor后为0。

重置在下发副本中以共享层块的均值和population std采样高斯值。C3/C4/C5共享**位置**，每槽位采样值仍独立，分布定义与C2完全相同。选择与采样均使用隔离CPU Generator，不消耗DataLoader随机序列。所有C2–C5使用同一锚定聚合：

`global_new[j] = global_old[j] + sum(n_c * mask_c[j] * (trained_c[j] - actual_dispatch_c[j])) / sum(n_c * mask_c[j])`

未覆盖坐标保留旧值；Full-only、BN、分类器不直接重置。无重置时调用原始AdaptiveFL覆盖平均，首轮独立初始化也走原始聚合。无Semantic Gate、Reliability、KD或额外损失。全部实现为新文件，旧三档、原始AdaptiveFL及FD入口不改。

## 输出

每组目录 `C0/` ... `C5/`：

- manifest.json：环境、同一冻结protocol及哈希；
- rounds.jsonl：全部300轮七档准确率、槽位、重置、覆盖更新、时间；
- best_full.pt：Full test-oracle peak对应的Full模型，附同轮七档测试结果；
- final_full.pt：第300轮Full；
- completed.json：真实完成轮数；失败时failed.json及组别stderr。

根目录保留各组stdout/stderr、GPU分配、两份显存检查结果、protocol/protocol.json。最终自动输出 `report.md`、`summary.json`、`profile_accuracy.csv`：Full峰值/轮次、第300轮、末20/50轮均值、七档结果、Full参与、计算和通信、重置及选择效果，以及C1−C0、C2−C1、C3−C2、C4/C5−C3等比较。

Full峰值统一标为 **test-oracle peak**；每轮暴露官方测试，不是独立验证选择的无偏性能。不会根据test自动修改比例、预算、排序方向或增加模块。C4不如C3时汇总会直接说明。单seed、并行CUDA非确定性和计算代理估算限制因果结论；本任务只做300轮。

可单独重新生成离线汇总（不训练）：

```bash
python -m experiments.summarize_hgkr7 runs/hgkr7_300_seed1
```

包完整性：`python tools/verify_hgkr7_package.py`。Shell要求Bash>=4.3（wait -n）。运行中保持前台会话；需要后台运行可由使用者自行用已有终端工具，不包含监控平台或复杂调度。
