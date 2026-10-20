# E03 / L04 受控验证

只运行三组正式实验，各 seed=1、400轮，GPU0串行：

| ID | 聚合 |
|---|---|
| R0 | 原始 M2，λ=4，REF只读诊断 |
| R1 | 原始 L04：所有轮次 classifier.* 用 Native，其余参数用 M2 |
| R2 | 第一轮严格 M2；第2–400轮与 L04 完全相同 |

固定服务器项目目录为 `/home/hh/TMC_re_e03_scope_components400_20261019`。
在该目录更新代码、启动实验，不再创建服务器项目或 checkout。
新实验结果保存于该项目的 `runs/e03_l04_control400_seed1/`，原结果全部保留。

## 数值执行控制

旧代码设置 Python、NumPy、Torch CPU/CUDA seed，但不强制确定性 GPU 算法。
本地真实首轮槽位1（物理客户端76、P3，462样本）的两次5-epoch SGD：
还原同一权重和 RNG，首次输入和前向 logits 完全相同，首次反向已有33个梯度张量不同，
梯度差 L2=0.0015460，客户端最终权重差 L2=1.3382653，结束 RNG 仍相同。
这将分叉定位到 GPU 反向数值计算，而非 M2/L02 聚合差异或客户端抽样变化。
首轮10槽位重复中6槽最终权重不同；并未定位到具体 cuDNN 内核实现。

`--deterministic-execution` 默认关闭，保留旧代码执行行为。开启时：
严格 `torch.use_deterministic_algorithms(True, warn_only=False)`；cuDNN deterministic=True、
benchmark=False；cuBLAS `:4096:8`；TF32关闭。遇到不支持的非确定性操作直接报错，
不静默回退。执行策略写入 manifest/checkpoint identity。
这是三组共同采用的新数值执行环境，不是修改 SGD 或 FL 数学定义。
不能把新结果与旧 Linux E03 或旧本地 REF 当作严格配对。

Full-only 仍使用既有 Python/NumPy/CPU/CUDA RNG 隔离；不修改测试函数。
DataLoader shuffle、完整分区、七模型独立初始化和聚合后全部同步不变。
本地真实 GPU REF/L02 各3轮逐张量、评价、RNG和槽位精确比较。
预检是固定预算的短程检查，不标记为正式400轮成功。
不假定跨GPU、CUDA版本、操作系统位级一致。用户后续明确要求只跑三组，
不扩展诊断，不实时监控；服务器不追加其他方法或 REF/L02 性能重跑。

## 分类器保护

沿用 `Local12Hooks(l04)`。唯一新增参数
`--classifier-protection-start-round 2` 实现 R2；默认1保持旧L04。
`rounds_seen` 已随 method_state 保存和恢复。第一轮 R2返回与原M2逐张量相同的结果，
第2轮后分类器覆盖归一化使用 Native，其他键仍用原 M2输出。
方向评分、λ、实际 SGD 增量及下发融合规则均未重写。

## 执行与结果

在 conda hh 下：

```bash
python -u tools/run_l04_control.py --execute \
  --dataset-root /home/hh/ICML_finish/data \
  --output-root runs/e03_l04_control400_seed1 \
  --gate docs/evidence/l04_repro_preflight/passed.json
```

脚本串行调用原训练入口，完整400轮并核验 latest/best/CSV 后进入下一组。
失败停止后续任务，保留日志和权重，不自动重试。全部完成后自动生成 report/ 下
comparison.csv、summary.json、逐轮曲线CSV、Accuracy/CE图和报告。
不读逐轮日志实时监控，只在结束后汇总。

解释重点是 R1/R2相对新R0的 Peak、Final、Last20/50和CE；分开首轮与后续分类器变化。
峰值为官方测试集 test-oracle peak；单seed探索不能证明独立泛化收益。
结合旧L03/L05只作参数作用域讨论，不能把classifier保护直接称为语义知识保护。
若无一致优势，停止L04调参；最多这三组，不自动追加。
