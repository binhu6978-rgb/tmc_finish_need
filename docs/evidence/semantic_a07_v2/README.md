# V2 短程证据

- `summary.json`：服务器生成、本地从归档重新计算后逐字段完全相同的汇总。
- `pytest_local.log`、`pytest_linux_final.log`：两端各270项测试通过；25项V2、245项旧专项。
- `synthetic_activation_local.log`、`synthetic_activation_linux.log`：真实七模型形状的构造输入激活检查；不是实际训练收益。
- `short_evidence.tar.gz`：79份原始标量记录、CSV、manifest、状态、资源和日志。包含三模式各2/20轮的完整证据；不含模型权重或训练图像。原始路径为归档内 `runs/semantic_a07_v2_short_20261022/` 和 `runs/semantic_a07_v2_evidence_20261022/`。

解压到独立临时目录后可重新检查，不执行训练：

```bash
python tools/report_semantic_a07_v2.py <解压目录>/runs/semantic_a07_v2_short_20261022 --output <新文件>.json
```

训练源码SHA `863087e0bcead3e73f15f9317aa9fe79da1b0c92`，不能用后续证据/文档提交替代。正式科研解释见 `docs/SEMANTIC_A07_V2_RESULTS.md`。
