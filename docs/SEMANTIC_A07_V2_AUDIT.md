# V2 实现与科学边界审计

V2 是新的 task-grounded potential complementarity 假设。基线为 v1 的 `9cfe4a9686dbd337a4ffd848f1cadeb35fba30f9`；旧 No-Go 结论和证据不改。本次不执行400轮，也不把短程可运行等同于机制有效。

## 真实数据流

|模块|实际作用|不代表什么|
|---|---|---|
|原本地 Task CE/SGD|仍由原 `LocalUpdate_AdaptiveFL` 执行|没有新增 Backbone 辅助损失|
|Adapter、冻结Head、Prototype|复用 v1，在 detach 特征上拟合 Adapter 候选；三个固定头17/29/43保留辅助概率、原型和稳定性观测|**不进入 U 或聚合权重**，未证明 Adapter 带来聚合收益|
|主模型 task probability|固定训练样本、同一增广与BN批次下的训练前后正确类概率|类别坐标相同不保证输入分布或概率校准相同|
|TierSemanticMemory|过去能力层/类别参考，决定相对其他来源的优势 a|不是全局有益知识标签|
|TierReliabilityMemory|过去独立来源支持、年龄、变化决定可用性及 rho|不是旧 S1 可靠性解释，也不是个人信誉|
|V2来源融合|同一 U 仅调整多能力来源共享 Feature 的 A07 份额|不是逐层或逐参数语义归因|

新包继承旧 Memory 以保留辅助记录，但**决策只读取新增的 tier/category task 状态**。旧客户端/profile S1 历史不进入 rho。移除任一新增 Memory 将因缺证据而中性；改变参考或稳定性会改变实际 U，专项测试覆盖这一数据流。

## A07 保留方式

`semantic_loop.Hooks.aggregate` 首先直接调用原 `Final40Hooks.aggregate`。其真实区域来自 `build_regions`，层内按实际覆盖 slot 样本量，层间按 P04 反平方根质量；重复上传保持为真实 slot。Feature/BN 原 A07，分类器 Native，单能力来源 Native，首轮 Native，未覆盖保留旧 Full。

V2 所有中性条件直接返回原结果对象，不用代数等价式重算。非中性时只克隆该结果并使用原 `source_weights` 和 `write_weighted_region` 重写合格 Feature cell。权重采用 `w'_i=w_i*pi_new(g)/pi_a07(g)`，因此组内相对样本权重不变。BN affine由真实模块类型识别并排除；classifier严格保留原结果；Full-only与单能力来源不改。融合使用真实上传参数状态，**没有替换为旧Full锚点增量**。

`capability_final40.py`、`native_group_aggregation.py`、`semcons_scope.py`、整个旧 `semcons_a07/`、原训练循环、本地更新、模型、原聚合、历史配置和旧报告相对基准没有修改。

## 简洁伪代码

```text
for round:
    read frozen past tier references, past reliability, Adapter bank
    run original client sampling and original seven-profile local SGD
    measure paired pre/post task responses without changing SGD/RNG/BN/gradients
    deduplicate physical identities for semantic support only
    compute same-class fair U from past other-tier references and past rho
    base = original_A07(unchanged original uploads)
    if first round, lambda=0, or evidence missing: next = base
    else for eligible shared Feature region:
        get original A07 tier shares and within-tier slot weights
        blend shares once using fixed lambda=0.2 and (1+U)
        fuse original upload states using original region writer
    synchronize all seven models and evaluate Full by original protocol
    commit current task references, reliability and Adapter candidates
```

完整公式、共同类别规则和冷启动在 `SEMANTIC_A07_V2_DESIGN.md` 冻结。没有用测试 Accuracy/CE 选择公式、样本、参考、阈值或 lambda。合成语义记录只在单元测试出现，不注入真实训练。

## 研究限制与运行边界

1. 测量样本来自本地训练分区，可能反映训练内拟合；没有独立验证集。跨能力层同类概率差异也可能由样本难度、数据分布和 profile 构成造成。
2. 历史 EMA 可能滞后，稳定并不等于正确。rho 的量纲是证据使用系数，不是统计置信概率。当前宏平均只覆盖共同有效类别，可能存在类别选择偏差。
3. 旧 v1 负结果不能证明新 U 有效。新 U 没有通过接收端反事实效用或跨客户端迁移验证；当前任务不重做该验证，也不启用旧 gamma。
4. 短程真实非中性只证明实施了来源份额变化，不能说明 Full Accuracy 改善，更不能说明最终方法优于 A07。
5. 成本分开报告：测量额外前向/Adapter反向样本、墙钟、决策JSON编码、辅助Prototype逻辑上传、Adapter张量、tier Memory编码和固定probe缓存。JSON不是物理网络流量；这些存储项不是整个Python进程内存。服务端模拟器可读训练数据，不代表真实部署无需隐私和客户端额外通信。
6. 没有保存PT，没有中断恢复设计；失败保留记录，不伪造恢复。旧实验不覆盖，不追加种子或参数搜索。

## 后续入口（本次不执行）

默认固定配置为 lambda=0.2；正式比较应以相同入口的 reference/neutral 为配对参考，不能把短程测试或旧 v1 结果代替新400轮收益证据。以下仅提供后续获准后的入口：

```bash
CUDA_VISIBLE_DEVICES=0 /home/hh/miniconda3/envs/hh/bin/python -u tools/run_semantic_a07_v2.py \
  --mode active --rounds 400 --execute-formal \
  --dataset-root /home/hh/ICML_finish/data \
  --prepared runs/capability_final40_400_seed1_20261021/prepared \
  --output-root runs/semantic_a07_v2_formal_NOT_STARTED
```

本次只运行 `tools/run_semantic_a07_v2_short.py` 的六条2/20轮检查；实际结果、训练commit、配置/分区SHA和证据见 `SEMANTIC_A07_V2_RESULTS.md`。
