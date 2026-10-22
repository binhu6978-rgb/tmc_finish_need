# 最终审计与交付

**科研决定：No-Go。阶段C完成；D/E未启动。** 完成检查已停用，不自动追加训练。

|状态|结论|
|---|---|
|软件实现|观测Adapter、Prototype、两类Memory、原A07调用及诊断可运行；非中性gamma按C失败停止条件保持锁定|
|A07回归|两版真实2/20轮全七模型状态、训练RNG及评价完全一致；400轮Accuracy/CE/调度与历史A07相同|
|机制验证|3996对真实utility；预选S0及两类Memory未共同获得独立信息支持|
|正式语义性能实验|未启动六臂400轮、最终多seed或跨数据集语义Gate实验|
|论文级四阶段证据|不足，不宣称完整语义算法有效或收敛|

## 来源与执行

仓库binhu6978-rgb/tmc_finish_need；新分支feature/semconsfl-a07-integrated-v1。原分支未改、无force push。A07原源码24e79dbb26ce718b533e7bc2c440a91c628caa25，398d44a是其增加启动证据的后代。阶段C实际代码01e8a92049fc096dc977fbbbe64d029ad4de887e；最终证据/报告在后续提交，不能混作训练SHA。

唯一服务器项目/home/hh/TMC_re_e03_scope_components400_20261019；Python /home/hh/miniconda3/envs/hh/bin/python，PyTorch2.7.1+cu118，物理GPU0 RTX4090，GPU1其他项目未干预。数据/home/hh/ICML_finish/data；完整50000/10000 CIFAR100，原100客户端分区、七模型、10slot、5epoch、batch50、原SGD，确定性CUDA及Full-only。

真实命令：

```bash
CUDA_VISIBLE_DEVICES=0 /home/hh/miniconda3/envs/hh/bin/python -u tools/run_semcons_a07_mechanism.py \
  --dataset-root /home/hh/ICML_finish/data \
  --prepared runs/capability_final40_400_seed1_20261021/prepared \
  --equivalence-evidence runs/semcons_a07_regression_v2_20261022/equivalence_20.json \
  --output-root runs/semcons_a07_mechanism_v2_20261022
```

这是已完成的命令记录，不需要再次执行。父4188329、训练子4188333均正常退出；status completed、success true、400条连续有限CE/Accuracy；无PT，无恢复或重试。配置及分区完整SHA在MECHANISM_VALIDATION及归档manifest。14个固定诊断点、三头种子、开发/确认划分和Go标准未因观察结果改变。采样可行性修正及原V1证据保留在DESIGN和Git历史中。

## 代码与交付清单

- semcons_a07：config、adapter、measurement、prototypes、semantic_memory、reliability_memory、aggregation、diagnostics、trainer。
- configs/semcons_a07_observation.json及冻结V2；tools/run_semcons_a07_observation.py、regression.py、mechanism.py；check_semcons_a07_equivalence.py、plan_semcons_a07_pairs.py、analyze_semcons_a07.py、finalize_semcons_a07.py。
- 四组必要测试文件test_semcons_a07、analysis、regression_evidence、final_evidence；旧专项测试继续执行。
- docs下SOURCE_AUDIT、DESIGN、IMPLEMENTATION、REGRESSION、MECHANISM_VALIDATION、RESULTS、REVIEWER_ALIGNMENT及本审计。
- docs/evidence/semcons_a07/final：原始配对/每轮观测/诊断/状态journal无损gzip，400轮CSV、完成/资源/source manifest、原始统计、SHA索引、本地重算校验及两张图。大体积展开标量副本保留服务器与本地ignored cache，不提交图像、更新张量、PT、私人论文或凭据。

未改Algorithm/Training_AdaptiveFL.py、models/Update.py、models/Fed.py、models/resnet.py、capability_final40.py或native_group_aggregation.py。原始数据及所有旧实验保留。客户端参与/模型同步全七档；语义测量仅复制模型/读取训练数据，主梯度和RNG有专项断言。当前Prototype不能进入本轮Anchor；首轮不建语义历史；重复物理身份不伪造独立票数。

## 核验范围与限制

finalize工具核对全部11988条head记录的配对身份、Native注入系数、支持区域、逐类CE加权、相同utility；验证3996对每对三头、14诊断点、无当前轮Anchor/历史年龄越界、四channel100类及全400中性。原始配对统计在本地独立重算，779数值字段最大差1.015e-11。245项当前专项测试结果见local_final_tests.log及linux_final_tests.log；此前Linux240项与真实GPU smoke均保留。

最后Linux测试实际245通过（13.07s）；独立在服务器原始run目录执行finalize审计，退出码0。该次核验代码提交ed07801cf38465bc604038fb440e1c664b9c485c，后续提交只补齐其日志/交付记录，不改变评分或训练源码。两张交付图已完成视觉核验。原始展开记录仍在服务器run及本地ignored cache，发布gzip保留同一原始字节与SHA索引。

历史400轮没有保存权重，因此只有2/20轮具备全状态精确对照；400轮可以核验全部指标/调度和初始状态，不伪称400轮历史权重逐bit比较。Probe使用私有训练样本，是条件性训练侧效用而非泛化标签。开发/确认同时有阶段与能力层转移；Reliability支持明显越界，不能把线性外推误差理解为Memory本质无用。负LOO确认33对不足以独立支持该子群结论。

实现没有证明Prototype具有跨客户端效用可比性；S0无效用增量，S1仅弱关联，S2有少量Anchor回归迹象仍未过Go。完整方法未优于A07，因为没有获准激活或开展其性能实验。不能强行把无作用Memory写入论文有效模块。原型utility核心问题尚未解决，相关方法、模型×数据异构控制、其他模态、真实扩展成本、多seed及适用的新理论仍欠缺，详见审稿映射。
