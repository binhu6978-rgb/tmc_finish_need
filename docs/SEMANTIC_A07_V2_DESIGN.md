# Semantic-Guided A07 V2：冻结实现定义

基准分支feature/semconsfl-a07-integrated-v1，HEAD9cfe4a9686dbd337a4ffd848f1cadeb35fba30f9；本地与远端一致，无新增提交。仅保留的旧未跟踪文件不动。新分支feature/semconsfl-semantic-a07-v2。v1 No-Go、原始数据和A07历史证据全部保留，不重写v1模块。

## 原A07及接口

capability_final40.Final40Hooks.aggregate先调用原Native覆盖聚合；九区域来自真实七模型前缀形状。多能力来源Feature/BN使用P04：区域层质量M_g为真实覆盖slot样本数之和，pi_g正比M_g^-1/2，层内样本权重；单层来源Native；classifier.weight/bias Native；首轮严格Native；未覆盖保留旧Full。V2每轮先执行该原函数，所有中性情况直接返回其原result，不用等价增量重算。七档SGD和split_model同步不变。

复用semcons_a07.Measurement和Observer.wrap：固定每类2个训练样本、batch50，主模型复制后eval/no_grad，同一Adapter快照和同样输入测前后，隔离RNG/梯度/BN状态。Adapter在detach特征上拟合下一轮候选，同轮重复slot不读未提交状态。三个冻结head17/29/43只辅助描述，**不进入V2评分或权重**。任务概率直接来自主模型classifier。

## 类别/能力层统计

同物理客户端、同类别多个slot/profile的正确类概率先等权平均；同能力层再对独立物理客户端取标量中位数，形成q_pre(g,c)、q_post(g,c)，两者使用完全相同身份/类/固定样本集合。不同profile保持为真实上传slot，不合并SGD；语义统计才去重。同层独立客户端不足2、类别测量不足2或非有限则missing。

新的TierSemanticMemory在轮开始取深拷贝快照。类别/层参考q_ref采用上一已提交参考0.9 EMA+当前稳健q_post；过期100轮重置。保存最近提交轮次、独立客户端名单/支持、原始观测q及与上次原始q的绝对变化。首轮不提交。旧v1类别Prototype和S1观测仍保留为辅助日志。

新的TierReliabilityMemory单独保存每个层/类有效提交序列，每轮每键只提交一次。使用最近两次有效历史：n=min(两次独立支持)，v=平均绝对响应变化，age=最近提交年龄。至少两次历史且有真实变化记录，n>=2且未过期，输出

rho_history = n/(n+1) × max(0,1-v) × max(0,1-age/100)。

rho只表示支持、时效和稳定性，不是客户端信誉或知识正确率。旧S1客户端历史不参与该系数。个人、profile历史仍保留辅助角色。

## 唯一互补分数

对每层g、类别c，从其他层的过去未过期参考中取最大q_ref。若该最高已知比较来源的Reliability不合格，类别missing，不替换成更容易超过的弱来源。自身历史也须有效。rho(g,c)取当前支持n_current/(n_current+1)、自身历史rho、该比较来源历史rho三者最小值。

d=max(0,q_post-q_pre)；a=max(0,q_post-max_other_past_q_ref)；u=rho×min(d,a)。

对当前真实参与的全部能力层取共同有效类别交集，至少2类；各层在同一类别集合宏平均得到U(g)∈[0,1]。没有公平交集或某层缺证据，整轮中性，不能把missing当成U=0。首次有效比较至少在第4轮：首轮不建历史，第2/3轮提交，第4轮只读这些过去记录。这个保守时序不以触发率或测试准确率调整。

Semantic Memory决定跨能力来源优势a；Reliability Memory决定证据可用性和rho。二者通过不同实际数据流影响U，均有单元测试。该分数只称task-grounded potential complementarity。

## 单次来源融合

每个实际多能力层共享区域，调用原source_weights取得A07 slot权重并汇总pi；pi_sem∝pi(1+U)，pi_new=(1−lambda)pi+lambda*pi_sem，默认lambda=0.2，可显式设0，无搜索。原slot权重仅乘pi_new/pi，保留同层相对样本权重；调用原write_weighted_region融合真实上传参数。

仅作用于共享Feature非BN的cell；BN保持原A07，classifier Native。首轮、未覆盖、单能力来源、缺证据或相同U均直接保持原A07对应结果。无Full锚点增量替换，无逐层评分/Top-K/方向过滤/保护更新/蒸馏/Trusted Subset。

## 检查和运行边界

先纯函数测试：中性精确路径、权重合法/组内比、作用范围、去重、历史快照、两个Memory和合成非中性激活。随后2轮和20轮各运行reference、neutral、active，完整真实数据/原本地训练，记录主状态/RNG/调度、共同类、rho/U、区域份额和实际差异。实际数据若不触发，如实报告；不注入合成分数或降低条件。短程模式无400轮调用，后续正式入口有单独显式执行参数，本次不用。

## 科学限制

正确类概率来自本地训练样本，有训练内偏差；各层客户端样本和类别难度不同，即使同类坐标也不代表相同输入分布或校准程度。层级q、历史和U仅是代理证据，不能证明跨客户端迁移或全局收益。全局来源U不是参数/卷积层语义归因。Adapter辅助观测不能被描述为聚合收益来源。

v1研究Prototype-anchor/有符号变化能否预测接收端CE效用，结果No-Go；V2换成训练侧正向类别增益与过去其他层优势的有界来源份额融合，含归一化，不是启用旧gamma。旧负结果不是V2成功证据。软件激活、短程可训练均不能证明优于A07或取得论文资格。正式效用、独立验证及多seed尚未完成。

额外记录测量时间/前向和Adapter反向样本、旧辅助tensor payload、实际构造的决策JSON编码字节、Adapter tensor存储及Tier Memory JSON编码大小；这些是仿真逻辑量，不冒充物理网络或Python进程resident内存。历史更新只在聚合/同步/评价完成后提交。官方测试仅按原Full-only协议记录，不用来调lambda或筛选公式。
