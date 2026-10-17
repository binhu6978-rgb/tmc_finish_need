"""Create a Chinese mechanism report and an unexecuted fixed-upload protocol."""
import argparse
import csv
import hashlib
from pathlib import Path

import numpy as np

from research.risk_diagnostics import dump, read, sha


def table(headers, rows):
    return ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"] + ["| " + " | ".join(map(str,row)) + " |" for row in rows]


def csvrows(path):
    with path.open(encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args(); output=args.output; root=Path.cwd()
    s=read(output/"summary.json"); inv=read(output/"artifact_inventory.json"); provenance=read(output/"provenance.json")
    assert provenance["analysis_script_sha256"]==sha("research/risk_diagnostics.py")
    assert [s['A3'][k] for k in ('slots','calibrated','anomaly_below_median','anomaly_equal_median','anomaly_above_median','nonneutral')]==[1000,906,882,0,24,3]
    assert s['A3_shadow']['ecdf_rank']['nonneutral']==612 and s['A3_shadow']['short20']['nonneutral']==118
    manifest=read(root/"runs/pair100_A3_seed1/manifest.json")
    partition_path=root/"data"/manifest["partition"]["file"]
    assert sha(partition_path)==manifest["partition"]["sha256"]
    users=read(partition_path)["train_data"]; ids=[i for indices in users.values() for i in indices]
    assert sorted(ids)==list(range(50000)) and len(users)==100
    proposed=sorted(sorted(range(50000),key=lambda i:hashlib.sha256(f"FD-utility-probe-v1/{i}".encode()).digest())[:2048])
    pilot=inv["minimal_frozen_state_pilot"]
    command=read(root/"runs/four100_control_20261017/status.json")["runs"]["A3"]["command"]
    train_path=Path(command[command.index("--dataset-root")+1])/"cifar-100-python/train"
    probe={"proposal_only_no_forward_or_training_executed":True,"indices":proposed,
        "index_selection":"Lowest SHA256(FD-utility-probe-v1/index), then ascending dataset index; no label/test/risk selection",
        "ordered_indices_sha256":hashlib.sha256(",".join(map(str,proposed)).encode()).hexdigest(),
        "original_partition_sha256":sha(partition_path),"train_file_sha256":sha(train_path),
        "dataset_train_path":str(train_path),"batch_size":128,"batches":16,
        "transform":manifest["transform"],"no_new_holdout_or_partition_change":True,
        "status":"TRAINING probe only; no independent generalization claim",
        "probe_client_overlap_counts":{c:len(set(indices)&set(proposed)) for c,indices in users.items()}}
    dump(output/"proposed_probe_indices.json",probe)
    dump(output/"data_partition_check.json",{"path":str(partition_path),"sha256":sha(partition_path),
        "clients":len(users),"indices":len(ids),"unique_indices":len(set(ids)),"unchanged":True})
    flow=s["A3"]; tr=s["trends"]; lag=s["A3_late_history_lag_slot_weighted"]
    blocks=csvrows(output/"A3_flow_by_round20_block.csv")
    profiles=csvrows(output/"A3_flow_by_profile.csv")
    ranges=csvrows(output/"A3_flow_by_sample_range.csv")
    clients=csvrows(output/"A3_flow_by_client.csv")
    clients=sorted(clients,key=lambda r:(-int(r["positive_anomaly"]),-int(r["slots"])))
    lines=["# FD-SemConsFL：语义风险与 Reliability Memory 机制诊断", "",
        "本轮仅使用既有实验产物进行CPU离线统计、评分重放和实验协议设计。没有新的联邦训练、模型前向评价或上传重采集；未修改 main、AdaptiveFL、正式 A2/A3、模型、分区、SGD、评估协议或 Gate 参数。", "",
        "**最重要的结论**", "",
        "A3大量z=0的直接原因已经确定：906个可标定槽位中，882个anomaly严格小于旧同档median，0个相等，24个大于；882个z=0与882个below一一对应。不是IQR下界错误，也不是把缺失证据误填成零。", "",
        "真实异常分数随训练明显下降，而原同档历史混合了较早训练阶段的更大分数；这给出了历史基准滞后的实证证据。但没有真实更新效用标签，不能据此断言median阈值错误，或断言被压成零的更新都应被门控。", "",
        "稀疏交集主要来自稀疏的当前z与继承大量零值的旧客户端历史。699个历史条件齐备的槽位中，z>0仅10个，r>0仅64个，交集3个。按独立边际频率作描述性计算的期望交集仅0.916个，实际3个更大；不能声称z与r表现为统计上的相互排斥。", "",
        "当前可验证的是数值流程、时间顺序及分数稀疏机制；尚不能验证其识别有害更新的能力。正式100轮没有逐轮上传张量，因此不能从签名/标量范数恢复服务器反事实模型。下一步无需先重跑100轮：可申请一个基于最终已存Full/Memory的单批次固定状态采集试验，先检验反事实聚合与效用标签是否可测。", "",
        "**数据来源与重放核验**", "",
        "原始A2/A3均来自提交`b546112e482c4dbdeb6ed49744cf9a8ae64452fc`。分析直接重读各自100轮rounds.jsonl及manifest，重构旧Anchor、正式异常和Reliability；没有把旧报告中的数字当输入结论。两组各1000槽位的正式决策全部字段逐项精确相等，最终Semantic/ Reliability Memory与checkpoint精确相等，并交叉核对旧slots.csv。", "",
        "原始签名未记录低于最小类别支持数的概率行，故这些类别的confidence/error-mass遥测无法从签名重放；其count、validity、reason可核验，且它们不参与评分或Anchor更新。不存在为了对齐而丢弃有效评分输入的做法。", "",
        "A2重新计数为12个正异常槽位、1次非中性Gate；A3为980个正异常槽位、3次Gate，与此前事实一致。原始分区文件SHA256、50000个索引全覆盖及100客户端均再次核验。", "",
        "A3共1000个真实槽位，但只有99个物理客户端、944个client-round，额外重复槽位56个。所有逐槽位CSV均标注同轮物理客户端及同档重复次数。分组表同时给出槽位数、独立客户端数和client-round数；重复槽位保留其真实聚合含义，绝不作为独立受试样本计算显著性。", "",
        "**A/B：信号流失分解**", ""]
    definitions=[("有训练样本的类别对",64286,100000,"slot-class"),
        ("B/A类别支持与数值有效",29367,64286,"slot-class"),
        ("可与旧Anchor比较",28720,29367,"slot-class"),
        ("JS正漂移",20314,28720,"slot-class"),
        ("有效异常槽位",980,1000,"slot"),("正异常（在有效槽位内）",980,980,"slot"),
        ("同档历史足够、z可计算",906,980,"slot"),("高于旧median / z>0",24,906,"slot"),
        ("旧r>0（全部历史条件齐备时）",64,699,"slot，独立旁路"),
        ("z>0且r>0（历史条件齐备时）",3,699,"slot"),("实际gamma<1",3,1000,"slot")]
    lines+=table(["阶段","计数/分母","比例","统计单位"],[(a,f"{n}/{d}",f"{n/d*100:.2f}%",u) for a,n,d,u in definitions])
    lines += ["", "r是与当前z相乘的旁路历史量，64不是24的后续子集。类别对与槽位也不是同一统计单位，不能将这些比例连乘解释为单一独立概率链。", "",
        "24个z>0槽位进一步分解：14个尚缺客户端历史，7个客户端历史已足够但旧z全为零，只有3个旧r也为正。另有61个槽位r>0而当前z=0。699个历史齐备槽位的2×2计数如下。", ""]
    lines+=table(["","r=0","r>0"],[("z=0",628,61),("z>0",7,3)])
    lines += ["", "历史齐备时P(z>0)=10/699，P(r>0)=64/699，P(交集)=3/699；P(r>0|z>0)=3/10。3个正例不足以稳定估计关联强度，上述独立边际乘积不是假设检验或因果证据。", "",
        f"64个r>0槽位中，{s['A3_r_positive_using_only_other_profile_positive_history']}个的旧正z记录全部来自当前profile以外的档位。物理客户端跨档历史共享是正式设计，不是实现错误；但历史信号与当前档位/训练阶段的可比性值得后续效用检验。", "",
        "**时间趋势与旧中位数滞后**", ""]
    lines+=table(["通信轮次","有效异常/可标定槽位","anomaly均值","旧median均值（可标定槽位）","旧IQR均值","z>0","Gate"],
        [(f"{(int(r['group'])-1)*20+1}–{int(r['group'])*20}",f"{r['valid_anomaly']}/{r['calibrated']}",f"{float(r['anomaly_mean']):.5f}",f"{float(r['median_mean_calibrated']):.5f}",f"{float(r['iqr_mean_calibrated']):.5f}",r['z_positive'],r['nonneutral']) for r in blocks])
    lines += ["", "第一段anomaly均值使用180个有效槽位，而median/IQR只使用106个可标定槽位；其余段二者均为200个槽位。避免将首段不同分母的均值直接解释为配对差。第41–60轮全部200槽位均低于旧中位数。", "",
        f"anomaly与轮次的Spearman相关（先合并同client/profile/round重复槽位）为{tr['round_vs_anomaly_client_profile_round_spearman_descriptive']:.4f}。七档内相关均为负，范围{min(tr['within_profile_round_rho'].values()):.4f}至{max(tr['within_profile_round_rho'].values()):.4f}。在269个跨轮重复的client-profile配对中257个斜率为负；控制同一client-profile后，log(anomaly)每轮斜率为{tr['within_client_profile_log_slope_per_round']:.5f}。", "",
        "以92个物理客户端为cluster、1000次分析bootstrap得到斜率区间[-0.02717,-0.02413]。这仅描述同一固定训练轨迹中客户端构成变化下的稳定性，不是独立联邦种子重复、因果置信区间或泛化性能区间。", "",
        f"第81–100轮按实际槽位对应旧profile状态加权：旧median均值{lag['median']:.5f}，仅用过去20轮的median均值{lag['short20_median']:.5f}，当前anomaly均值0.01840；旧记录平均年龄{lag['mean_age']:.2f}轮。旧池中来自前20轮的记录比例为{lag['early20_fraction']*100:.2f}%。", "",
        f"只剔除前20轮后，旧median仍为{lag['median_excluding_early20']:.5f}，高于短期基准。因此不是最早20轮单独控制了中位数，更符合多个较早阶段共同混入非平稳尺度的解释。最大旧池仅{s['A3_formal_profile_history_max_records']}条，小于容量200；本次100轮中年龄上限也尚未迫使早期有效记录退出。不能据此推断1000轮中位数永不适应。", "",
        "上述证据支持训练阶段尺度漂移以及旧基准适应较慢；它不证明被忽略的较小漂移就是有害更新。median的一侧截断本来就应将正常低异常归零，缺少更新效用标签时无法计算漏检率。", "",
        "**分档、样本量与物理客户端**", ""]
    lines+=table(["profile","槽位","正anomaly","z>0","r>0","Gate","anomaly均值","旧median均值"],
        [(r['group'],r['slots'],r['positive_anomaly'],r['z_positive'],r['r_positive'],r['nonneutral'],f"{float(r['anomaly_mean']):.5f}",f"{float(r['median_mean_calibrated']):.5f}") for r in profiles])
    lines += [""]
    lines+=table(["本地样本数","物理客户端/槽位","z>0","r>0","Gate","anomaly均值"],
        [(r['group'],f"{r['physical_clients']}/{r['slots']}",r['z_positive'],r['r_positive'],r['nonneutral'],f"{float(r['anomaly_mean']):.5f}") for r in ranges])
    lines += ["", "小于400样本的12个客户端共147个槽位、146个有效正异常，但137个可标定槽位全部z=0。这是评分与样本量关联的具体表现，不是该组更新一定更正常的证明。原始本地训练固定epochs，实际SGD步数为5×floor(n/50)，因此样本量还与训练步数和签名估计精度同时关联。", "",
        "先控制精确轮次和profile固定效应，再按client-round等权汇总至物理客户端，关联结果如下；无独立槽位p值。", ""]
    lines+=table(["客户端特征","客户端数","原始分数Spearman","轮次/profile调整后Spearman"],
        [(r['feature'],r['physical_clients'],f"{r['client_level_rho_raw']:.3f}",f"{r['client_level_rho_round_profile_adjusted']:.3f}") for r in s['client_associations']])
    lines += ["", "样本量呈中等正关联，支持样本占比有较弱正关联；标签数、归一化标签熵、最大类别占比和支持类别数的整体关联较弱。这不足以排除具体类别组成、训练难度或客户端能力层的混杂，不能据此声称评分与标签组成无关。", "",
        "反复产生正anomaly但很少得到当前相对异常的例子：", ""]
    lines+=table(["client","槽位/client-round","正anomaly","z>0","r>0","Gate"],
        [(r['group'],f"{r['slots']}/{r['client_rounds']}",r['positive_anomaly'],r['z_positive'],r['r_positive'],r['nonneutral']) for r in clients[:8]])
    lines += ["", "这些是高频正漂移客户端，不是已识别的有害客户端。例如client93全部22个槽位为正异常，但z始终为零；client91有2次z>0和6次r>0，却未在同槽位同时满足。完整99客户端表与24个正z事件已单独导出。", "",
        "**C：只读shadow比较**", "",
        "三个视角在同一A3签名、Anchor和训练轨迹上顺序计算，始终只读当前轮之前的历史。每个候选分别积累自己的旧校准分数、去重和客户端历史，绝不复用正式算法未来历史或正式r来拼装候选risk。", "",
        "A formal：原median/IQR和全部正式条件。B ecdf_rank：q=(旧值小于当前的数量+0.5×相等数量)/旧池数量；排名本身先作信息观察。为显示覆盖，再将q作为有界shadow校准量送入其自己的旧客户端历史，risk=q×r_q。C short20：仅将profile原始分数池限制为过去20个通信轮次，其余公式、最少10条、物理客户端100轮年龄/5条容量/至少2条、收缩系数及Gate强度均保留。D是正式z与r的分开诊断，不增加另一套Gate。", "",
        "20轮在看shadow结果前固定，只比较这一个短窗口，没有搜索多个阈值/窗口，也没有使用官方测试准确率选择候选。", ""]
    methods=s['A3_shadow']
    lines+=table(["shadow视角","可标定","z/q>0","旧r>0","Gate槽位/client-round","Full Gate","gamma最小/均值"],
        [(m,d['calibrated'],d['z_positive'],d['r_positive'],f"{d['nonneutral']}/{d['client_rounds_gated']}",f"{d['profiles']['6']['nonneutral']}/151",f"{d['distributions']['gamma']['min']:.6f} / {d['distributions']['gamma']['mean']:.6f}") for m,d in methods.items()])
    lines += ["", "formal的882个零z中，768个旧分位排名仍为正且低于0.5，另114个排名为0；被合并为零的排名中位数为0.08792。说明一侧median截断确实折叠了低于中位数的顺序信息，但这些信息是否与有害性有关尚无证据。", "",
        "在568个同轮同profile、不同物理客户端的可比较配对上：ecdf_rank打破450个formal风险平局，严格反序0；short20打破146个平局、严格反序1。对应跨槽位risk Spearman分别为0.1011和0.1662；大量平局使这些相关很低，不能解读成排名更准确。", "",
        "潜在误判：ECDF把中位数附近的普通更新也映射为正值；全体异常恰好为零且同分时，midrank仍为0.5（已用纯历史测试验证），所以更活跃可能只是无条件缩步。short20减轻阶段滞后，却只依赖约20–30个近期profile记录，受近期客户端组成与估计方差影响，可能把正常训练变化当异常。两者都没有可计算的真实误报/漏报率。", "",
        "所有shadow结果都是固定旧轨迹上的反事实评分，既没有重算历史上传，也没有让Gate反馈训练；612或118个shadow Gate绝不是闭环新算法结果，更不是准确率收益。", "",
        "**D：已有产物与更新效用可行性**", "",
        "四组正式目录各只有manifest.json、rounds.jsonl、latest.pt。逐一读取checkpoint后确认保存的是最终Full、最终Memory/RNG、模式、指纹及next_round=100；没有每轮聚合前Full或每槽位上传张量。最终七档可从已同步Full按原prefix恢复形状，但不能恢复历史参数。", "",
        "可用：profile形状和覆盖、每轮client/slot/n、Gate、签名、类别支持、每轮七档官方测试标量、固定原始训练索引、最终状态。不可用：正式100轮的历史Full张量、上传方向、每客户端边际反事实模型、对应反事实CE。", "",
        "coverage_update_diagnostics确实在原训练时基于当批真实上传计算过Gate=1反事实区域范数；这些已存标量可以证明已有的幅度变化，但无法反演方向、重新置换风险、构造缩步/删除客户端模型或评价CE。anomaly/z/r/gamma和轮际测试差同样不能替代边际效用。", "",
        "例外：独立的5个首轮诊断片段各保留10份上传及聚合后Full，初始状态仅有哈希；原CPU初始化可另行按源码与哈希验证，但这些不是正式A2/A3的后续训练上传。其首轮无旧Anchor且强制Gate=1，不能用来证明第18/86/91/99轮的语义风险效用。本阶段没有运行这些上传的反事实聚合。", "",
        "**E：最小后续采集方案（只设计，未执行）**", "",
        f"优先方案：只读A3第100轮最终Full、Memory和全部RNG，建立隔离的固定状态采集器。保持所有原始本地训练设定，按保存的Python/NumPy状态计算唯一的下一个available调度，收集10个新上传。这不是恢复已丢失的旧上传，也不是修改epochs指纹续训100/1000轮；不覆盖正式checkpoint，不在采集后继续闭环训练。", "",
        f"拟定下一个批次（仅RNG调度投影、无SGD）为`{pilot['original_next_schedule']}`，格式[client,profile]；共{pilot['distinct_physical_clients']}个物理客户端，client93在Full重复两次。固定状态对应零基round=100，学习率{pilot['learning_rate_at_this_state']:.10f}，预计{pilot['local_sgd_steps']}次原始本地SGD step。", "",
        f"需存：聚合前Full、逐槽位上传float32、profile与prefix/覆盖、n与client/slot、同轮分组、旧Anchor/Memory、pre/post签名、各风险、原始索引/源码/模型哈希、RNG、聚合后Full及原聚合核验结果。该批次pre+uploads+post约{pilot['float32_pre_upload_post_bytes']/2**20:.2f} MiB；元数据另计。", "",
        "训练采集粗预算：既有A3每轮本地训练均值10.66秒、测量1.61秒，均为双任务共享GPU观测，不能承诺隔离延迟。计入模型/数据加载、张量落盘及同机负载，先按1–3分钟的单批次采集量级规划；反事实前向评价另计。它只有9个客户端组，且可能没有非中性Gate，只能做可行性与方向探索；不得因结果不理想自动重抽客户端或换种子。", "",
        "该预定批次的Full只有client93一个物理客户端组，两个重复槽位不能当作两个Full受试者，无法做非退化的Full同档置换或客户端间风险排名。整个批次的同档置换最多4种不同组合。若需要识别Full语义选择效应，必须另行批准预定多个批次/状态；单批方案的目标是验证采集与效用测量能否工作。", "",
        "如要跨训练阶段研究，则另行批准一次原协议100轮采集并提前冻结采集轮次。按真实调度估算，全100轮每轮pre-Full+上传共24.89 GiB，额外全存post约29.15 GiB；只存20/40/60/80/100五轮约1.53 GiB。连续pre-Full可同时作为上一轮post并另存最后Full，以节省重复存储。完整重采参考已测单臂约35分钟共享GPU墙钟，附加IO/前向成本另计；不保证复现原CUDA非确定轨迹。", "",
        "有必要新增隔离的诊断采集入口，而不是改正式A2/A3。当前只提交方案和分析工具，未实现或执行任何采集训练入口。", "",
        "**固定上传反事实协议**", "",
        "具体公式、组级边际定义、缩步和置换控制见同目录update_utility_protocol.md。已冻结拟用的2048个原始训练probe索引及顺序，记录于proposed_probe_indices.json；不移除任何训练样本，不新增验证划分，尚未执行前向。", "",
        "**研究决策**", "",
        "事实：A2主要被置信度保护压低原始异常覆盖；A3移除该瓶颈后，仍受明显阶段漂移与长同档基准的一侧截断限制，零z又进入旧客户端持久性历史，最终risk更稀疏。统计上是profile校准与客户端历史相结合的瓶颈，当前z稀疏是最先出现的主瓶颈。", "",
        "科学限制：20314个JS正漂移类别对中20302个真实类置信度提高，且原始anomaly与置信度改善呈正关联（rho约0.338）。漂移可能混入正常本地学习进程，但这仍是本地训练量，不是全局效用标签。现有Reliability数值区分度低，却不能判断其筛掉的是正常更新还是有害更新；“统计信号变活跃”和“识别有害更新”须分开。", "",
        "应保留已验证的B/A同数据同批测量及状态/RNG恢复、旧Anchor/EMA与严格写时序、重复客户端去重、物理客户端历史边界、原始覆盖分母与只门控分子、正式A2定义和A3隔离方式。已验证的是实现和时序，不是语义有效性的全部假设。", "",
        "下一项最值得独立验证的候选：只把profile原始分数基准限定到预先固定的过去20轮，保留全部客户端历史和风险公式。先在固定上传上比较其效用排序及匹配范数/置换对照；本轮shadow只证明它改变了覆盖，尚不能作为采用依据。优先此候选而非删除r或直接使用ECDF风险。", "",
        "在出现以下证据前不建议继续大规模训练：同批固定上传的反事实CE可稳定重现；风险能在控制轮次/profile、n及上传范数后区分有害与有益组；语义门控优于同幅度缩步及组级置换，且效果超过中性算术/前向数值差异；再经预先规定的更多状态验证。原始训练probe只能支持训练效用判断，论文泛化结论仍需冻结的正式测试或另行授权的独立数据。", "",
        "**复核方式与分析工具修复说明**", "",
        "离线分析10项纯历史/标量测试通过。首次分析在本机NumPy最小二乘调用处无Python回溯退出，3×2最小输入也复现退出码1；没有确定底层运行库原因。分析工具改用等价的加性固定效应交替投影，并增加平衡/非平衡正交检验；本数据15次迭代收敛，最大组均值残差4.22e-15。未修改环境或训练算法。", "",
        "```powershell", "$env:PYTHONDONTWRITEBYTECODE='1'", "$env:PYTHONIOENCODING='utf-8'",
        r"& 'D:\software\minicoonda\envs\fd\python.exe' -m pytest -q tests/test_risk_diagnostics.py",
        r"& 'D:\software\minicoonda\envs\fd\python.exe' -m research.risk_diagnostics --a2 runs/pair100_A2_seed1 --a3 runs/pair100_A3_seed1 --output " + args.output.as_posix(),
        r"& 'D:\software\minicoonda\envs\fd\python.exe' -m research.write_risk_report --output " + args.output.as_posix(), "```", "",
        "首次重跑主分析需使用新的输出目录，避免覆盖已有产物；报告生成器可从同一数值汇总重建文字。provenance.json记录原始输入与脚本SHA256、实际命令和环境。", "",
        "可复核产物：slots_replayed.csv、shadow_slots.csv、profile_history_each_round.csv；A3_flow_by_round/profile/client/sample_range/round20_block.csv和profile_round20_block表；within_client_profile_trends.csv、client_features_and_adjusted_scores.csv、client_associations.csv、A3_positive_z_events.csv；summary.json、artifact_inventory.json、data_partition_check.json和proposed_probe_indices.json。", "",
        "分析已完成并停止，所有新上传采集、反事实前向、候选闭环训练和正式长程实验均等待后续明确授权。", ""]
    publication=output/"publication.json"
    if publication.exists():
        published=read(publication)
        lines += ["",f"分析代码发布分支：`{published['branch']}`；commit：`{published['commit']}`。正式训练源码仍为原参考版本。", ""]
    (output/"mechanism_report.md").write_text("\n".join(lines),encoding="utf-8")
    (output/"update_utility_protocol.md").write_text((root/"docs/FD_SEMCONSFL_UPDATE_UTILITY_PROTOCOL.md").read_text(encoding="utf-8"),encoding="utf-8")
    dump(output/"report_generation.json",{"script_sha256":sha(__file__),"output":str(output),
        "input_summary_sha256":sha(output/"summary.json"),"probe_ordered_indices_sha256":probe["ordered_indices_sha256"],
        "protocol_sha256":sha(root/"docs/FD_SEMCONSFL_UPDATE_UTILITY_PROTOCOL.md"),"no_training_or_forward_evaluation":True})
    print(str(output/"mechanism_report.md"))


if __name__=="__main__":
    main()
