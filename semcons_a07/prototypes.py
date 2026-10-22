import torch
from torch.nn import functional as F


def summarize(embeddings, probabilities, labels, minimum):
    result = {}
    for y in sorted(set(labels.tolist())):
        mask = labels == y
        count = int(mask.sum())
        if count < minimum:
            continue
        p = probabilities[mask].double().mean(0)
        record = dict(count=count, probability=p / p.sum())
        if embeddings is not None:
            record['embedding'] = F.normalize(embeddings[mask].double().mean(0), dim=0)
        result[int(y)] = record
    return result


def js(a, b):
    a = a.double().clamp_min(1e-12); a = a / a.sum()
    b = b.double().clamp_min(1e-12); b = b / b.sum()
    m = (a + b) / 2
    return float(((a * (a / m).log()).sum() + (b * (b / m).log()).sum()) / 2)


def scores(before, after, task_before, task_after, anchors, task_anchors, round_number, max_age, classes=None):
    values = dict(S0=[], S1=[], S2=[], adapter_probability_drift=[], current_embedding_change=[], current_task_change=[])
    ages = []; valid_classes = []
    for y in sorted(after if classes is None else classes):
        if y not in before or y not in after:
            continue
        values['current_embedding_change'].append(1-float(F.cosine_similarity(before[y]['embedding'], after[y]['embedding'], dim=0)))
        if y in task_before and y in task_after:
            values['current_task_change'].append(js(task_after[y]['probability'], task_before[y]['probability']))
        anchor = anchors.get(y)
        if anchor and 0 < round_number - anchor['round'] <= max_age:
            z = anchor['embedding']
            pre = 1 - float(F.cosine_similarity(before[y]['embedding'], z, dim=0))
            post = 1 - float(F.cosine_similarity(after[y]['embedding'], z, dim=0))
            values['S0'].append(post); values['S1'].append(post - pre)
            values['adapter_probability_drift'].append(js(after[y]['probability'], anchor['probability']) - js(before[y]['probability'], anchor['probability']))
            ages.append(round_number - anchor['round']); valid_classes.append(y)
        ta = task_anchors.get(y)
        if (ta and 0 < round_number - ta['round'] <= max_age and y in task_before and y in task_after):
            values['S2'].append(js(task_after[y]['probability'], ta['probability']) - js(task_before[y]['probability'], ta['probability']))
    return {k: sum(v) / len(v) if v else None for k, v in values.items()} | dict(
        valid_embedding_classes=valid_classes, mean_anchor_age=sum(ages)/len(ages) if ages else None,
        valid_task_classes=len(values['S2']))
