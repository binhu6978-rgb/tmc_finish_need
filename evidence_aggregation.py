"""E03-preserving evidence-scope and negative-projection controls.

No semantic scores, memories, local losses or original aggregators are changed.
The native M2 implementation computes ALL scores/weights, including slot LOO.
"""
import copy
import math

import torch

from adapterfl_methods import MethodHooks, EPS
from models.Fed import Aggregation_AdaptiveFL


class EvidenceHooks:
    is_evidence = True

    def __init__(self, args, models):
        if args.method not in ('c1', 'c2') or args.conflict_lambda != 4.:
            raise ValueError('Fixed candidates require c1/c2 and original M2 lambda=4')
        self.method = args.method
        native = copy.copy(args); native.method = 'm2'
        self.legacy = MethodHooks(native, models)
        self.config = dict(self.legacy.config, evidence_candidate=self.method,
            evidence_scope='intersection of ALL actual uploaded shapes, identical to native M2',
            dispatch_fusion='native M2 weighted dispatch fusion on evidence scope; native outside',
            c2_projection_gate='exp(original unshifted M2 logit), never stabilized weight',
            c2_learning_denominator='native sum of sample weights; only negative projection adjusted')
        self.round_diagnostics = {}
        self.rounds_seen = 0

    def configure_local(self, local):
        pass

    def state_dict(self):
        return {'method': self.method, 'config': self.config, 'legacy': self.legacy.state_dict(), 'rounds_seen':self.rounds_seen}

    def load_state_dict(self, state, full):
        if state['method'] != self.method or state['config'] != self.config:
            raise ValueError('Candidate configuration differs')
        self.legacy.load_state_dict(state['legacy'], full)
        self.rounds_seen = state['rounds_seen']
        if not isinstance(self.rounds_seen,int) or self.rounds_seen<0:raise ValueError('Invalid candidate round counter')

    @torch.no_grad()
    def aggregate(self, uploads, lens, previous, dispatched=None):
        # Call, rather than reimplement, E03 scoring and full reweighted aggregation.
        e03 = self.legacy.aggregate(uploads, lens, previous, dispatched)
        native = Aggregation_AdaptiveFL(uploads, lens, previous)
        diag = copy.deepcopy(self.legacy.round_diagnostics)
        sl = {k:tuple(slice(0,min(u[k].shape[d] for u in uploads)) for d in range(v.ndim))
              for k,v in previous.items()}
        counts = [previous[k][sl[k]].numel() for k in previous]
        total_numel = sum(v.numel() for v in previous.values())
        cosines = diag.get('cosine', [None]*len(uploads))
        negative = [i for i,v in enumerate(cosines) if v is not None and v<0]
        diag.update(candidate=self.method, common_evidence_numel=sum(counts),
                    common_evidence_fraction_full=sum(counts)/total_numel,
                    conflict_slot_indices=negative, conflict_slot_count=len(negative))
        if self.rounds_seen == 0:
            weights=diag.get('conflict_weights',[1.]*len(lens));mass=sum(lens)
            gated_mass=sum(n*g for n,g in zip(lens,weights));fusion_square=0.
            for k in previous:
                difference=sum((n*g/gated_mass-n/mass)*s[k][sl[k]].double() for n,g,s in zip(lens,weights,dispatched))
                fusion_square+=float(difference.square().sum())
            diag['initial_m2_dispatch_fusion_shift_from_native_l2']=math.sqrt(fusion_square)
        result = {k:v.clone() for k,v in native.items()}
        for k in result:
            result[k][sl[k]] = e03[k][sl[k]]  # C1: EXACT E03 on the evidence intersection.

        diag['projection_slots'] = []
        if self.method == 'c2' and negative:
            # Same subtraction precision and sample-weighted LOO as M2.
            deltas = [torch.cat([(u[k][sl[k]]-s[k][sl[k]]).reshape(-1).double() for k in previous])
                      for u,s in zip(uploads,dispatched)]
            mass = sum(lens)
            weighted_sum = sum(n*d for n,d in zip(lens,deltas))
            stabilized = diag['conflict_weights']
            beta = [n*g/sum(a*b for a,b in zip(lens,stabilized)) for n,g in zip(lens,stabilized)]
            alpha = [n/mass for n in lens]
            weighted_learning = sum(b*d for b,d in zip(beta,deltas))
            protected_learning = torch.zeros_like(weighted_learning)
            preserved_remainder = torch.zeros_like(weighted_learning)
            e03_remainder = torch.zeros_like(weighted_learning)
            for i,d in enumerate(deltas):
                ref = (weighted_sum-lens[i]*d)/(mass-lens[i])
                projection = torch.zeros_like(d)
                if i in negative:
                    # Unit reference avoids division by a potentially tiny squared norm.
                    norm = ref.norm()
                    if norm <= EPS: raise AssertionError('Native M2 negative cosine with zero reference')
                    unit = ref/norm
                    projection = torch.dot(d,unit).clamp(max=0)*unit
                remainder = d-projection
                gate = math.exp(diag['unnormalized_log_weights'][i])
                transformed = remainder+gate*projection
                protected_learning += alpha[i]*transformed
                preserved_remainder += alpha[i]*remainder
                e03_remainder += beta[i]*remainder
                diag['projection_slots'].append({'slot':i,'native_sample_fraction':alpha[i],
                    'e03_normalized_fraction':beta[i],'unshifted_gate':gate,
                    'delta_l2':float(d.norm()),'negative_projection_l2':float(projection.norm()),
                    'remaining_component_l2':float(remainder.norm()),
                    'projection_adjustment_l2':float(((gate-1)*projection).norm()),
                    'preserved_remaining_component':True})
            correction = protected_learning-weighted_learning
            offset = 0
            for (k,_),n in zip(previous.items(),counts):
                # Preserve C1/E03 dispatch fusion (also in independent-init round1).
                # Change only the learning term; outside the evidence stays native.
                result[k][sl[k]] = e03[k][sl[k]]+correction[offset:offset+n].reshape(result[k][sl[k]].shape).to(result[k].dtype)
                offset += n
            diag.update(e03_learning_evidence_l2=float(weighted_learning.norm()),
                        protected_learning_evidence_l2=float(protected_learning.norm()),
                        remaining_component_aggregate_difference_from_e03_l2=float((preserved_remainder-e03_remainder).norm()))
        # No conflict: exact native computation, no near-neutral numerical rewrites.
        if not negative:
            result = native
        squares = {'native_update':0.,'e03_update':0.,'actual_update':0.,'difference_from_native':0.,'difference_from_e03':0.}
        changed_native=0; changed_e03=0; dot=0.
        for k,v in previous.items():
            updates={'native_update':(native[k]-v).double(),'e03_update':(e03[k]-v).double(),
                     'actual_update':(result[k]-v).double(),'difference_from_native':(result[k]-native[k]).double(),
                     'difference_from_e03':(result[k]-e03[k]).double()}
            for name,d in updates.items():squares[name]+=float(d.square().sum())
            changed_native+=int(result[k].ne(native[k]).sum());changed_e03+=int(result[k].ne(e03[k]).sum())
            dot+=float((updates['actual_update']*updates['e03_update']).sum())
        diag.update({name+'_l2':math.sqrt(value) for name,value in squares.items()})
        diag.update(changed_from_native_fraction=changed_native/total_numel,changed_from_e03_fraction=changed_e03/total_numel,
            adjusted_projection_slot_count=sum(r['projection_adjustment_l2']>0 for r in diag['projection_slots']),
            update_l2_ratio_e03=math.sqrt(squares['actual_update']/squares['e03_update']) if squares['e03_update'] else None,
            effective_step_along_e03=dot/squares['e03_update'] if squares['e03_update'] else None,
            update_cosine_e03=dot/math.sqrt(squares['actual_update']*squares['e03_update']) if min(squares['actual_update'],squares['e03_update']) else None,
            formula_note='C1 uses exact E03 only on its evidence. C2 keeps that dispatch fusion; projection-only learning uses native coverage mass. Duplicate slots match E03, not physical-client LOO.')
        self.round_diagnostics = diag
        self.rounds_seen += 1
        return result
