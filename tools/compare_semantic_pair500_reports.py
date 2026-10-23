"""Numerically compare independently computed Linux/local JSON summaries."""
import argparse
import json
import math
from pathlib import Path


def flat(value,prefix=''):
    if isinstance(value,dict):
        for k,v in value.items():yield from flat(v,prefix+'/'+str(k))
    elif isinstance(value,list):
        for i,v in enumerate(value):yield from flat(v,prefix+'/'+str(i))
    else:yield prefix,value


def compare(first,second):
    a=dict(flat(json.loads(first.read_text())));b=dict(flat(json.loads(second.read_text())))
    assert a.keys()==b.keys();errors=[]
    for k,v in a.items():
        if isinstance(v,(int,float)) and not isinstance(v,bool):
            assert math.isclose(v,b[k],rel_tol=1e-9,abs_tol=1e-10),(k,v,b[k])
            errors.append(abs(v-b[k]))
        else:assert v==b[k],(k,v,b[k])
    return dict(success=True,numeric_fields=len(errors),maximum_absolute_difference=max(errors,default=0),
        relative_tolerance=1e-9,absolute_tolerance=1e-10,non_numeric_structure_equal=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('local',type=Path);p.add_argument('linux',type=Path)
    p.add_argument('--output',type=Path,required=True);opt=p.parse_args()
    result=compare(opt.local,opt.linux);opt.output.write_text(json.dumps(result,indent=2));print(result)
