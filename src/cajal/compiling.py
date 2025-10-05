from collections.abc import Mapping
from torch import tensor, Tensor
from cajal.typing import *
from functools import partial

def compile(tm: Tm, env):
    match tm:
        case TmVar(x):
            return env[x]
        case TmTrue():
            return tensor([1,0])
        case TmFalse():
            return tensor([0,1])
        case TmIf(tm1, tm2, tm3):
            b = compile(tm1, env)
            branch_if = compile(tm2, env)
            branch_else = compile(tm3, env)
            return b[0] * branch_if + b[1] * branch_else
        case TmFun(x, _, tm):
            return lambda arg, env=env: compile(tm, env | {x: arg})
        case TmApp(tm1, tm2):
            f_closure = compile(tm1, env)
            arg = compile(tm2, env)
            return f_closure(arg)


def compile_val(v: Val):
    match v:
        case VTrue():
            return tensor([1,0])
        case VFalse():
            return tensor([0,1])
        case VClosure(x, ty, tm, src_env):
            tgt_env = {y: compile_val(val_y) for y, val_y in src_env.items()}
            return lambda arg, env=tgt_env: compile(tm, tgt_env | {x: arg})