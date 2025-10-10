import torch
from collections.abc import Mapping, Callable
from torch import tensor, Tensor
from cajal.typing import *
from functools import partial

# TODO: Refactor compile and tests to take device as argument
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

type Vector = Tensor | LinearMap
type VectorEnv = Mapping[str, Vector]

type MultilinearMap = Callable[[VectorEnv], Vector]

def compile(tm: Tm) -> MultilinearMap:
    match tm:

        case TmVar(x):
            return lambda env: env[x]
        
        case TmTrue():
            tt = tensor([1., 0.], requires_grad=False, device=device)
            return lambda env: tt
        
        case TmFalse():
            ff = tensor([0., 1.], requires_grad=False, device=device)
            return lambda env: ff
        
        case TmZero():
            zero = tensor([1.], requires_grad=False, device=device)
            return lambda env: zero

        case TmSucc(tm):
            n = compile(tm)
            z = torch.tensor([0.], requires_grad=False, device=device)
            return lambda env: torch.concat([z, n(env)])
        
        case TmIf(tm1, tm2, tm3):
            b = compile(tm1)
            branch_if = compile(tm2)
            branch_else = compile(tm3)

            def execute(env):
                cond = b(env)
                return cond[0] * branch_if(env) + cond[1] * branch_else(env)
            
            return execute
        
        case TmFun(x, _, tm):
            body = compile(tm)
            return lambda env: LinearMap(lambda arg: body(env | {x: arg}))
        
        case TmApp(tm1, tm2):
            f = compile(tm1)
            x = compile(tm2)
            return lambda env: f(env)(x(env))

def compile_val(v: Val):
    match v:
        case VTrue():
            tt = tensor([1., 0.], requires_grad=False, device=device)
            return lambda env: tt
        case VFalse():
            ff = tensor([0., 1.], requires_grad=False, device=device)
            return lambda env: ff
        case VClosure(x, ty, tm, src_env):
            tgt_env = {y: compile_val(val_y)({}) for y, val_y in src_env.items()}
            body = compile(tm)
            return lambda env: LinearMap(lambda arg: body(tgt_env | {x: arg}))

class LinearMap:

    def __init__(self, f):
        self.f = f
        self.mat = None
        self.dim_in = None

    def __call__(self, x):
        return self.f(x)
    
    def __rmatmul__(self, W):
        # 1. Compute matrix of self.f
        self.f = tensor([[1.,0.],[0.,1.]], device=device)
        # 2. Do W @ flatten(self.mat)
        self.f = torch.flatten(self.f)
        # 3. Compute result
        res = W @ self.f

        # 4. Return LMap is output is func, otherwise return vec
        match self.ty_out:
            case TyFun(_, _):
                return LMap(lambda x : self._reshaping_matmul(res, x))
            case _:
                return res
    
    @staticmethod
    def _reshaping_matmul(w, x):
        W = torch.reshape(w, (-1, len(x)))
        return W @ x


        

def compile_val(v: Val):
    match v:
        case VTrue():
            return tensor([1.,0.], device=device)
        case VFalse():
            return tensor([0.,1.], device=device)
        case VClosure(x, ty, tm, src_env):
            tgt_env = {y: compile_val(val_y) for y, val_y in src_env.items()}
            return lambda arg, env=tgt_env: compile(tm, tgt_env | {x: arg})