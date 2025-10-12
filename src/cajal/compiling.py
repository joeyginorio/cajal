import torch
from collections.abc import Mapping, Callable
from torch import tensor, Tensor
from cajal.typing import *
from typing import NamedTuple

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

type MultilinearMap = Callable[[VectorEnv], Vector]

def compile(tm: Tm) -> MultilinearMap:
    match tm:

        case TmVar(x):
            return lambda env: env[x]
        
        case TmTrue():
            tt = TypedTensor(tensor([1., 0.], requires_grad=False, device=device), TyBool())
            return lambda env: tt
        
        case TmFalse():
            ff = TypedTensor(tensor([0., 1.], requires_grad=False, device=device), TyBool())
            return lambda env: ff
        
        case TmZero():
            zero = TypedTensor(tensor([1.], requires_grad=False, device=device), TyNat())
            return lambda env: zero

        case TmSucc(tm_n):
            n = compile(tm_n)
            z = torch.tensor([0.], requires_grad=False, device=device)
            return lambda env: TypedTensor(torch.concat([z.data, n(env).data]), TyNat())
        
        case TmIf(tm1, tm2, tm3):
            b = compile(tm1)
            branch_if = compile(tm2)
            branch_else = compile(tm3)

            def execute(env):
                cond = b(env).data
                return cond[0] * branch_if(env) + cond[1] * branch_else(env)
            
            return execute
        
        case TmFun(x, _, tm_body):
            body = compile(tm_body)
            return lambda env: LinearMap(lambda arg: body(env | {x: arg}), tm.ty_checked)
        
        # tensor tensor (tensor.__matmul__) good
        # tensor lambda (tensor.__matmul__)
        # lambda lambda (linmap.__call__) good
        # lambda tensor (linmap.__call__) good
        case TmApp(tm1, tm2):
            f = compile(tm1)
            x = compile(tm2)
            return lambda env: f(env)(x(env))

def compile_val(v: Val):
    match v:
        case VTrue():
            tt = TypedTensor(tensor([1., 0.], requires_grad=False, device=device), TyBool())
            return lambda env: tt
        case VFalse():
            tt = TypedTensor(tensor([0., 1.], requires_grad=False, device=device), TyBool())
            return lambda env: tt
        case VClosure(x, ty, tm_body, src_env):
            tgt_env = {y: compile_val(val_y)({}) for y, val_y in src_env.items()}
            body = compile(tm_body)
            return lambda env: LinearMap(lambda arg: body(tgt_env | {x: arg}), v.ty_checked)

type Vector = Tensor | LinearMap
type VectorEnv = Mapping[str, Vector]

class TypedTensor(NamedTuple):
    data: Tensor
    ty: Ty
    
    def __eq__(self, y):
        return all(self.data == y.data)

    def __call__(self, x):
        return self @ x

    def __rmul__(self, x):
        return TypedTensor(x * self.data, self.ty)

    def __add__(self, y):
        return TypedTensor(self.data + y.data, self.ty)

    def __matmul__(self, x):
        if isinstance(x, TypedTensor):
            result = self.data @ torch.flatten(x.data)
            reshaped_result = reshape_with_ty(result, self.ty.ty2)
            return TypedTensor(reshaped_result, self.ty.ty2)
        else:
            mat = torch.ones(dim(x.ty)) ## dummy
            result = self.data @ torch.flatten(mat)
            reshaped_result = reshape_with_ty(result, self.ty.ty2)
            return TypedTensor(reshaped_result, self.ty.ty2)

class LinearMap:

    def __init__(self, f, ty: Ty):
        self.f = f
        self.ty = ty

    def __call__(self, x):
        return self.f(x)
    
    def __rmul__(self, a):
        return LinearMap(lambda x : a * self.f(x), self.ty)
    
    def __add__(self, g):
        return LinearMap(lambda x : self.f(x) + g(x), self.ty)
    
    def __matmul__(self, x):
        return self.f(x)

def mat_of_lmap(lmap: LinearMap):
    match lmap.ty:
        case TyFun(ty_in, ty_out):
            # 1. get basis of input
            # 2. push each basis through lmap.f 
            # 2a. if ty_out is base type, we can observe directly, create the matrix
            # 2b. if ty_out is function type, we can't observe directly......
            # -- lmap.f(b1) = some function
            # -- lmap.f(b2) = some function
            # -- lmap.f(b3) = some function
            # -- lmap.f(b4) = some function
            # So then for each of these we then recursively compute the mat_of_lmap
            outs = []
            for basis in bases(ty_in):
                outs.append(lmap.f(basis))

        case _:
            raise TypeError(f"{lmap.ty=} is not a function, so can't build its matrix.")

def bases(ty: Ty):
    match ty:

        case TyBool():
            basis_mat = torch.eye(2)
            basis_tt = basis_mat[:,0]
            basis_ff = basis_mat[:,1]
            return [basis_tt, basis_ff]
        
        case TyFun(_, _):
            dim_ty = dim(ty)
            basis_mat = torch.eye(dim_ty)

            bases = []
            for i in range(dim_ty):
                bases.append(reshape_with_ty(basis_mat[:,i], ty))
            return bases
