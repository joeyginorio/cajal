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
            _zero = torch.zeros(10, requires_grad=False, device=device)
            _zero[0] = 1.
            zero = TypedTensor(_zero, TyNat())
            return lambda env: zero

        case TmSucc(tm_n):
            n = compile(tm_n)
            z = torch.tensor([0.], requires_grad=False, device=device)
            return lambda env: TypedTensor(torch.concat([z, n(env).data[:9]]), TyNat())
        
        case TmIf(tm1, tm2, tm3):
            b = compile(tm1)
            branch_if = compile(tm2)
            branch_else = compile(tm3)

            def execute(env):
                cond = b(env).data
                return cond[0] * branch_if(env) + cond[1] * branch_else(env)
            
            return execute
        
        case TmIter(tm1, y, tm2, tm3):
            base = compile(tm1)
            rec = compile(tm2)
            num = compile(tm3)

            def execute(env):
                n = num(env).data
                base_val = base(env)
                rec_f = lambda y_tgt: rec(env | {y: y_tgt})

                total = n[0] * base_val
                for i in range(1, len(n)):
                    total += n[i] * apply_n(rec_f, i)(base_val)
                
                return total
            
            return execute
        
        case TmFun(x, _, tm_body):
            body = compile(tm_body)
            return lambda env: LinearMap(lambda arg: body(env | {x: arg}), tm.ty_checked)

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

    def __add__(self, x):
        if isinstance(x, TypedTensor):
            result = self.data + x.data
            return TypedTensor(result, self.ty)
        else:
            mat = mat_of_lmap(x)
            result = self.data + mat.data
            return TypedTensor(result, self.ty)

    def __matmul__(self, x):
        if isinstance(x, TypedTensor):
            result = self.data @ torch.flatten(x.data)
            reshaped_result = reshape_with_ty(result, self.ty.ty2)
            return reshaped_result
        else:
            mat = mat_of_lmap(x)
            result = self.data @ torch.flatten(mat.data)
            reshaped_result = reshape_with_ty(result, self.ty.ty2)
            return reshaped_result

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

            if not isinstance(ty_out, TyFun):
                outs = []
                for basis in bases(ty_in):
                    outs.append(lmap.f(basis).data.reshape(-1,1))
                return TypedTensor(torch.hstack(outs), lmap.ty)
            else:
                outs = []
                for basis in bases(ty_in):
                    out = lmap.f(basis)
                    if isinstance(out, TypedTensor):
                        outs.append(torch.reshape(out.data, (-1, 1)))
                    elif isinstance(out, LinearMap):
                        mat_out = mat_of_lmap(out).data.reshape(-1,1)
                        outs.append(mat_out)

                return TypedTensor(torch.hstack(outs), lmap.ty)

        case _:
            raise TypeError(f"{lmap.ty=} is not a function, so can't build its matrix.")


def bases(ty: Ty):
    match ty:

        case TyBool():
            basis_mat = torch.eye(2, device=device)
            basis_tt = basis_mat[:,0]
            basis_ff = basis_mat[:,1]
            return [TypedTensor(basis_tt, TyBool()), 
                    TypedTensor(basis_ff, TyBool())]
        
        case TyNat():
            basis_mat = torch.eye(10, device=device)
            basis_i = [basis_mat[:,i] for i in range(10)]
            return [TypedTensor(basis, TyNat()) for basis in basis_i]
        
        case TyFun(_, _):
            dim_ty = dim(ty)
            basis_mat = torch.eye(dim_ty, device=device)

            bases = []
            for i in range(dim_ty):
                bases.append(reshape_with_ty(basis_mat[:,[i]], ty))
            return bases

def apply_n(f, n):

    def composed(x):
        for _ in range(n):
            x = f(x)
        return x
    
    return composed



def dim(ty: Ty):
    match ty:
        case TyBool():
            return 2
        case TyNat():
            return 10
        case TyFun(ty1, ty2):
            return dim(ty1) * dim(ty2)

def reshape_with_ty(tens: Tensor, ty: Ty):
    match ty:
        case TyFun(ty_in, ty_out):
            return TypedTensor(torch.reshape(tens, (dim(ty_out), dim(ty_in))), ty)
        case _:
            return TypedTensor(tens, ty)
