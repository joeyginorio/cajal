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

                outs = [base_val]
                cur = base_val
                for _ in range(1, len(n)):
                    cur = rec_f(cur)
                    outs.append(cur)

                total = n[0] * outs[0]
                for i in range(1, len(n)):
                    total += n[i] * outs[i]
                
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

def test1():
    tm = TmFun('x', TyBool(), TmVar('x'))
    check(tm, {})
    c_tm = compile(tm)
    e_tm = c_tm({})
    print(mat_of_lmap(e_tm).data)

def test2():
    tm = TmFun('x', TyBool(),
               TmIf(TmVar('x'),
                    TmTrue(),
                    TmFalse()))
    check(tm, {})
    c_tm = compile(tm)
    e_tm = c_tm({})
    print(mat_of_lmap(e_tm).data)

def test3():
    tm = TmFun('x', TyBool(),
               TmIf(TmVar('x'),
                    TmTrue(),
                    TmTrue()))
    check(tm, {})
    c_tm = compile(tm)
    e_tm = c_tm({})
    print(mat_of_lmap(e_tm).data)

def test4():
    tm = TmFun('x', TyFun(TyBool(), TyBool()), TmVar('x'))
    check(tm, {})
    c_tm = compile(tm)
    e_tm = c_tm({})
    print(mat_of_lmap(e_tm).data)

def test5():
    tm = TmFun('x', TyFun(TyBool(), TyBool()),
               TmIf(TmApp(TmVar('x'), TmTrue()),
                    TmFun('y', TyBool(), TmVar('y')),
                    TmFun('z', TyBool(), TmVar('z'))))
    check(tm, {})
    c_tm = compile(tm)
    e_tm = c_tm({})
    print(mat_of_lmap(e_tm).data)

def test6():
    tm = TmFun('x', TyFun(TyBool(), TyBool()),
               TmIf(TmApp(TmVar('x'), TmTrue()),
                    TmApp(TmFun('y', TyBool(), TmVar('y')), TmTrue()),
                    TmApp(TmFun('z', TyBool(), TmVar('z')), TmTrue())))
    check(tm, {})
    c_tm = compile(tm)
    e_tm = c_tm({})
    print(mat_of_lmap(e_tm).data)

def test7():
    iden = TypedTensor(torch.eye(4, device=device), TyFun(TyFun(TyBool(), TyBool()),TyFun(TyBool(), TyBool())))
    tm = TmApp(TmVar('f'), TmFun('x', TyBool(), TmVar('x')))
    check(tm, {'f' : TyFun(TyFun(TyBool(), TyBool()),TyFun(TyBool(), TyBool()))})
    tm_compiled = compile(tm)
    res = tm_compiled({'f': iden})
    return res

def test8():
    iden = TypedTensor(torch.eye(4, device=device), TyFun(TyFun(TyBool(), TyBool()),TyFun(TyBool(), TyBool())))
    arg = TypedTensor(torch.tensor([[1.,1.],[0.,0.]], device=device), TyFun(TyBool(), TyBool()))
    tm = TmApp(TmVar('f'), TmVar('x'))
    check(tm, {'f' : TyFun(TyFun(TyBool(), TyBool()), TyFun(TyBool(), TyBool())),
               'x': TyFun(TyBool(), TyBool())})
    tm_compiled = compile(tm)
    res = tm_compiled({'f': iden, 'x': arg})
    return res

def test9():
    arg = TypedTensor(torch.tensor([1.,1.], device=device), TyBool())
    tm = TmApp(TmFun('y', TyBool(), TmVar('y')), TmVar('x'))
    check(tm, {'x': TyBool()})
    tm_compiled = compile(tm)
    res = tm_compiled({'x': arg})
    return res

def test10():
    iden = TypedTensor(torch.eye(2, device=device), TyFun(TyBool(), TyBool()))
    tm = TmApp(TmVar('f'), TmFalse())
    check(tm, {'f' : TyFun(TyBool(), TyBool())})
    tm_compiled = compile(tm)
    res = tm_compiled({'f': iden})
    return res

def test11():
    tm = TmIter(TmTrue(),
            'y',
            TmApp(
                TmFun('x', TyBool(), TmIf(TmVar('x'), TmFalse(), TmTrue())), 
                TmVar('y')), 
            TmSucc(TmSucc(TmZero())))
    check(tm, {})
    c_tm = compile(tm)
    return c_tm({})

def test12():
    tm = TmFun('z', TyBool(),
               TmIter(TmVar('z'),
                'y',
                TmApp(
                    TmFun('x', TyBool(), TmIf(TmVar('x'), TmFalse(), TmTrue())), 
                    TmVar('y')), 
                TmSucc(TmSucc(TmZero()))))
    check(tm, {})
    c_tm = compile(tm)
    return c_tm({})

def test13():
    tm = TmFun('z', TyBool(),
               TmIter(TmVar('z'),
                'y',
                TmApp(
                    TmFun('x', TyBool(), TmIf(TmVar('x'), TmFalse(), TmTrue())), 
                    TmVar('y')), 
                    TmSucc(TmZero())))
    check(tm, {})
    c_tm = compile(tm)
    return c_tm({})

def test14():
    tm = TmFun('z', TyBool(),
               TmIter(TmVar('z'),
                'y',
                TmApp(
                    TmFun('x', TyBool(), TmIf(TmVar('x'), TmFalse(), TmTrue())), 
                    TmVar('y')), 
                    TmVar('n')))
    check(tm, {'n': TyNat()})
    c_tm = compile(tm)
    return c_tm({'n': torch.tensor([1.3, 2.3, 3.4], device=device)})


def test14():
    tm = TmFun('n', TyNat(), TmVar('n'))
    check(tm, {})
    c_tm = compile(tm)
    return c_tm({})

def test15():
    tm = TmFun('n', TyNat(), TmIter(TmSucc(TmVar('n')), 'y', TmSucc(TmVar('y')), TmZero()))
    check(tm, {})
    c_tm = compile(tm)
    return c_tm({})

def test16():
    tm = TmApp(
        TmVar('f'),
        TmSucc(TmZero()))
    iden = TypedTensor(torch.eye(10, device=device), TyFun(TyNat(), TyNat()))
    check(tm, {'f' : TyFun(TyNat(), TyNat())})
    c_tm = compile(tm)
    return c_tm({'f': iden})

def test17():
    tm = TmApp(
        TmVar('f'),
        TmSucc(TmZero()))
    iden = TypedTensor(torch.eye(10, device=device), TyFun(TyNat(), TyNat()))
    f_iden = LinearMap(lambda x: (4 * (iden + iden)) @ x, TyFun(TyNat(), TyNat()))
    check(tm, {'f' : TyFun(TyNat(), TyNat())})
    c_tm = compile(tm)
    return c_tm({'f': f_iden})

def test18():
    tm = TmApp(
        TmVar('f'),
        TmSucc(TmZero()))
    iden = TypedTensor(torch.eye(10, device=device), TyFun(TyNat(), TyNat()))
    f_iden = LinearMap(lambda x: (4 * (iden + iden)) @ x, TyBool())
    check(tm, {'f' : TyFun(TyNat(), TyNat())})
    c_tm = compile(tm)
    return c_tm({'f': f_iden})

def test19():
    base_val = TypedTensor(torch.eye(10, device=device).unsqueeze(0), TyNat())
    n_val = TypedTensor(torch.ones(10, device=device), TyNat())

    tm = TmIter(TmVar('base'), 
                'y', 
                TmApp(TmVar('f'), TmVar('y')),
                TmVar('n'))
    conv = torch.nn.Conv2d(1,1,
                           kernel_size=7, 
                           stride=1, 
                           padding=3, 
                           bias=False, 
                           padding_mode="zeros")
    conv.to(device)

    def f_val(x):
        return TypedTensor(conv(x.data), x.ty)
    c_tm = compile(tm)
    
    return c_tm({'f': f_val, 'base': base_val, 'n': n_val})

def test20(weights):
    base_val = TypedTensor(torch.eye(10, device=device).unsqueeze(0), TyNat())
    n_val = TypedTensor(torch.ones(10, device=device), TyNat())

    tm = TmIter(TmVar('base'), 
                'y', 
                TmApp(TmVar('f'), TmVar('y')),
                TmVar('n'))
    conv = torch.nn.Conv2d(1,1,
                           kernel_size=7, 
                           stride=1, 
                           padding=3, 
                           bias=False, 
                           padding_mode="zeros")
    conv.to(device)
    conv.weight = weights

    def f_val(x):
        return TypedTensor(conv(x.data), TyFun(TyBool(), TyBool()))
    c_tm = compile(tm)
    c_tm = torch.vmap(c_tm,
                      in_dims=({'f': None,
                                'base': TypedTensor(0, None),
                                'n': TypedTensor(0, None)},),
                      out_dims=TypedTensor(0, None))
    
    base_val = torch.randn(5,1,10,10,device=device)
    num_val = torch.randn(5,10,device=device)
    return c_tm({'f': f_val,
                 'base': TypedTensor(base_val, TyBool()),
                 'n': TypedTensor(num_val, TyBool())})
