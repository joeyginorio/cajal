from dataclasses import dataclass
from collections.abc import Mapping
from cajal.syntax import *

# ------------------------------------- Typing ------------------------------------- #
type Ctx = Mapping[str, Ty]

def _check(tm: Tm, ctx: Ctx) -> tuple[Ty, Ctx]:
    match tm:
        case TmVar(x):
            if x not in ctx:
                raise TypeError(f"TmVar: {x} not in {ctx=}.")

            ctx_remain = {y: ty for (y, ty) in ctx.items() if y != x}
            return ctx[x], ctx_remain

        case TmTrue():
            return TyBool(), ctx

        case TmFalse():
            return TyBool(), ctx
        
        case TmZero():
            return TyNat(), ctx
    
        case TmSucc(tm):
            ty, ctx_remain = _check(tm, ctx)
            match ty:
                case TyNat():
                    return TyNat(), ctx_remain
                case _:
                    raise TypeError(f"TmSucc: Successor not applied to a Nat, {tm=} is a {ty=}.")

        case TmFun(x, ty1, e):
            ctx_extend = ctx | {x: ty1}
            ty2, ctx_remain = _check(e, ctx_extend)
            return TyFun(ty1, ty2), ctx_remain

        case TmIter(e1, y, e2, e3):
            ty3, ctx_remain3 = _check(e3, ctx)
            match ty3:
                case TyNat():
                    ty1, ctx_remain1 = _check(e1, ctx_remain3)
                    ty2, ctx_remain2 = _check(e2, ctx_remain1 | {y: ty1})
                    return ty2, ctx_remain2
                case _:
                    raise TypeError(f"TmIter: Iter not applied to a Nat, {tm3=} is a {ty3=}.")

        case TmApp(e1, e2):
            ty1, ctx_remain = _check(e1, ctx)

            match ty1:
                case TyFun(ty11, ty12):
                    ty2, ctx_remain = _check(e2, ctx_remain)
                    if ty11 != ty2:
                        raise TypeError(f"TmApp: Function's input must be a {ty11}, but is a {ty2}.")
                    return ty12, ctx_remain
                
                case _:
                    raise TypeError(f"TmApp: LHS isn't a function, it's a {ty1}.")
                
        case TmIf(e1, e2, e3):
            ty1, ctx_remain1 = _check(e1, ctx)
            if ty1 != TyBool():
                raise TypeError(f"TmIf: Condition not a boolean, is a {ty1}.")

            ty2, ctx_remain2 = _check(e2, ctx_remain1)
            ty3, ctx_remain3 = _check(e3, ctx_remain1)
            if ty2 != ty3:
                raise TypeError(f"TmIf: If-branch returns a {ty2}, but else-branch returns a {ty3}.")
            if ctx_remain2 != ctx_remain3:
                raise TypeError(f"TmIf: If-branch leaves {ctx_remain2=}, but else-branch leaves {ctx_remain3=}.")

            return ty3, ctx_remain3
        
        case _:
            raise TypeError(f"_check: Unhandled term {tm=}.")

def check(tm: Tm, ctx: Ctx) -> Ty:
    ty, ctx_remain = _check(tm, ctx)
    if ctx_remain:
        raise TypeError(f"check: Some context remains {ctx_remain=}, when checking {tm} with {ctx}.")
    return ty

def check_val(val: Val) -> Ty:
    match val:
        case VTrue():
            return TyBool()
        case VFalse():
            return TyBool()
        case VClosure(x, ty, tm, c_env):
            ctx = {y: check_val(tm_y) for (y, tm_y) in c_env.items()}
            ctx |= {x: ty}
            return TyFun(ty, check(tm, ctx))