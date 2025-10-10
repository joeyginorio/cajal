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
            tm.ty_checked = ctx[x]
            return tm.ty_checked, ctx_remain

        case TmTrue():
            tm.ty_checked = TyBool()
            return tm.ty_checked, ctx

        case TmFalse():
            tm.ty_checked = TyBool()
            return tm.ty_checked, ctx
        
        case TmZero():
            tm.ty_checked = TyNat()
            return tm.ty_checked, ctx
    
        case TmSucc(tm):
            ty, ctx_remain = _check(tm, ctx)
            match ty:
                case TyNat():
                    tm.ty_checked = TyNat()
                    return tm.ty_checked, ctx_remain
                case _:
                    raise TypeError(f"TmSucc: Successor not applied to a Nat, {tm=} is a {ty=}.")

        case TmFun(x, ty1, e):
            ctx_extend = ctx | {x: ty1}
            ty2, ctx_remain = _check(e, ctx_extend)
            tm.ty_checked = TyFun(ty1, ty2)
            return tm.ty_checked, ctx_remain

        case TmIter(e1, y, e2, e3):
            ty3, ctx_remain3 = _check(e3, ctx)
            match ty3:
                case TyNat():
                    ty1, ctx_remain1 = _check(e1, ctx_remain3)
                    ty2, ctx_remain2 = _check(e2, ctx_remain1 | {y: ty1})
                    tm.ty_checked = ty2
                    return tm.ty_checked, ctx_remain2
                case _:
                    raise TypeError(f"TmIter: Iter not applied to a Nat, {tm3=} is a {ty3=}.")

        case TmApp(e1, e2):
            ty1, ctx_remain = _check(e1, ctx)

            match ty1:
                case TyFun(ty11, ty12):
                    ty2, ctx_remain = _check(e2, ctx_remain)
                    if ty11 != ty2:
                        raise TypeError(f"TmApp: Function's input must be a {ty11}, but is a {ty2}.")
                    
                    tm.ty_checked = ty12
                    return tm.ty_checked, ctx_remain
                
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

            tm.ty_checked = ty3
            return tm.ty_checked, ctx_remain3
        
        case _:
            raise TypeError(f"_check: Unhandled term {tm=}.")

def check(tm: Tm, ctx: Ctx) -> Ty:
    ty, ctx_remain = _check(tm, ctx)
    if ctx_remain:
        raise TypeError(f"check: Some context remains {ctx_remain=}, when checking {tm} with {ctx}.")
    tm.ty_checked = ty
    return tm.ty_checked

def check_val(val: Val) -> Ty:
    match val:
        case VTrue():
            val.ty_checked = TyBool()
            return val.ty_checked
        case VFalse():
            val.ty_checked = TyBool()
            return val.ty_checked
        case VZero():
            val.ty_checked = TyNat()
            return val.ty_checked
        case VSucc(v):
            val.ty_checked = check_val(v)
            return val.ty_checked
        case VClosure(x, ty, tm, c_env):
            ctx = {y: check_val(y_val) for (y, y_val) in c_env.items()}
            ctx |= {x: ty}
            ty_body = check(tm, ctx)
            val.ty_checked = TyFun(ty, ty_body)
            return TyFun(ty, ty_body)