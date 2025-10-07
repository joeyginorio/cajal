from cajal.syntax import *

def evaluate(tm: Tm, env: Env) -> Val:
    match tm:
        case TmVar(x):
            v = env[x]
            return v
        case TmTrue():
            return VTrue()
        case TmFalse():
            return VFalse()
        case TmZero():
            return VZero()
        case TmSucc(tm):
            v = evaluate(tm, env)
            return VSucc(v)
        case TmFun(x, ty, tm):
            return VClosure(x, ty, tm, env)
        case TmIter(tm1, name2, tm2, tm3):
            v3 = evaluate(tm3, env)
            match v3:
                case VZero():
                    return evaluate(tm1, env)
                case VSucc(v):
                    v_n_minus_1 = evaluate(TmIter(tm1, name2, tm2, v), env)
                    v_n = evaluate(tm2, env | {name2: v_n_minus_1})
                    return v_n
                case _:
                    raise TypeError(f"TmIter: {tm3=} doesn't evaluate to a nat, {v3=}.")
        case TmIf(tm1, tm2, tm3):
            v1 = evaluate(tm1, env)
            match v1:
                case VTrue():
                    return evaluate(tm2,env)
                case VFalse():
                    return evaluate(tm3, env)
                case _:
                    raise TypeError(f"TmIf: {tm1=} doesn't evaluate to a boolean, {v1=}.")
        case TmApp(tm1, tm2):
            v1 = evaluate(tm1, env)
            v2 = evaluate(tm2, env)
            match v1:
                case VClosure(x, ty, tm, c_env):
                    c_env |= {x: v2}
                    return evaluate(tm, c_env)
                case _:
                    raise TypeError(f"TmApp: {tm1=} doesn't evaluate to a closure, {v1=}.")
        case _:
            return tm