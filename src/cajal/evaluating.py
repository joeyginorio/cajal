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
        case TmFun(x, ty, tm):
            return VClosure(x, ty, tm, env)
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