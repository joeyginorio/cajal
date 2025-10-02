from cajal.syntax import *

def eval(tm: Ty, env: Env) -> Val:
    match tm:
        case TmVar(x):
            v, _ = env[x]
            return v
        case TmTrue():
            return VTrue()
        case TmFalse():
            return VFalse()
        case TmFun(x, ty, tm):
            return VClosure(x, ty, tm, env)
        case TmIf(tm1, tm2, tm3):
            v1 = eval(tm1, env)
            match v1:
                case VTrue():
                    return eval(tm2,env)
                case VFalse():
                    return eval(tm3, env)
                case _:
                    raise TypError(f"TmIf: {tm1=} doesn't evaluate to a boolean, {v1=}.")
        case TmApp(tm1, tm2):
            v1 = eval(tm1, env)
            v2 = eval(tm2, env)
            match v1:
                case VClosure(x, ty, tm, c_env):
                    c_env |= {x: (v2, ty)}
                    return eval(tm, c_env)
                case _:
                    raise TypeError(f"TmApp: {tm1=} doesn't evaluate to a closure, {v1=}.")