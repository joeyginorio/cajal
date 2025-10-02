import hypothesis.strategies as st
from hypothesis import given, settings, HealthCheck, Verbosity
from random import randint, shuffle, seed
from time import time
from cajal.typing import *
from cajal.evaluating import *

# ---------------------------------  Programs  ------------------------------------ #

type NCtx = list[tuple[Tm, Ty]]
type PCtx = list[tuple[str, Ty]]

@st.composite
def gen_closed_prog(draw):
    ctx, tm, ty_goal = draw(gen_prog())
    for (x, ty) in ctx.items():
        v = draw(gen_prog_ty([], [], ty))
        tm = capture_subst(tm, x, v)
    ty = check(tm, {})
    assert ty == ty_goal
    return {}, tm, ty

@st.composite
def gen_prog(draw):
    ctx = draw(gen_ctx())
    ctx_neg, ctx_pos = neg_pos(ctx)

    ty = draw(gen_ty()) 
    tm = draw(gen_prog_ty(ctx_neg, ctx_pos, ty))
    return dict(ctx), tm, ty


# Generate programs 
def gen_prog_ty(ctx_neg: NCtx, ctx_pos : PCtx, ty: Ty):
    match ty:
        case TyBool():
            return gen_prog_bool(ctx_neg, ctx_pos)
        case TyFun(ty1, ty2):
            return gen_prog_fun(ctx_neg, ctx_pos, ty1, ty2)

# Generate programs of type Bool
@st.composite
def gen_prog_bool(draw, ctx_neg: NCtx, ctx_pos: PCtx):

    if ctx_pos:
        (x, ty_pos), *ctx_pos_remain = ctx_pos
        match ty_pos:
            case TyBool():
                condition = TmVar(x) if isinstance(x, str) else x
                then_branch = draw(gen_prog_bool(ctx_neg, ctx_pos_remain))
                else_branch = draw(gen_prog_bool(ctx_neg, ctx_pos_remain))
                return TmIf(condition, then_branch, else_branch)
            case _:
                # TODO: Nat
                ...

    if ctx_neg:
        (tm, ty_neg), *ctx_neg_remain = ctx_neg
        match ty_neg:

            case TyFun(ty_in, ty_out):
                input = draw(gen_prog_ty(ctx_neg_remain, [], ty_in))

                if positive(ty_out):
                    return draw(gen_prog_bool([], [(TmApp(tm, input), ty_out)])) 
                
                else:
                    ctx_neg = (TmApp(tm, input), ty_out), *ctx_neg_remain
                    return draw(gen_prog_bool([(TmApp(tm, input), ty_out)], []))

            case TyBool():
                raise TypeError("gen_prog_bool: Bool is positive but inside a negative context.")

    return draw(st.one_of([st.just(TmTrue()), st.just(TmFalse())]))

# Generate programs of type A -o B
@st.composite
def gen_prog_fun(draw, ctx_neg: NCtx, ctx_pos: PCtx, ty_in: Ty, ty_out: Ty):

    if positive(ty_in):
        names_neg = []
        for x, _ in ctx_neg:
            if isinstance(x, str):
                names_neg += [x]
            else: 
                names_neg += tm_names(x)
        name = gen_fresh(ctx_pos + names_neg)
        ctx_pos = [(name, ty_in)] + ctx_pos
        body = draw(gen_prog_ty(ctx_neg, ctx_pos, ty_out))
        return TmFun(name, ty_in, body)

    else:
        names_neg = []
        for x, _ in ctx_neg:
            if isinstance(x, str):
                names_neg += [x]
            else: 
                names_neg += tm_names(x)
        name = gen_fresh(ctx_pos + names_neg)
        ctx_neg = [(TmVar(name), ty_in)] + ctx_neg
        body = draw(gen_prog_ty(ctx_neg, ctx_pos, ty_out))
        return TmFun(name, ty_in, body)


# ---------------------------------  Syntax  -------------------------------------- #

@st.composite
def gen_ty(draw):
    return draw(st.one_of(gen_ty_bool(), gen_ty_fun()))

def gen_ty_bool():
    return st.just(TyBool())

@st.composite
def gen_ty_fun(draw):
    ty1 = draw(gen_ty())
    ty2 = draw(gen_ty())
    return TyFun(ty1, ty2)

# Generate contexts
@st.composite
def gen_ctx(draw):
    max_size = randint(0,6)

    xs = []
    tys = []
    for i in range(max_size):
        x = gen_fresh(xs)
        xs.append(x)
        tys.append(draw(gen_ty()))

    return list(zip(xs,tys))


# ---------------------------------  Helpers  --------------------------------------- #


def capture_subst(tm: Tm, x: str, v: Tm) -> Tm:
    match tm:
        case TmVar(y):
            if x == y:
                return v
            else:
                return TmVar(y)
        case TmTrue():
            return TmTrue()
        case TmFalse():
            return TmFalse()
        case TmFun(y, ty, tm_body):
            tm_subst = capture_subst(tm_body, x, v)
            return TmFun(y, ty, tm_subst)
        case TmApp(tm1, tm2):
            tm1_subst = capture_subst(tm1, x, v)
            tm2_subst = capture_subst(tm2, x, v)
            return TmApp(tm1_subst, tm2_subst)
        case TmIf(tm1, tm2, tm3):
            tm1_subst = capture_subst(tm1, x, v)
            tm2_subst = capture_subst(tm2, x, v)
            tm3_subst = capture_subst(tm3, x, v)
            return TmIf(tm1_subst, tm2_subst, tm3_subst)


def tm_names(tm: Tm) -> list[str]:
    match tm:
        case TmVar(x):
            return [x]
        case TmTrue():
            return []
        case TmFalse():
            return []
        case TmFun(x, ty, tm):
            return [x] + tm_names(tm)
        case TmApp(tm1, tm2):
            return tm_names(tm1) + tm_names(tm2)
        case TmIf(tm1, tm2, tm3):
            return tm_names(tm1) + tm_names(tm2) + tm_names(tm3)


# Generate fresh variable names, not already in context
def gen_fresh(ctx: Ctx):
    n = randint(0,1000000000)
    
    if f"#x{n}" not in ctx:
        return f"x{n}"
    
    return gen_fresh(ctx)


def one_of_weighted(gens_ws):

    _, ws = list(zip(*gens_ws))
    ubound = sum(ws)

    n = randint(0, ubound-1)

    lbound = 0
    for gen, w in gens_ws:
        if lbound <= n < lbound + w:
            return gen
        lbound += w

# Partition a context into 'negative' and 'positive' assumptions
def neg_pos(ctx: list[tuple[str | Tm, Ty]]) -> tuple[Ctx, Ctx]:

    negative_ctx = []
    positive_ctx = []
    for (x, ty) in ctx:
        if positive(ty):
            positive_ctx += [(x, ty)]
        if not positive(ty):
            negative_ctx += [(TmVar(x), ty)]

    return negative_ctx, positive_ctx
    

def positive(ty: Ty) -> bool:
    match ty:
        case TyBool():
            return True
        case TyFun(_, _):
            return False


@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow], verbosity=Verbosity.normal)
@given(gen_prog())
def test_gen_prog(ctx_tm_ty):
    ctx, tm, ty = ctx_tm_ty
    ty_check = check(tm, ctx)
    assert ty == ty_check


@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow], verbosity=Verbosity.normal)
@given(gen_closed_prog())
def test_eval(ctx_tm_ty):
    _, tm, ty_term = ctx_tm_ty
    v = eval(tm, {})
    ty_val = check_val(v)
    assert ty_term == ty_val