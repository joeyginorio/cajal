import hypothesis.strategies as st
from hypothesis import given, settings
from random import randint, shuffle, seed
from time import time
from cajal.typing import *


# ---------------------------------  Programs  ------------------------------------ #

type NCtx = Ctx
type PCtx = list[tuple[str, Ty]]

# Generate programs
def gen_prog(ctx_neg: NCtx, ctx_pos : PCtx, ty: Ty):
    match ty:
        case TyBool():
            return gen_prog_bool(ctx_neg, ctx_pos, ty)
        case TyFun(_, _):
            ...

# Generate programs of type Bool
def gen_prog_bool(ctx_neg: NCtx, ctx_pos: PCtx, ty_goal: Ty):
    
    if len(ctx_neg) == 0 and len(ctx_pos) == 0:
        return st.one_of([st.just(TmTrue()), st.just(TmFalse())])
    
    if ctx_pos:
        (x, ty_pos), *ctx_pos_remain = ctx_pos
        match ty_pos:
            case TyBool():
                condition = st.just(TmVar(x))
                branch = gen_prog(ctx_neg, ctx_pos_remain, ty_goal)
                return st.builds(TmIf, condition, branch, branch)
            case _:
                # TODO: Nat
                ...

    # TODO: Focus on negative

# ---------------------------------  Syntax  -------------------------------------- #

def gen_ty():
    return one_of_weighted([
            (gen_ty_bool(), 4),
            (gen_ty_fun(), 1)
        ])

def gen_ty_bool():
    return st.just(TyBool())

def gen_ty_fun():
    return st.builds(TyFun, gen_ty(), gen_ty())


# Generate contexts
@st.composite
def gen_ctx(draw):
    xs = draw(st.lists(gen_fresh({})))
    tys = draw(st.lists(gen_ty))
    return dict(zip(xs,tys))


# ---------------------------------  Typing  --------------------------------------- #

def gen_tm_var(ctx: Ctx, ty: Ty):
    assert len(ctx) == 1
    
    [(x, tyx)] = ctx.items()
    assert tyx == ty
    
    return st.just(TmVar(x))


def gen_tm_true(ctx: Ctx, ty: Ty):
    assert len(ctx) == 0 and ty == TyBool()
    return st.just(TmTrue())


def gen_tm_false(ctx: Ctx, ty: Ty):
    assert len(ctx) == 0 and ty == TyBool()
    return st.just(TmFalse())


# Generate if-then-else
def gen_tm_if(ctx: Ctx, ty: Ty):
    ctx1, ctx2 = split(ctx)
    
    condition = gen_prog(ctx1, TyBool())
    then_branch = gen_prog(ctx2, ty)
    else_branch = gen_prog(ctx2, ty)
    return st.builds(TmIf, condition, then_branch, else_branch)


@st.composite
def gen_tm_fun(draw, ctx: Ctx, ty: Ty):
    assert isinstance(ty, TyFun)

    x = draw(gen_fresh(ctx))
    tm = draw(gen_prog(ctx | {x: ty.ty1}, ty.ty2))
    return TmFun(x, ty.ty1, tm) 


@st.composite
def gen_tm_app(draw, ctx: Ctx, ty_out: Ty):
    ctx1, ctx2 = split(ctx)

    ty_in = draw(gen_ty())
    tm1 = draw(gen_prog(ctx1, TyFun(ty_in, ty_out)))
    tm2 = draw(gen_prog(ctx2, ty_in))

    return TmApp(tm1, tm2)



# ---------------------------------  Helpers  --------------------------------------- #

# Generate fresh variable names, not already in context
def gen_fresh(ctx: Ctx):
    condition = lambda x : x not in ctx
    names = st.integers(min_value=0).map(lambda n : f"#x{n}")
    return names.filter(condition)


# Generate context split
def split(ctx: Ctx):

    xs = list(ctx.keys())
    shuffle(xs)

    if len(xs) <= 1:
        split_idx = 0
    else:
        split_idx = randint(1, len(xs)-1)
        
    xs1 = xs[:split_idx]
    xs2 = xs[split_idx:]

    ctx1 = {x1: ctx[x1] for x1 in xs1}
    ctx2 = {x2: ctx[x2] for x2 in xs2}

    return ctx1, ctx2


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
def neg_pos(ctx: Ctx) -> tuple[Ctx, Ctx]:

    positive_ctx = []
    negative_ctx = {}
    for (x, ty) in ctx.items():
        if positive(ty):
            positive_ctx += (x, ty)
        if not positive(ty):
            negative_ctx |= {x: ty}

    return positive_ctx, negative_ctx

def positive(ty: Ty) -> bool:
    match ty:
        case TyBool():
            return True
        case TyFun(_, _):
            return False


