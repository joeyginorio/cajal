import hypothesis.strategies as st
from hypothesis import given, settings
from random import randint, shuffle, seed
from time import time
from cajal.typing import *

# ---------------------------------  Programs  ------------------------------------ #

type NCtx = list[tuple[Tm, Ty]]
type PCtx = list[tuple[str, Ty]]

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
        name = draw(gen_fresh(ctx_pos))
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
        name = draw(gen_fresh(names_neg))
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
    xs = draw(st.lists(gen_fresh({}), max_size=8))
    tys = draw(st.lists(gen_ty(), max_size=8))
    return list(zip(xs,tys))


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
    
    condition = gen_prog_ty(ctx1, TyBool())
    then_branch = gen_prog_ty(ctx2, ty)
    else_branch = gen_prog_ty(ctx2, ty)
    return st.builds(TmIf, condition, then_branch, else_branch)


@st.composite
def gen_tm_fun(draw, ctx: Ctx, ty: Ty):
    assert isinstance(ty, TyFun)

    x = draw(gen_fresh(ctx))
    tm = draw(gen_prog_ty(ctx | {x: ty.ty1}, ty.ty2))
    return TmFun(x, ty.ty1, tm) 


@st.composite
def gen_tm_app(draw, ctx: Ctx, ty_out: Ty):
    ctx1, ctx2 = split(ctx)

    ty_in = draw(gen_ty())
    tm1 = draw(gen_prog_ty(ctx1, TyFun(ty_in, ty_out)))
    tm2 = draw(gen_prog_ty(ctx2, ty_in))

    return TmApp(tm1, tm2)



# ---------------------------------  Helpers  --------------------------------------- #

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


