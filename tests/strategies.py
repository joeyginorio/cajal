import hypothesis.strategies as st
from hypothesis import given, settings
from random import randint, shuffle, seed
from time import time
from cajal.typing import *


# ---------------------------------  Programs  ------------------------------------ #

# Generate programs
def gen_prog(ctx: Ctx, ty: Ty):
    match ty:
        case TyBool():
            return gen_prog_bool(ctx, ty)
        case TyFun(_, _):
            return gen_prog_fun(ctx, ty)
        case _:
            ...

# Generate programs of type Bool
def gen_prog_bool(ctx: Ctx, ty: Ty):
    match len(ctx):
        case 0:
            return one_of_weighted([
                (gen_tm_true(ctx, ty), 2),
                (gen_tm_false(ctx, ty), 2),
                (gen_tm_if(ctx, ty), 1)
            ])
        case 1:
            [(x, tyx)] = ctx.items()
            if tyx == ty:
                return one_of_weighted([
                    (gen_tm_var(ctx, ty), 8),
                    (gen_tm_if(ctx, ty), 1)
                ])
            else: 
                return one_of_weighted([
                    (gen_tm_if(ctx, ty), 1)
                ])
        case _:
            return st.one_of([
                gen_tm_if(ctx, ty)
            ])

# Generate programs of type A -> B
def gen_prog_fun(ctx: Ctx, ty: Ty):
    match len(ctx):
        case 0:
            return one_of_weighted([
                (gen_tm_fun(ctx, ty), 4),
                (gen_tm_if(ctx, ty), 1)
            ])
        case 1:
            [(x, tyx)] = ctx.items()
            if tyx == ty:
                return one_of_weighted([
                    (gen_tm_var(ctx, ty), 8),
                    (gen_tm_fun(ctx, ty), 4),
                    (gen_tm_if(ctx, ty), 1)
                ])
            else:
                return one_of_weighted([
                    (gen_tm_fun(ctx, ty), 4),
                    (gen_tm_if(ctx, ty), 1)
                ])
        case _:
            return one_of_weighted([
                (gen_tm_fun(ctx, ty), 4),
                (gen_tm_if(ctx, ty), 1)
            ])



# ---------------------------------  Syntax  -------------------------------------- #

# Generate types
gen_ty = st.recursive(
    st.builds(TyBool),
    lambda ty: st.builds(TyFun, ty, ty)
)

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
@st.composite
def gen_tm_if(draw, ctx: Ctx, ty: Ty):
    ctx1, ctx2 = split(ctx)
    
    tm1 = draw(gen_prog(ctx1, TyBool()))
    tm2 = draw(gen_prog(ctx2, ty))
    tm3 = draw(gen_prog(ctx2, ty))
    return TmIf(tm1, tm2, tm3)


@st.composite
def gen_tm_fun(draw, ctx: Ctx, ty: Ty):
    assert isinstance(ty, TyFun)

    x = draw(gen_fresh(ctx))
    tm = draw(gen_prog(ctx | {x: ty.ty1}, ty.ty2))
    return TmFun(x, ty.ty1, tm) 


# ---------------------------------  Helpers  --------------------------------------- #

# Generate fresh variable names, not already in context
def gen_fresh(ctx: Ctx):
    condition = lambda x : x not in ctx
    names = st.integers(min_value=0).map(lambda n : f"#x{n}")
    return names.filter(condition)


# Generate context split
def split(ctx: Ctx):
    seed()
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

    seed()
    n = randint(0, ubound-1)

    lbound = 0
    for gen, w in gens_ws:
        if lbound <= n < lbound + w:
            return gen
        lbound += w


def debug_strategy(gen, n_examples=100, timing=True):
    examples = []
    def draw_n():
        exs = [gen.example() for _ in range(n_examples)]
        examples[:] = exs

    if timing:
        start = time()
        draw_n()
        elapsed = time() - start

        print(f"Generated {n_examples} examples in {elapsed:.3f} milliseconds.")
        print(f"Generation rate: {(elapsed / n_examples)*1000 :.3f} msec/example.")
        print("------")
        n_unique = len(set(examples))
        print(f"Unique examples: {n_unique}.")
        print(f"Unique generation rate: {(elapsed / n_unique)*1000 :.3f} msec/example.")

    else:
        draw_n()
    
    return examples