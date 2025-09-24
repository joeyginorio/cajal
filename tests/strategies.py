import hypothesis.strategies as st
from hypothesis import given, settings
from random import randint, shuffle
from random import seed as random_seed
from cajal.typing import *

# ---------------------------------  Programs  ------------------------------------ #

# Generate programs
def gen_prog(ctx: Ctx, ty: Ty):
    match ty:
        case TyBool():
            return gen_bool(ctx, ty)
        case TyFun(_, _):
            return gen_arrow(ctx, ty)
        case _:
            ...

# Generate programs of type Bool
def gen_bool(ctx: Ctx, ty: Ty):
    match len(ctx):
        case 0:
            return one_of_weighted([
                (gen_true(ctx, ty), 2),
                (gen_false(ctx, ty), 2),
                (gen_if(ctx, ty), 1)
            ])
        case 1:
            [(x, tyx)] = ctx.items()
            if tyx == ty:
                return one_of_weighted([
                    (gen_var(ctx, ty), 3),
                    (gen_if(ctx, ty), 1)
                ])
            else: 
                return one_of_weighted([
                    (gen_if(ctx, ty), 1)
                ])
        case _:
            return st.one_of([
                gen_if(ctx, ty)
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

def gen_var(ctx: Ctx, ty: Ty):

    if len(ctx) != 1:
        return st.nothing()

    [(x, tyx)] = ctx.items()
    if tyx != ty:
        return st.nothing()
    
    return st.just(TmVar(x))


def gen_true(ctx: Ctx, ty: Ty):

    if len(ctx) != 0 or ty != TyBool():
        return st.nothing()
    
    return st.just(TmTrue())


def gen_false(ctx: Ctx, ty: Ty):

    if len(ctx) != 0 or ty != TyBool():
        return st.nothing()
    
    return st.just(TmFalse())


# Generate if-then-else
@st.composite
def gen_if(draw, ctx: Ctx, ty: Ty):
    ctx1, ctx2 = split(ctx)
    
    tm1 = draw(gen_prog(ctx1, TyBool()))
    tm2 = draw(gen_prog(ctx2, ty))
    tm3 = draw(gen_prog(ctx2, ty))
    return TmIf(tm1, tm2, tm3)


@st.composite
def gen_fun(draw, ctx: Ctx, ty: Ty):
    match ty:
        case TyFun(ty1, ty2):
            x = draw(gen_fresh(ctx))
            # tm = draw(gen_tm(ctx | {x: ty1}, ty2))
            tm = draw(st.sampled_from([TmTrue(), TmFalse()]))
            return TmFun(x, ty1, tm)
        case _:
            return draw(st.nothing())
            


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


def debug_strategy(gen, n_examples=100, print=False):
    examples = []

    @settings(deadline=None, max_examples=1)
    @given(st.data())
    def draw_n(data):
        exs = [data.draw(gen) for _ in range(n_examples)]
        examples[:] = exs
        if print:
            for i, ex in enumerate(exs, 1):
                    print(f"Example {i}: {ex}")
    draw_n()
    return examples


def unique(xs):
    unique = []
    for x in xs:
        if x not in unique:
            unique.append(x)
    return unique
