from hypothesis import given, settings, HealthCheck, Verbosity
from strategies import gen_prog, gen_prog_observable, gen_closed_prog_observable
from cajal.typing import check
from cajal.syntax import TyBool

@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow], verbosity=Verbosity.normal)
@given(gen_prog())
def test_gen_prog(ctx_tm_ty):
    ctx, tm, ty = ctx_tm_ty
    ty_check = check(tm, ctx)
    assert ty == ty_check

@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow], verbosity=Verbosity.normal)
@given(gen_prog_observable())
def test_gen_prog_observable(ctx_tm_ty):
    ctx, tm, ty = ctx_tm_ty
    ty_check = check(tm, ctx)
    assert ty == ty_check == TyBool()

@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow], verbosity=Verbosity.normal)
@given(gen_closed_prog_observable())
def test_gen_closed_prog_observable(ctx_tm_ty):
    ctx, tm, ty = ctx_tm_ty
    ty_check = check(tm, ctx)
    assert ty == ty_check == TyBool()