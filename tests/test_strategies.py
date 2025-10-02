from hypothesis import given, settings, HealthCheck, Verbosity
from strategies import gen_prog
from cajal.typing import check

@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow], verbosity=Verbosity.normal)
@given(gen_prog())
def test_gen_prog(ctx_tm_ty):
    ctx, tm, ty = ctx_tm_ty
    ty_check = check(tm, ctx)
    assert ty == ty_check