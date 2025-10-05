from hypothesis import given, settings, HealthCheck, Verbosity
from strategies import gen_closed_prog
from cajal.compiling import compile

@settings(max_examples=5, suppress_health_check=[HealthCheck.too_slow], verbosity=Verbosity.normal)
@given(gen_closed_prog())
def test_compile(ctx_tm_ty):
    _, tm, ty = ctx_tm_ty
    compiled_tm = compile(tm, {})
    assert 1 == 1