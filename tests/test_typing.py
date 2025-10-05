from hypothesis import given, settings, HealthCheck, Verbosity
from strategies import gen_closed_prog
from cajal.typing import check_val
from cajal.evaluating import evaluate

@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow], verbosity=Verbosity.normal)
@given(gen_closed_prog())
def test_eval(ctx_tm_ty):
    _, tm, ty_term = ctx_tm_ty
    v = evaluate(tm, {})
    ty_val = check_val(v)
    assert ty_term == ty_val