from hypothesis import given, settings, HealthCheck, Verbosity, Phase
from strategies import gen_closed_prog_observable
from cajal.compiling import compile, compile_val
from cajal.evaluating import evaluate

@settings(max_examples=100, 
          suppress_health_check=[HealthCheck.too_slow], 
          verbosity=Verbosity.normal,
          deadline=None)
@given(gen_closed_prog_observable())
def test_compile(ctx_tm_ty):

    _, tm, _ = ctx_tm_ty
    v = evaluate(tm, {})

    compiled_tm = compile(tm, {})
    compiled_v = compile_val(v)

    assert all(compiled_tm == compiled_v)