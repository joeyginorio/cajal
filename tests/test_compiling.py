from hypothesis import given, settings, HealthCheck, Verbosity, Phase
from strategies import gen_closed_prog_observable, gen_closed_prog_observable_except
from cajal.compiling import compile, compile_val
from cajal.evaluating import evaluate
from cajal.syntax import *
from torch import tensor, isfinite

@settings(max_examples=100, 
          suppress_health_check=[HealthCheck.too_slow], 
          verbosity=Verbosity.normal,
          deadline=None)
@given(gen_closed_prog_observable())
def test_compiler_correctness(ctx_tm_ty):

    _, tm, _ = ctx_tm_ty
    v = evaluate(tm, {})

    compiled_tm = compile(tm, {})
    compiled_v = compile_val(v)

    assert all(compiled_tm == compiled_v)

@settings(max_examples=100, 
          suppress_health_check=[HealthCheck.too_slow],
          verbosity=Verbosity.normal,
          deadline=None)
@given(gen_closed_prog_observable_except({'x': TyBool()}))
def test_grad_exists(ctx_tm_ty):
    _, tm, _ = ctx_tm_ty
    env = {'x' : tensor([1., 2.], requires_grad=True)}
    y = compile(tm, env)
    y.sum().backward()
    assert all(isfinite(env['x'].grad))

def test_grad_not():
    ''' 
    f(x) = x[0] * [0,1]^T + x[1] * [1,0]^T
    df/x0 = [0,1]^T
    df/x1 = [1,0]^T
    df/dx = [0, 1]
          = [1, 0] 
    '''
    env = {'x' : tensor([1., 2.], requires_grad=True)}
    y = compile(TmIf(TmVar('x'), TmFalse(), TmTrue()), env)
    y.sum().backward()
    assert all(env['x'].grad == tensor([1., 1.]))

