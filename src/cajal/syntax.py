from dataclasses import dataclass, field
from collections.abc import Mapping
from typing import Optional

# ------------------------------------- Types ------------------------------------- #

type Ty = (  TyBool
           | TyNat
           | TyFun
           )

@dataclass
class TyBool: ...

@dataclass
class TyNat: ...

@dataclass
class TyFun: 
    ty1 : Ty
    ty2 : Ty

# ------------------------------------- Terms ------------------------------------- #

type Tm = (  TmVar
           | TmTrue 
           | TmFalse 
           | TmZero 
           | TmSucc 
           | TmIf 
           | TmFun
           | TmApp
           )

@dataclass
class Term:
    ty_checked: Optional[Ty] = field(default=None, init=False)

@dataclass
class TmVar(Term):
    name: str

@dataclass
class TmTrue(Term): ...

@dataclass
class TmFalse(Term): ...

@dataclass
class TmZero(Term): ...

@dataclass
class TmSucc(Term):
    tm: Tm

@dataclass
class TmIter(Term):
    tm1: Tm
    name2: str
    tm2: Tm
    tm3: Tm

@dataclass
class TmIf(Term):
    tm1: Tm
    tm2: Tm
    tm3: Tm

@dataclass
class TmFun(Term):
    name: str
    ty: Ty
    tm: Tm

@dataclass
class TmApp(Term):
    tm1 : Tm
    tm2 : Tm


# ------------------------------------- Values ------------------------------------- #

type Val = (  VTrue 
            | VFalse
            | VZero
            | VSucc
            | VClosure
            )

@dataclass
class VTrue(Term): ...

@dataclass
class VFalse(Term): ...

@dataclass
class VZero(Term): ...

@dataclass
class VSucc(Term):
    v: Val

type Env = Mapping[str, tuple[Val, Ty]]

@dataclass
class VClosure(Term):
    name: str
    ty: Ty
    tm: Tm
    env: Env

