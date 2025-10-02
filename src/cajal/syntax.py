from dataclasses import dataclass
from collections.abc import Mapping

# ------------------------------------- Types ------------------------------------- #

type Ty = TyBool | TyFun

@dataclass
class TyBool: ...

@dataclass
class TyFun: 
    ty1 : Ty
    ty2 : Ty

# ------------------------------------- Terms ------------------------------------- #

type Tm = TmVar | TmTrue | TmFalse | TmIf | TmFun | TmApp

@dataclass
class TmVar:
    name: str

@dataclass
class TmTrue: ...

@dataclass
class TmFalse: ...

@dataclass
class TmIf:
    tm1: Tm
    tm2: Tm
    tm3: Tm

@dataclass
class TmFun:
    name: str
    ty: Ty
    tm: Tm

@dataclass
class TmApp:
    tm1 : Tm
    tm2 : Tm


# ------------------------------------- Values ------------------------------------- #

type Val = VTrue | VFalse | VClosure

@dataclass
class VTrue: ...

@dataclass
class VFalse: ...

type Env = Mapping[str, tuple[Val, Ty]]

@dataclass
class VClosure:
    name: str
    ty: Ty
    tm: Tm
    env: Env

