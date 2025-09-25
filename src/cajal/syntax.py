from dataclasses import dataclass

# ------------------------------------- Types ------------------------------------- #
type Ty = TyBool | TyFun

@dataclass(eq=True, frozen=True, unsafe_hash=True)
class TyBool: ...

@dataclass(eq=True, frozen=True, unsafe_hash=True)
class TyFun: 
    ty1 : Ty
    ty2 : Ty

# ------------------------------------- Terms ------------------------------------- #
type Tm = TmVar | TmTrue | TmFalse | TmIf | TmFun | TmApp

@dataclass(eq=True, frozen=True, unsafe_hash=True)
class TmVar:
    name: str

@dataclass(eq=True, frozen=True, unsafe_hash=True)
class TmTrue: ...

@dataclass(eq=True, frozen=True, unsafe_hash=True)
class TmFalse: ...

@dataclass(eq=True, frozen=True, unsafe_hash=True)
class TmIf:
    tm1: Tm
    tm2: Tm
    tm3: Tm

@dataclass(eq=True, frozen=True, unsafe_hash=True)
class TmFun:
    name: str
    ty: Ty
    tm: Tm

@dataclass(eq=True, frozen=True, unsafe_hash=True)
class TmApp:
    tm1 : Tm
    tm2 : Tm