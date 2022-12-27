---  Cajal: Graded coeffects over 1  ---
---  ==============================  ---
---  Joey Velez-Ginorio              ---

module Cajal where

import Data.Text (Text, pack)
import Prettyprinter
import Control.Monad.Trans (MonadTrans (lift))
import Control.Monad.Trans.Reader (Reader,
                                   ReaderT,
                                   ask,
                                   local,
                                   runReader,
                                   runReaderT)

import qualified Data.Set as Set

---  ==========================  Syntax  =============================  ---

---  Identifiers ---

type Id = Text

---  Types  ---

data Ty = TyUnit
        | TyAbs Ty Ty
        deriving (Show, Eq)

---  Terms  ---

data Tm = TmUnit
        | TmVar Id
        | TmAbs Id Ty Tm
        | TmApp Tm Tm
        deriving Show

---  Contexts  ---

type Ctx = [(Id,Ty)]

---  Pretty printing  ---

instance Pretty Ty where
  pretty TyUnit = pretty "1"
  pretty (TyAbs t1 t2) = pretty t1 <+> pretty "->" <+> pretty t2

instance Pretty Tm where
  pretty TmUnit = pretty "*"
  pretty (TmVar x) = pretty x
  pretty (TmAbs x ty tm) =
    hsep [pretty "lam", pretty x, pretty ":", pretty ty, pretty "."] <+>
    line <+>
    indent 2 (pretty tm)
  pretty (TmApp tm1 tm2) = hsep [pretty tm1, pretty tm2]


---  ==========================  Dynamics  ===========================  ---

---  Generate infinite list of fresh variables  ---

ids :: [Id]
ids = (\n -> pack ('#' : 'x' : show n)) <$> [0..]

--- Collect the free variables in a term  ---

fvs :: Tm -> Set.Set Id
fvs TmUnit             = Set.empty
fvs (TmVar x)          = Set.insert x Set.empty
fvs (TmAbs x _ tm)     = Set.delete x $ fvs tm
fvs (TmApp tm1 tm2)    = fvs tm1 `Set.union` fvs tm2

---  Convert all x's to y's in a term  ---

aconv :: Id -> Id -> Tm -> Tm
aconv _ _ TmUnit           = TmUnit
aconv x y (TmVar z)        | x == z    = TmVar y
                           | otherwise = TmVar z
aconv x y (TmAbs z ty tm)  | x == z    = TmAbs y ty (aconv x y tm)
                           | otherwise = TmAbs z ty (aconv x y tm)
aconv x y (TmApp tm1 tm2)  = TmApp (aconv x y tm1) (aconv x y tm2)

---  Capture-avoiding substitution  ---

type ETm = Reader [Id] Tm

subst :: Id -> Tm -> Tm -> ETm
subst _ _ TmUnit          = return TmUnit
subst x t (TmVar y)
  | x == y                = return t
  | otherwise             = return $ TmVar y
subst x t s@(TmAbs y ty tm)
  | x == y                = return $ TmAbs y ty tm
  | Set.member y (fvs t)  = do ids' <- ask
                               let z  = head ids'
                               let s' = aconv y z s
                               tm' <- local tail (subst x t s')
                               return tm'
  | otherwise             = do tm' <- subst x t tm
                               return $ TmAbs y ty tm'
subst x t (TmApp tm1 tm2) = do tm1' <- subst x t tm1
                               tm2' <- subst x t tm2
                               return $ TmApp tm1' tm2'

---  Call-by-name evaluation  ---

eval' :: Tm -> ETm
eval' (TmApp (TmAbs x _ tm1) tm2) = do tm <- subst x tm2 tm1
                                       eval' tm
eval' (TmApp tm1 tm2)             = do tm1' <- eval' tm1
                                       eval' $ TmApp tm1' tm2
eval' tm = return tm

eval :: Tm -> Tm
eval tm = runReader (eval' tm) ids


---  ==========================  Statics  ============================  ---

---  Typing errors  ---

data Err = EVar  Id      -- Variable not in context
         | EAbs1 Tm Tm   -- Second term not valid input to first term
         | EAbs2 Tm      -- First term isn't a funtion
         deriving Show

---  TcTy: Reader + Either Monad Stack  ---
--   (1) 'Reader' passes around the context.
--   (2) 'Either' passes around informative typecheck errors.

type TcTy = ReaderT Ctx (Either Err) Ty

find :: Id -> TcTy
find x = do ctx <- ask
            lift $ case lookup x ctx of
                     Nothing -> Left $ EVar x
                     Just ty -> Right ty


tyCheck :: Tm -> TcTy
tyCheck TmUnit             = return TyUnit
tyCheck (TmVar x)          = find x
tyCheck (TmAbs x ty1 tm)   = do ty2 <- local ((x,ty1):) $ tyCheck tm
                                return $ TyAbs ty1 ty2
tyCheck (TmApp tm1 tm2)    = do ty1 <- tyCheck tm1
                                ty2 <- tyCheck tm2
                                lift $ case ty1 of
                                         (TyAbs ty11 ty12)
                                           | ty11 == ty2 -> Right ty12
                                           | otherwise   -> Left $ EAbs1 tm1 tm2
                                         _               -> Left $ EAbs2 tm1

tyCheckCl :: Tm -> Either Err Ty
tyCheckCl tm = runReaderT (tyCheck tm) []
