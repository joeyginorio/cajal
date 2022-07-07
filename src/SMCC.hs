---  SMC: Symmetric monoidal categories  ---
---  ==================================  ---
---  Joey Velez-Ginorio                  ---

module SMCC where

import Data.Text (Text, pack)
import Control.Monad.Trans (MonadTrans (lift))
import Control.Monad.Trans.Reader (ReaderT,
                                   ask,
                                   local,
                                   runReaderT)

---  ==========================  Syntax  =============================  ---

data Ty = TyU
        | TyTens Ty Ty
        | TyArr  Ty Ty
        deriving Show


data Tm = TmID Tm        -- Identity
        | TmCOMP Tm Tm   -- Composition
        | TmTENS Tm Tm   -- Tensor
        | TmU            -- Unit
        | TmLTU Tm       -- Left unit of Tensor
        | TmRTU Tm       -- Right unit of Tensor
        | TmA Tm         -- Associativity
        | TmS Tm         -- Symmetry
        | TmEVAL Tm      -- Eval
        | TmCUR Tm       -- Curry


---  ==========================  Dynamics  ===========================  ---




---  ==========================  Statics  ============================  ---


---  Typing errors  ---

data Err = EComp         -- Domain/codomain don't align
         deriving Show

---  TcTy: Reader + Either Monad Stack  ---
--   (1) 'Reader' passes around the context.
--   (2) 'Either' passes around informative typecheck errors.

type TcTy = ReaderT Ty (Either Err) Ty

-- tyCheck :: Tm -> TcTy
-- tyCheck TmID = ask

-- tyCheck :: Tm -> TcTy
-- tyCheck TmUnit             = return TyUnit
-- tyCheck (TmVar x)          = find x
-- tyCheck (TmAbs x ty1 tm)   = do ty2 <- local ((x,ty1):) $ tyCheck tm
--                                 return $ TyAbs ty1 ty2
-- tyCheck (TmApp tm1 tm2)    = do ty1 <- tyCheck tm1
--                                 ty2 <- tyCheck tm2
--                                 lift $ case ty1 of
--                                          (TyAbs ty11 ty12)
--                                            | ty11 == ty2 -> Right ty12
--                                            | otherwise   -> Left $ EAbs1 tm1 tm2
--                                          _               -> Left $ EAbs2 tm1

