---  Parser for Cajal    ---
---  ==================  ---
---  Joey Velez-Ginorio  ---

{-

prog := decl
        prog

decl := tydecl
        tmdecl

tydecl := identifier :: ty

tmdecl := identifier params = tm

tm := \ identifier : ty . tm
     | tm tm
     | identifier
     | *
     | ( tm )

ty := 1 | ty -> ty


Example program
===============

foo :: 1 -> 1
foo x = (\x -> x)x

-}


import Cajal
import Text.Parsec
import Text.Parsec.Pos
import Text.Parsec.Error
import Data.Text (Text, pack)
import Data.Char (isSpace)
import Control.Monad.Trans.Except
import Control.Monad.Trans (MonadTrans (lift))
import Control.Monad.Trans.Reader (Reader,
                                   ReaderT,
                                   ask,
                                   local,
                                   runReader,
                                   runReaderT)

type Parser = Parsec String ()

---  ==========================  Utility  ============================  ---

ident :: Parser String
ident = (:) <$> lower <*> many alphaNum

atom :: Parser a -> Parser a
atom p = spaces *> p <* spaces

identifier :: Parser String
identifier = atom ident

symbol :: String -> Parser String
symbol s = atom (string s)

parens :: Parser a -> Parser a
parens p = symbol "(" *> p <* symbol ")"

---  ==========================  Cajal  ============================  ---

---  Parsing types  ----

tyUnit :: Parser Ty
tyUnit = symbol "1" >> pure TyUnit

tyAbsOp :: Parser (Ty -> Ty -> Ty)
tyAbsOp = symbol "->" >> pure TyAbs

tyAtom :: Parser Ty
tyAtom = tyUnit <|> parens ty

ty :: Parser Ty
ty = tyAtom `chainr1` tyAbsOp

---  Parsing terms  ---

tmUnit :: Parser Tm
tmUnit = symbol "*" >> pure TmUnit

tmVar :: Parser Tm
tmVar = TmVar . pack <$> identifier

tmAbs :: Parser Tm
tmAbs = TmAbs . pack <$>
  (symbol "lam" *> identifier <* symbol ":") <*>
  (ty <* symbol ".") <*>
  tm

tmAppOp :: Parser (Tm -> Tm -> Tm)
tmAppOp = pure TmApp

tmAtom :: Parser Tm
tmAtom = tmAbs <|> tmUnit <|> tmVar <|> parens tm

tm :: Parser Tm
tm = (tmAtom `chainl1` tmAppOp) <* symbol "."

---  Parsing declarations  ---

type TyDecl = (String, Ty)
type TmDecl = (String, [String], Tm)
type Decl = (TyDecl, TmDecl)

tyDecl :: Parser TyDecl
tyDecl = (,) <$>
         (spaces *> identifier) <*>
         (symbol "::" *> ty)

tmDecl :: Parser TmDecl
tmDecl = (,,) <$>
         (spaces *> identifier) <*>
         (many identifier) <*>
         (symbol "=" *> tm)

tytmDecl :: Parser Decl
tytmDecl = (,) <$> tyDecl <*> tmDecl

---  Parsing programs  ---

type Prog = [Decl]

comm :: Parser ()
comm = symbol "--" *> skipMany alphaNum <* char '\n'

prog :: Parser Prog
prog = many (many comm *> tytmDecl)

parseCajal' :: Parser a -> FilePath -> IO (Either ParseError a)
parseCajal' p fname = do input <- readFile fname
                         return $ runParser p () fname input

parseCajal :: FilePath -> IO (Either ParseError Tm)
parseCajal fname = do input <- readFile fname
                      return $ parseCj input


parseCj :: String -> Either ParseError Tm
parseCj s = do p <- runParser prog () "" s
               let tm = tms2tm $ fmap decl2tm p
               case tyCheckCl tm of
                 Left err ->
                   Left $ newErrorMessage (Message $ show err) (initialPos "")
                 Right tctm -> return tm

decl2tm :: Decl -> (Id, Tm, Ty)
decl2tm ((n, t),          (_, [], e))     = (pack n,e,t)
decl2tm ((n, TyAbs t1 t2),(_, (p:ps), e)) =
  (pack n,
   (TmAbs . pack) p t1 ((\(x,y,z) -> y) $ decl2tm ((n,t2), (n,ps,e))),
   TyAbs t1 t2)

tms2tm :: [(Id, Tm, Ty)] -> Tm
tms2tm ((n,p,t):[]) = p
tms2tm ((n,p,t):npts) = TmApp (TmAbs n t (tms2tm npts)) p
