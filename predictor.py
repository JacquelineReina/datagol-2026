from __future__ import annotations
import json, math
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy.special import gammaln

ROOT=Path(__file__).resolve().parent
MODELS=ROOT/'models'
DATA=ROOT/'data'
LABELS=np.array(['H','D','A'])
FEATURES=['elo_diff','gf10_diff','ga10_diff','points10_diff','goal_diff10_diff','rest_diff','home_adv','tournament_weight','experience_diff']
MAX_GOALS=8


def temperature_scale(probs: np.ndarray,temp: float)->np.ndarray:
    logits=np.log(np.clip(probs,1e-12,1.0))/temp
    logits-=logits.max(axis=1,keepdims=True)
    ex=np.exp(logits)
    return ex/ex.sum(axis=1,keepdims=True)


def ordered_probs(model,row:pd.DataFrame)->np.ndarray:
    raw=model.predict_proba(row)[0]
    classes=list(model.named_steps['clf'].classes_) if hasattr(model,'named_steps') else list(model.classes_)
    return np.array([raw[classes.index(x)] for x in LABELS],dtype=float)


class FrozenPredictor:
    def __init__(self):
        self.meta=json.loads((MODELS/'production_metadata.json').read_text(encoding='utf-8'))
        self.dc=json.loads((MODELS/'dixon_coles.json').read_text(encoding='utf-8'))
        self.state=json.loads((MODELS/'inference_state.json').read_text(encoding='utf-8'))
        self.logit=joblib.load(MODELS/'elo_logit.joblib')
        self.gb=joblib.load(MODELS/'gradient_boosting.joblib')
        self.teams=self.dc['teams']
        self.idx={t:i for i,t in enumerate(self.teams)}
        self.attack=np.asarray(self.dc['attack'],dtype=float)
        self.defence=np.asarray(self.dc['defence'],dtype=float)

    def _team_state(self,team:str)->dict:
        avg=float(self.state['global_avg_goals_per_team'])
        return self.state['teams'].get(team,{'elo':1500.0,'n_matches':0,'last_date':None,'gf10':avg,'ga10':avg,'points10':1.0,'goal_diff10':0.0})

    def _dc_lambdas(self,a:str,b:str,host_a:bool=False,host_b:bool=False):
        ia=self.idx.get(a); ib=self.idx.get(b)
        aa=self.attack[ia] if ia is not None else 0.0; da=self.defence[ia] if ia is not None else 0.0
        ab=self.attack[ib] if ib is not None else 0.0; db=self.defence[ib] if ib is not None else 0.0
        inter=float(self.dc['intercept']); home=float(self.dc['home_adv_log'])
        la=math.exp(inter+aa-db+(home if host_a else 0.0))
        lb=math.exp(inter+ab-da+(home if host_b else 0.0))
        return float(np.clip(la,0.03,5.5)),float(np.clip(lb,0.03,5.5))

    def score_matrix(self,a:str,b:str,host_a:bool=False,host_b:bool=False):
        la,lb=self._dc_lambdas(a,b,host_a,host_b)
        rho=float(self.dc['rho']); goals=np.arange(MAX_GOALS+1)
        mat=np.outer(np.exp(goals*np.log(la)-la-gammaln(goals+1)),np.exp(goals*np.log(lb)-lb-gammaln(goals+1)))
        mat[0,0]*=(1-la*lb*rho); mat[0,1]*=(1+la*rho); mat[1,0]*=(1+lb*rho); mat[1,1]*=(1-rho)
        mat=np.clip(mat,1e-14,None); mat/=mat.sum()
        return mat,la,lb

    def _features(self,a:str,b:str,host_a:bool,host_b:bool,rest_a:int,rest_b:int)->pd.DataFrame:
        sa=self._team_state(a); sb=self._team_state(b)
        row={
          'elo_diff':(float(sa['elo'])-float(sb['elo']))/400.0,
          'gf10_diff':float(sa['gf10'])-float(sb['gf10']),
          'ga10_diff':float(sa['ga10'])-float(sb['ga10']),
          'points10_diff':float(sa['points10'])-float(sb['points10']),
          'goal_diff10_diff':float(sa['goal_diff10'])-float(sb['goal_diff10']),
          'rest_diff':float(np.clip(rest_a-rest_b,-10,10))/10.0,
          'home_adv':float(host_a)-float(host_b),
          'tournament_weight':1.0,
          'experience_diff':float(np.tanh((int(sa['n_matches'])-int(sb['n_matches']))/50.0)),
        }
        return pd.DataFrame([row],columns=FEATURES)

    def predict(self,a:str,b:str,host_a:bool=False,host_b:bool=False,rest_a:int=7,rest_b:int=7)->dict:
        mat,la,lb=self.score_matrix(a,b,host_a,host_b)
        dc=np.array([np.tril(mat,-1).sum(),np.trace(mat),np.triu(mat,1).sum()],dtype=float)
        row=self._features(a,b,host_a,host_b,rest_a,rest_b)
        logit=temperature_scale(ordered_probs(self.logit,row).reshape(1,-1),float(self.meta['logit_temperature']))[0]
        gb=temperature_scale(ordered_probs(self.gb,row).reshape(1,-1),float(self.meta['gb_temperature']))[0]
        w=self.meta['ensemble_weights']
        ens=float(w['dixon_coles'])*dc+float(w['elo_logit'])*logit+float(w['gradient_boosting'])*gb
        ens=ens/ens.sum()
        scores=[]
        for i in range(mat.shape[0]):
          for j in range(mat.shape[1]): scores.append({'score':f'{i}-{j}','prob':float(mat[i,j])})
        scores.sort(key=lambda x:x['prob'],reverse=True)
        return {'teams':[a,b],'host_flags':[host_a,host_b],'rest_days':[rest_a,rest_b],'expected_goals':[la,lb],'probabilities':{'dixon_coles':dc.tolist(),'elo_logit':logit.tolist(),'gradient_boosting':gb.tolist(),'ensemble':ens.tolist()},'top5_scores':scores[:5]}


def load_fixtures():
    df=pd.read_csv(DATA/'worldcup_2026_group_stage.csv',parse_dates=['date'])
    return df


def venue_country(ground:str)->str:
    mexico={'Mexico City','Guadalajara (Zapopan)','Monterrey (Guadalupe)'}
    canada={'Toronto','Vancouver'}
    return 'Mexico' if ground in mexico else ('Canada' if ground in canada else 'United States')


def fixture_context(fixtures:pd.DataFrame,index:int):
    r=fixtures.iloc[index]
    team_a=str(r.team_a); team_b=str(r.team_b); ground=str(r.ground); date=r.date
    country=venue_country(ground)
    host_a=(team_a==country); host_b=(team_b==country)
    previous={}
    before=fixtures[fixtures.date<date]
    for team in [team_a,team_b]:
      played=before[(before.team_a==team)|(before.team_b==team)]
      previous[team]=7 if played.empty else int((date-played.date.max()).days)
    return {'row':r,'team_a':team_a,'team_b':team_b,'host_a':host_a,'host_b':host_b,'rest_a':previous[team_a],'rest_b':previous[team_b],'venue_country':country}
