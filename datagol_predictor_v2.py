from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import gammaln

ROOT=Path(__file__).resolve().parent
MODEL=ROOT/'models'/'dixon_coles_v2.json'
DATA=ROOT/'data'
MAX_GOALS=8

class FrozenDixonColesPredictor:
    def __init__(self):
        self.model=json.loads(MODEL.read_text(encoding='utf-8'))
        self.teams=list(self.model['teams'])
        self.idx={t:i for i,t in enumerate(self.teams)}
        self.attack=np.asarray(self.model['attack'],dtype=float)
        self.defence=np.asarray(self.model['defence'],dtype=float)
        self.intercept=float(self.model['intercept'])
        self.home=float(self.model['home_adv_log'])
        self.rho=float(self.model['rho'])
    def lambdas(self,team_a,team_b,host_a=False,host_b=False):
        ia=self.idx.get(team_a); ib=self.idx.get(team_b)
        aa=self.attack[ia] if ia is not None else 0.0
        da=self.defence[ia] if ia is not None else 0.0
        ab=self.attack[ib] if ib is not None else 0.0
        db=self.defence[ib] if ib is not None else 0.0
        la=math.exp(self.intercept+aa-db+(self.home if host_a else 0.0))
        lb=math.exp(self.intercept+ab-da+(self.home if host_b else 0.0))
        return float(np.clip(la,0.03,5.5)),float(np.clip(lb,0.03,5.5))
    def score_matrix(self,team_a,team_b,host_a=False,host_b=False):
        la,lb=self.lambdas(team_a,team_b,host_a,host_b)
        g=np.arange(MAX_GOALS+1)
        mat=np.outer(np.exp(g*np.log(la)-la-gammaln(g+1)),np.exp(g*np.log(lb)-lb-gammaln(g+1)))
        mat[0,0]*=(1-la*lb*self.rho); mat[0,1]*=(1+la*self.rho); mat[1,0]*=(1+lb*self.rho); mat[1,1]*=(1-self.rho)
        mat=np.clip(mat,1e-14,None); mat/=mat.sum()
        return mat,la,lb
    def predict(self,team_a,team_b,host_a=False,host_b=False):
        mat,la,lb=self.score_matrix(team_a,team_b,host_a,host_b)
        probs=np.array([np.tril(mat,-1).sum(),np.trace(mat),np.triu(mat,1).sum()],dtype=float)
        scores=[]
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]): scores.append({'score':f'{i}-{j}','prob':float(mat[i,j])})
        scores.sort(key=lambda x:x['prob'],reverse=True)
        return {'teams':[team_a,team_b],'host_flags':[bool(host_a),bool(host_b)],'expected_goals':[la,lb], 'probabilities':probs.tolist(),'top5_scores':scores[:5]}

def load_fixtures(): return pd.read_csv(DATA/'worldcup_2026_group_stage.csv',parse_dates=['date'])
def venue_country(ground):
    mexico={'Mexico City','Guadalajara (Zapopan)','Monterrey (Guadalupe)'}; canada={'Toronto','Vancouver'}
    return 'Mexico' if ground in mexico else ('Canada' if ground in canada else 'United States')
def fixture_context(fixtures,index):
    row=fixtures.iloc[index]; team_a=str(row.team_a); team_b=str(row.team_b); country=venue_country(str(row.ground))
    return {'row':row,'team_a':team_a,'team_b':team_b,'host_a':team_a==country,'host_b':team_b==country,'venue_country':country}
