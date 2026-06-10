import json
from pathlib import Path
import pandas as pd
import streamlit as st
from predictor import FrozenDixonColesPredictor, load_fixtures, fixture_context
ROOT=Path(__file__).resolve().parent
META=json.loads((ROOT/'models'/'production_metadata_v2.json').read_text(encoding='utf-8'))
st.set_page_config(page_title='DataGol 2026 v2.0',page_icon='⚽',layout='wide')
NOMBRES={'Ivory Coast':'Costa de Marfil','United States':'Estados Unidos','South Africa':'Sudáfrica','South Korea':'Corea del Sur','Czech Republic':'República Checa','Netherlands':'Países Bajos','Cape Verde':'Cabo Verde','Saudi Arabia':'Arabia Saudita','DR Congo':'RD del Congo','Curaçao':'Curazao','Bosnia and Herzegovina':'Bosnia y Herzegovina','Mexico':'México'}
def es(x): return NOMBRES.get(x,x)
@st.cache_resource
def predictor(): return FrozenDixonColesPredictor()
@st.cache_data
def fixtures(): return load_fixtures()
p=predictor(); fx=fixtures()
st.title('⚽ DataGol 2026 v2.0')
st.subheader('Predictor probabilístico congelado y validado retrospectivamente en mundiales')
st.caption(f"Modelo publicado: Dixon-Coles completo penalizado · Corte de datos: {META['cutoff']} · Estado: {META['publication_status']}")
modo=st.sidebar.radio('Modo',['Calendario Mundial 2026','Comparación manual'])
if modo=='Calendario Mundial 2026':
 labels=[f"{r.date.date()} · Grupo {r.group} · {es(r.team_a)} vs. {es(r.team_b)}" for _,r in fx.iterrows()]
 selected=st.sidebar.selectbox('Partido',range(len(labels)),format_func=lambda i:labels[i]); ctx=fixture_context(fx,selected)
 team_a,team_b,host_a,host_b=ctx['team_a'],ctx['team_b'],ctx['host_a'],ctx['host_b']; st.sidebar.caption(f"Sede: {ctx['row'].ground}")
else:
 teams=sorted(set(fx.team_a).union(fx.team_b)); team_a=st.sidebar.selectbox('Selección A',teams,format_func=es); team_b=st.sidebar.selectbox('Selección B',teams,index=1,format_func=es); host_a=st.sidebar.checkbox(f'{es(team_a)} juega como anfitrión'); host_b=st.sidebar.checkbox(f'{es(team_b)} juega como anfitrión')
if team_a==team_b: st.warning('Seleccione dos equipos diferentes.'); st.stop()
pr=p.predict(team_a,team_b,host_a,host_b); probs=pr['probabilities']; la,lb=pr['expected_goals']
st.markdown(f'## {es(team_a)} vs. {es(team_b)}')
c1,c2,c3=st.columns(3); c1.metric(f'Victoria de {es(team_a)}',f'{probs[0]*100:.1f} %'); c2.metric('Empate',f'{probs[1]*100:.1f} %'); c3.metric(f'Victoria de {es(team_b)}',f'{probs[2]*100:.1f} %')
c4,c5,c6=st.columns(3); c4.metric(f'Goles esperados: {es(team_a)}',f'{la:.2f}'); c5.metric('Marcador modal',pr['top5_scores'][0]['score']); c6.metric(f'Goles esperados: {es(team_b)}',f'{lb:.2f}')
st.markdown('### Cinco marcadores exactos más probables')
st.dataframe(pd.DataFrame([{'Posición':i+1,'Marcador':f"{es(team_a)} {x['score']} {es(team_b)}",'Probabilidad':f"{x['prob']*100:.1f} %"} for i,x in enumerate(pr['top5_scores'])]),hide_index=True,use_container_width=True)
st.caption('El marcador modal es el escenario exacto individual con mayor probabilidad. No es una certeza.')
st.markdown('### Validación específica en mundiales anteriores')
wc=META['worldcup_retrospective_validation']['aggregate_metrics']['dixon_coles_full']; a,b,c,d=st.columns(4); a.metric('Log Loss mundialista',f"{wc['log_loss']:.3f}"); b.metric('Brier Score mundialista',f"{wc['brier']:.3f}"); c.metric('Exactitud 1X2 mundialista',f"{wc['accuracy_1x2']*100:.1f} %"); d.metric('ECE mundialista',f"{wc['ece']:.3f}")
with st.expander('Criterio científico de publicación'):
 st.markdown('''- Se depuraron los marcadores a 90 minutos de Brasil 2014, Rusia 2018 y Catar 2022.\n- La validación retrospectiva se realizó sin utilizar resultados futuros para predecir el pasado.\n- El ensamble experimental no mostró una mejora mundialista estadísticamente confirmada frente a la familia Poisson/Dixon-Coles.\n- Por parsimonia, coherencia entre probabilidades y marcadores, y trazabilidad, la versión pública utiliza Dixon-Coles completo penalizado.\n- La aplicación solo realiza inferencia con artefactos congelados; no reentrena al abrirse.''')
st.divider(); st.caption('Herramienta académica de predicción probabilística a 90 minutos. No garantiza resultados y no constituye asesoría de apuestas.')
