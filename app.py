import pandas as pd
import streamlit as st
from predictor import FrozenPredictor,load_fixtures,fixture_context

st.set_page_config(page_title='DataGol 2026',page_icon='⚽',layout='wide')

NOMBRES={'Ivory Coast':'Costa de Marfil','United States':'Estados Unidos','South Africa':'Sudáfrica','South Korea':'Corea del Sur','Czech Republic':'República Checa','Netherlands':'Países Bajos','Cape Verde':'Cabo Verde','Saudi Arabia':'Arabia Saudita','DR Congo':'RD del Congo','Curaçao':'Curazao','Bosnia and Herzegovina':'Bosnia y Herzegovina'}
def es(x): return NOMBRES.get(x,x)

@st.cache_resource
def predictor(): return FrozenPredictor()
@st.cache_data
def fixtures(): return load_fixtures()

p=predictor(); fx=fixtures()
st.title('⚽ DataGol 2026')
st.subheader('Predictor probabilístico congelado y validado temporalmente')
st.caption(f"Versión {p.meta['version']} · Corte de datos: {p.meta['cutoff']} · La aplicación solo realiza inferencia; no reentrena al abrirse.")

modo=st.sidebar.radio('Modo',['Calendario del Mundial 2026','Comparación manual'])
if modo=='Calendario del Mundial 2026':
  labels=[f"{r.date.date()} · Grupo {r.group} · {es(r.team_a)} vs. {es(r.team_b)}" for _,r in fx.iterrows()]
  selected=st.sidebar.selectbox('Partido',range(len(labels)),format_func=lambda i:labels[i])
  c=fixture_context(fx,selected)
  a,b=c['team_a'],c['team_b']; host_a,host_b=c['host_a'],c['host_b']; rest_a,rest_b=c['rest_a'],c['rest_b']
  st.sidebar.caption(f"Sede: {c['row'].ground}")
  st.sidebar.caption(f"Descanso estimado: {es(a)} {rest_a} días · {es(b)} {rest_b} días")
else:
  teams=sorted(set(fx.team_a).union(fx.team_b))
  a=st.sidebar.selectbox('Selección A',teams,format_func=es)
  b=st.sidebar.selectbox('Selección B',teams,index=1,format_func=es)
  host_a=st.sidebar.checkbox(f'{es(a)} juega como anfitrión')
  host_b=st.sidebar.checkbox(f'{es(b)} juega como anfitrión')
  rest_a=st.sidebar.slider(f'Días de descanso: {es(a)}',2,14,7)
  rest_b=st.sidebar.slider(f'Días de descanso: {es(b)}',2,14,7)
if a==b: st.warning('Seleccione dos equipos diferentes.'); st.stop()

pred=p.predict(a,b,host_a,host_b,rest_a,rest_b)
ens=pred['probabilities']['ensemble']; la,lb=pred['expected_goals']
st.markdown(f'## {es(a)} vs. {es(b)}')
c1,c2,c3=st.columns(3)
c1.metric(f'Victoria de {es(a)}',f'{ens[0]*100:.1f} %')
c2.metric('Empate',f'{ens[1]*100:.1f} %')
c3.metric(f'Victoria de {es(b)}',f'{ens[2]*100:.1f} %')
c4,c5,c6=st.columns(3)
c4.metric(f'Goles esperados: {es(a)}',f'{la:.2f}')
c5.metric('Marcador modal',pred['top5_scores'][0]['score'])
c6.metric(f'Goles esperados: {es(b)}',f'{lb:.2f}')

st.markdown('### Cinco marcadores exactos más probables')
top=pd.DataFrame([{'Posición':i+1,'Marcador':f"{es(a)} {x['score']} {es(b)}",'Probabilidad':f"{x['prob']*100:.1f} %"} for i,x in enumerate(pred['top5_scores'])])
st.dataframe(top,hide_index=True,use_container_width=True)
st.caption('El marcador exacto proviene del modelo Dixon-Coles. Es el escenario modal, no una certeza.')

st.markdown('### Comparación de modelos')
models=[('Dixon-Coles completo','dixon_coles'),('Elo + logística multinomial','elo_logit'),('Gradient Boosting calibrado','gradient_boosting'),('Ensamble publicado','ensemble')]
comparison=[]
for label,key in models:
 probs=pred['probabilities'][key]
 comparison.append({'Modelo':label,f'Victoria {es(a)}':f'{probs[0]*100:.1f} %','Empate':f'{probs[1]*100:.1f} %',f'Victoria {es(b)}':f'{probs[2]*100:.1f} %'})
st.dataframe(pd.DataFrame(comparison),hide_index=True,use_container_width=True)

st.markdown('### Validación fuera de muestra')
m=p.meta['test_metrics']['ensemble']; sm=p.meta['score_metrics']
v1,v2,v3,v4=st.columns(4)
v1.metric('Exactitud 1X2',f"{m['accuracy_1x2']*100:.1f} %")
v2.metric('Log Loss',f"{m['log_loss']:.3f}")
v3.metric('Brier Score',f"{m['brier']:.3f}")
v4.metric('ECE',f"{m['ece']:.3f}")
st.caption(f"Marcador exacto: top 1 {sm['score_top1_coverage']*100:.1f} % · top 3 {sm['score_top3_coverage']*100:.1f} % · top 5 {sm['score_top5_coverage']*100:.1f} %. El resultado modal más frecuente en prueba fue {sm['most_common_modal_score']} con {sm['most_common_modal_share']*100:.1f} %: no existe colapso hacia un solo marcador.")

with st.expander('Metodología y alcance'):
 st.markdown(f"""
- **Probabilidad 1X2 publicada:** ensamble validado de Dixon-Coles ({p.meta['ensemble_weights']['dixon_coles']*100:.1f} %) y Elo-logística ({p.meta['ensemble_weights']['elo_logit']*100:.1f} %).
- **Gradient Boosting:** entrenado, calibrado y evaluado; recibió peso {p.meta['ensemble_weights']['gradient_boosting']*100:.1f} % porque no mejoró el ensamble en calibración.
- **Marcadores exactos:** Dixon-Coles completo penalizado, con decaimiento temporal.
- **Base pública conservadora:** partidos amistosos y clasificatorios con objetivo compatible con 90 minutos.
- **Corte de datos:** {p.meta['cutoff']}.

El sistema estima probabilidades. No garantiza resultados y no debe utilizarse como asesoría de apuestas.
""")
