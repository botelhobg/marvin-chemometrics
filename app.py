from copy import deepcopy
import io
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from core.pls import validate_pls, component_curve
from core.exploratory import preprocess, exploratory_pca
from core.dd_simca import DDSIMCA
from core.pipeline import validate_pipeline
from core.classification import pls_da, simca_cross_validation
from core.data import (VERSION, ROLES, read_table, excel_sheets, default_config,
                       validate_config, build_dataset, demo, save_project, load_project)

st.set_page_config(page_title='Marvin · Quimiometria', page_icon='🔬', layout='wide')
st.markdown('''<style>
.block-container {max-width:1400px;padding-top:2rem;}
h1 {letter-spacing:-0.04em;}
[data-testid="stSidebar"] {background-color:#eef3f4;}
</style>''',unsafe_allow_html=True)

def install(df,cfg,name):
    st.session_state.table=df.copy(deep=True)
    st.session_state.config=deepcopy(cfg)
    st.session_state.project_name=name
    st.session_state.revision=st.session_state.get('revision',0)+1

with st.sidebar:
    st.markdown('### MARVIN\n**Quimiometria**')
    st.caption(f'v{VERSION} · Quimiometria e classificação')
    page=st.radio('Área de trabalho',['Importar','Organizar','Explorar','Pré-processamento e PCA','PLS','Classificação','Pipeline validado','Projeto'],key='page')
    st.divider()
    if st.button('Carregar exemplo sintético',use_container_width=True,key='demo'):
        df,cfg=demo()
        install(df,cfg,'Espectros sintéticos')
        st.success('Exemplo carregado. Abra Explorar.')
    if 'table' in st.session_state:
        st.caption(f"{len(st.session_state.table)} linhas · {len(st.session_state.table.columns)} colunas")
        st.caption(st.session_state.project_name)
    st.caption('Os dados ficam nesta sessão. Baixe o projeto antes de fechar ou recarregar a página.')

st.title('Seu laboratório de dados')
st.caption('Organize amostras, examine sinais e mantenha cada análise reproduzível.')

if page=='Importar':
    st.subheader('Comece pelos dados')
    st.write('Uma linha por amostra e uma coluna por variável ou metadado. Você define os papéis na etapa Organizar.')
    upload=st.file_uploader('Tabela de dados',type=['csv','tsv','txt','xlsx'],key='table_upload')
    a,b,c=st.columns(3)
    header=a.checkbox('Primeira linha contém os nomes',value=True)
    sep=b.selectbox('Separador (texto)',['Ponto e vírgula','Vírgula','Tabulação'],index=0)
    dec=c.selectbox('Decimal (texto)',['Vírgula','Ponto'],index=0)
    enc=st.selectbox('Codificação (texto)',['UTF-8','Latin-1'])
    sheet=None
    if upload and upload.name.lower().endswith('.xlsx'):
        try:
            sheet=st.selectbox('Aba do Excel',excel_sheets(upload.getvalue()))
            st.caption('Excel: valores numéricos são lidos diretamente. Fórmulas dependem do último cálculo salvo pelo Excel; recalcule e salve antes de importar.')
        except Exception as e:
            st.error(f'Não foi possível abrir a planilha: {e}')
    if st.button('Importar tabela',type='primary',disabled=upload is None):
        try:
            df,decimal=read_table(upload.getvalue(),upload.name,
                sep={'Ponto e vírgula':';','Vírgula':',','Tabulação':'\t'}[sep],
                decimal=',' if dec=='Vírgula' else '.',encoding='utf-8-sig' if enc=='UTF-8' else 'latin-1',header=header,sheet=sheet)
            install(df,default_config(df,decimal),upload.name.rsplit('.',1)[0])
            st.success('Tabela carregada. Abra Organizar para conferir identificadores, X, Y e classes.')
        except Exception as e:
            st.error(f'Importação não concluída: {e}')
    st.caption('Até 20 MB por arquivo e 2 milhões de células. Separadores de milhares não são interpretados. CSV com nomes duplicados é recusado.')
    if 'table' in st.session_state:
        st.dataframe(st.session_state.table.head(100),use_container_width=True,hide_index=True)
        st.caption('Prévia: primeiras 100 linhas. Os dados completos permanecem no projeto.')

elif 'table' not in st.session_state:
    st.info('Importe uma tabela ou carregue o exemplo sintético pela barra lateral.')
else:
    df=st.session_state.table
    cfg=st.session_state.config
    rev=st.session_state.revision
    if page=='Organizar':
        st.subheader('Dê significado a cada coluna')
        st.write('X = variáveis preditoras; Y = respostas numéricas; Classe = categoria; Grupo = lote, indivíduo ou conjunto de réplicas.')
        st.caption('A sugestão inicial marca colunas numéricas como X. Confira especialmente códigos numéricos, respostas e classes antes de aplicar.')
        with st.form(f'mapping_{rev}'):
            mapping=pd.DataFrame({'Coluna':df.columns,'Papel':[cfg['roles'][c] for c in df]})
            edited=st.data_editor(mapping,hide_index=True,use_container_width=True,disabled=['Coluna'],
                column_config={'Papel':st.column_config.SelectboxColumn(options=ROLES,required=True)},key=f'roles_{rev}')
            c1,c2,c3=st.columns(3)
            decimal=c1.selectbox('Separador decimal dos dados',['.',','],index=['.',','].index(cfg['decimal']))
            mode=c2.selectbox('Eixo X dos sinais',['Posição','Cabeçalhos numéricos'],index=['Posição','Cabeçalhos numéricos'].index(cfg['axis_mode']))
            label=c3.text_input('Nome do eixo',value=cfg['axis_label'])
            unit=st.text_input('Unidade do eixo',value=cfg['axis_unit'],placeholder='nm, cm⁻¹, min…')
            excluded=st.multiselect('Linhas a excluir (numeração a partir de 1)',list(range(len(df))),default=cfg['excluded'],format_func=lambda i:f'Linha {i+1}')
            notes=st.text_area('Notas, unidades de Y e origem dos dados',value=cfg['notes'])
            submitted=st.form_submit_button('Aplicar organização',type='primary')
        if submitted:
            candidate={'roles':dict(zip(edited.Coluna,edited.Papel)),'decimal':decimal,'axis_mode':mode,
                       'axis_label':label,'axis_unit':unit,'excluded':excluded,'notes':notes}
            try:
                validate_config(df,candidate)
                st.session_state.config=candidate
                cfg=candidate
                st.success('Organização aplicada. Dados originais preservados.')
            except ValueError as e:
                st.error(str(e))
        st.caption('Explorar e Projeto usam a última organização aplicada. Campos vazios em X/Y são preservados, sem imputação automática.')
        st.dataframe(df.head(100),use_container_width=True,hide_index=True)
    elif page=='Explorar':
        try:
            data=build_dataset(df,cfg)
        except ValueError as e:
            st.warning(f'Abra Organizar: {e}')
            st.stop()
        x=data['X']; y=data['Y']
        cols=st.columns(4)
        cols[0].metric('Amostras ativas',len(x))
        cols[1].metric('Variáveis X',x.shape[1])
        cols[2].metric('Respostas Y',y.shape[1])
        cols[3].metric('Valores ausentes em X',int(x.isna().sum().sum()))
        if x.isna().any().any() or y.isna().any().any():
            st.warning('Há valores ausentes. Gráficos mantêm lacunas; não foi feita imputação. Estatísticas usam os valores disponíveis.')
        constant=x.columns[x.nunique(dropna=True)<=1].tolist()
        if constant:
            st.warning(f'{len(constant)} variável(is) X constante(s) ou sem dados: '+', '.join(constant[:8]))
        signals,distribution,table=st.tabs(['Sinais','Distribuições','Tabela e resumo'])
        with signals:
            selected=st.multiselect('Amostras no gráfico (até 100)',data['ids'],default=data['ids'][:min(12,len(x))],max_selections=100,key=f'plot_{rev}')
            st.caption('A seleção afeta apenas o gráfico; use Organizar para excluir amostras da análise.')
            axis=np.asarray(data['axis'])
            if len(axis)>1 and not (np.all(np.diff(axis)>0) or np.all(np.diff(axis)<0)):
                st.info('O eixo não é monotônico. A ordem original foi mantida.')
            reverse=st.checkbox('Inverter direção do eixo',key=f'reverse_{rev}')
            fig=go.Figure()
            for identifier in selected:
                i=data['ids'].index(identifier)
                group=data['classes'][i] if data['classes'] else ''
                fig.add_trace(go.Scatter(x=axis,y=x.iloc[i].to_numpy(),name=f'{identifier} · {group}' if group else identifier,
                                        mode='lines',connectgaps=False))
            axisname=cfg['axis_label']+(f" ({cfg['axis_unit']})" if cfg['axis_unit'] else '')
            fig.update_layout(template='plotly_white',height=500,xaxis_title=axisname,yaxis_title='Sinal (unidade original)',legend_title='Amostra')
            if reverse: fig.update_xaxes(autorange='reversed')
            st.plotly_chart(fig,use_container_width=True)
            st.download_button('Baixar gráfico interativo',fig.to_html(include_plotlyjs=True).encode(),file_name='sinais.html',mime='text/html')
        with distribution:
            options=x.columns.tolist()+y.columns.tolist()
            var=st.selectbox('Variável ou resposta',options,key=f'distribution_{rev}')
            values=x[var] if var in x else y[var]
            plotdf=pd.DataFrame({'Valor':values.to_numpy(),'Amostra':data['ids']})
            if data['classes']: plotdf['Classe']=data['classes']
            fig2=px.histogram(plotdf,x='Valor',color='Classe' if data['classes'] else None,barmode='overlay',opacity=.7,title=var)
            st.plotly_chart(fig2,use_container_width=True)
            if data['classes']:
                st.dataframe(pd.Series(data['classes']).replace('','(vazia)').value_counts().rename_axis('Classe').reset_index(name='Amostras'),hide_index=True)
        with table:
            st.dataframe(x.set_axis(data['ids']),use_container_width=True)
            st.dataframe(x.describe().T,use_container_width=True)
            if not y.empty:
                st.write('Respostas Y')
                st.dataframe(y.set_axis(data['ids']),use_container_width=True)
            # Numeric export only, labels travel in the project for exact round-trip.
            st.download_button('Baixar matriz X (CSV)',x.to_csv(index=False).encode('utf-8-sig'),file_name='matriz_X.csv',mime='text/csv')
            st.caption('CSV de X inclui apenas variáveis, em ordem das amostras ativas. O projeto completo preserva identificadores, classes e grupos.')
    elif page=='Pré-processamento e PCA':
        st.subheader('Pré-processamento espectral e PCA exploratória')
        st.caption('Os sinais originais não são modificados. Ajuste os parâmetros, visualize e exporte a matriz processada.')
        try:
            data=build_dataset(df,cfg)
            x=data['X']
            if x.isna().any().any():
                st.error('Há valores ausentes em X. Resolva-os antes do processamento.'); st.stop()
            if x.shape[0]<3 or x.shape[1]<2:
                st.error('São necessárias pelo menos três amostras e duas variáveis X.'); st.stop()
            a,b,c=st.columns(3)
            snv=a.checkbox('SNV',key=f'snv_{rev}')
            msc=b.checkbox('MSC',key=f'msc_{rev}')
            baseline=c.checkbox('Linha de base linear (extremos)',key=f'base_{rev}')
            a,b,c,d=st.columns(4)
            smooth=a.checkbox('Savitzky–Golay',key=f'smooth_{rev}')
            deriv=b.selectbox('Derivada', [0,1,2],key=f'deriv_{rev}')
            upper=x.shape[1] if x.shape[1]%2==1 else x.shape[1]-1
            if upper>=3:
                window=c.number_input('Janela SG (ímpar)',min_value=3,max_value=upper,value=min(11,upper),step=2,key=f'window_{rev}')
                order=d.number_input('Grau do polinômio',min_value=max(1,int(deriv)),max_value=int(window)-1,value=max(2,int(deriv)),key=f'poly_{rev}')
            else:
                window=3;order=2
                if smooth or deriv:
                    st.error('São necessárias pelo menos 3 variáveis para Savitzky–Golay.'); st.stop()
            z=preprocess(x.to_numpy(),snv=snv,msc=msc,baseline=baseline,
                         smooth=smooth,derivative=deriv,window=window,polyorder=order)
            st.info('Ordem aplicada: linha de base → Savitzky–Golay/derivada → SNV → MSC. Para modelos preditivos, parâmetros dependentes das amostras devem ser ajustados apenas na calibração de cada fold.')
            fig=go.Figure()
            for i in range(min(len(z),15)):
                fig.add_trace(go.Scatter(x=data['axis'],y=z[i],mode='lines',name=str(data['ids'][i])))
            fig.update_layout(template='plotly_white',height=350,xaxis_title=cfg['axis_label'],yaxis_title='Sinal processado')
            st.plotly_chart(fig,use_container_width=True)
            st.download_button('Exportar X processado (CSV)',pd.DataFrame(z,columns=x.columns).to_csv(index=False).encode('utf-8-sig'),file_name='marvin_X_processado.csv',mime='text/csv')
            st.divider()
            st.subheader('Análise de componentes principais')
            a,b=st.columns(2)
            maxpc=min(z.shape[0]-1,z.shape[1],np.linalg.matrix_rank(z-z.mean(axis=0)))
            if maxpc<1:
                st.warning('A matriz não tem variação suficiente para PCA.');st.stop()
            pc=a.number_input('Número de PCs',min_value=1,max_value=int(maxpc),value=min(2,int(maxpc)),key=f'pca_n_{rev}')
            autoscale=b.checkbox('Autoescalar variáveis antes de PCA',key=f'pca_scale_{rev}')
            result=exploratory_pca(z,int(pc),autoscale)
            st.dataframe(pd.DataFrame({'PC':np.arange(1,int(pc)+1),'Variância explicada (%)':result['explained']*100,'Acumulada (%)':np.cumsum(result['explained'])*100}),hide_index=True)
            t=result['scores']
            plot=go.Figure(go.Scatter(x=t[:,0],y=t[:,1] if int(pc)>1 else np.zeros(len(t)),mode='markers+text',text=data['ids'],textposition='top center'))
            plot.update_layout(template='plotly_white',height=440,xaxis_title='PC1',yaxis_title='PC2' if int(pc)>1 else '(somente PC1)')
            st.plotly_chart(plot,use_container_width=True)
            l=go.Figure()
            for i in range(int(pc)):
                l.add_trace(go.Scatter(x=data['axis'],y=result['loadings'][:,i],name=f'PC{i+1}',mode='lines'))
            l.update_layout(template='plotly_white',height=330,xaxis_title=cfg['axis_label'],yaxis_title='Loading')
            st.plotly_chart(l,use_container_width=True)
            diag=pd.DataFrame({'Amostra':data['ids'],'Q (resíduos)':result['q'],'T² Hotelling':result['t2']})
            st.dataframe(diag,use_container_width=True,hide_index=True)
            st.download_button('Exportar scores e diagnósticos (CSV)',pd.concat([pd.DataFrame({'Amostra':data['ids']}),pd.DataFrame(t,columns=[f'PC{i+1}' for i in range(int(pc))]),diag.iloc[:,1:]],axis=1).to_csv(index=False).encode('utf-8-sig'),file_name='marvin_PCA_scores.csv',mime='text/csv')
            st.caption('PCA exploratória: Q e T² descritivos, sem limites estatísticos de rejeição nesta versão.')
        except (ValueError,TypeError) as e:
            st.warning(str(e))
    elif page=='Pipeline validado':
        st.subheader('Pipeline de pesquisa · Validação sem vazamento')
        st.caption('Todos os passos são ajustados novamente no conjunto de calibração de cada fold; MSC nunca usa o espectro médio da validação.')
        try:
            data=build_dataset(df,cfg); x=data['X']; y=data['Y']
            task=st.radio('Método',['PLS','PLS-DA'],horizontal=True,key='pipeline_task')
            if task=='PLS' and y.empty:
                st.warning('Configure ao menos uma resposta Y em Organizar.'); st.stop()
            if task=='PLS-DA':
                if data['classes'] is None:
                    st.warning('Configure as classes no organizador.');st.stop()
                target=np.asarray(data['classes'],dtype=str)
            else:
                target=y.to_numpy(dtype=float)
            a,b,c=st.columns(3)
            lv=a.number_input('Componentes',min_value=1,max_value=max(1,min(x.shape[1],len(x)-2)),value=2 if min(x.shape[1],len(x)-2)>=2 else 1,key='pipeline_lv')
            folds=b.number_input('Folds',min_value=2,max_value=min(10,max(2,len(x)-1)),value=min(5,len(x)-1),key='pipeline_folds')
            scale=c.checkbox('Autoescalar no PLS',value=True,key='pipeline_scale')
            st.markdown('**Pré-processamento espectral (ordem fixa)**')
            a,b,c=st.columns(3)
            baseline=a.checkbox('Linha de base linear',key='pp_baseline')
            snv=b.checkbox('SNV',key='pp_snv')
            msc=c.checkbox('MSC (referência somente da calibração)',key='pp_msc')
            a,b,c=st.columns(3)
            window=a.selectbox('Savitzky–Golay: janela',[0,5,7,9,11,15,21],key='pp_window')
            poly=b.number_input('Ordem do polinômio',1,5,2,key='pp_poly')
            der=c.selectbox('Derivada',[0,1,2],key='pp_der')
            groupmode=st.checkbox('Usar validação por grupo (replicatas juntas)',disabled=data['groups'] is None,key='pp_groupmode')
            if st.button('Executar validação do pipeline',type='primary'):
                settings=dict(baseline=baseline,window=window,polyorder=int(poly),derivative=der,snv=snv,msc=msc)
                result=validate_pipeline(x.to_numpy(),target,task=task,n_components=int(lv),folds=int(folds),scale=scale,groups=data['groups'] if groupmode else None,settings=settings)
                st.session_state['validated_pipeline_result']=result
                st.session_state['validated_pipeline_task']=task
                st.session_state['validated_pipeline_y']=target
                st.session_state['validated_pipeline_ids']=data['ids']
            if 'validated_pipeline_result' in st.session_state:
                res=st.session_state['validated_pipeline_result']; method=st.session_state['validated_pipeline_task']
                labels=st.session_state['validated_pipeline_y']; ids=st.session_state['validated_pipeline_ids']
                if method=='PLS':
                    st.dataframe(pd.DataFrame(res['metrics']),hide_index=True,use_container_width=True)
                    result_df=pd.DataFrame({'Amostra':ids})
                    for j in range(res['cv_pred'].shape[1]):
                        result_df[f'Y{j+1}_real']=labels[:,j];result_df[f'Y{j+1}_CV']=res['cv_pred'][:,j];result_df[f'Y{j+1}_cal']=res['pred'][:,j]
                else:
                    st.metric('Acurácia balanceada · CV',f"{res['balanced_accuracy']:.1%}")
                    st.dataframe(pd.DataFrame(res['confusion'],index=res['classes'],columns=res['classes']),use_container_width=True)
                    result_df=pd.DataFrame({'Amostra':ids,'Classe real':labels,'Classe CV':res['cv_pred'],'Classe calibração':res['pred']})
                st.dataframe(result_df,hide_index=True,use_container_width=True)
                st.download_button('Exportar predições CV',result_df.to_csv(index=False).encode('utf-8-sig'),'marvin_pipeline_cv.csv','text/csv')
                st.caption('CV é estimativa interna de desempenho: para publicação, reserve conjunto externo independente. Ajuste de hiperparâmetros por CV exige validação aninhada ou externa.')
        except (ValueError,TypeError,KeyError) as exc:
            st.warning(str(exc))
    elif page=='PLS':
        st.subheader('PLS1 / PLS2 · Regressão multivariada')
        st.caption('PLS1 utiliza uma resposta Y; PLS2 utiliza múltiplas respostas Y. A validação cruzada refaz o ajuste de cada fold, evitando vazamento de dados.')
        try:
            data=build_dataset(df,cfg)
            x=data['X']; y=data['Y']
            if y.empty:
                st.warning('Configure ao menos uma coluna como Y na área Organizar.')
                st.stop()
            if x.isna().any().any() or y.isna().any().any():
                st.error('Há valores ausentes em X ou Y. Trate-os antes de executar PLS.')
                st.stop()
            if x.nunique().le(1).any() or y.nunique().le(1).any():
                st.error('Existem variáveis X ou respostas Y constantes. Remova-as na organização.')
                st.stop()
            c1,c2,c3,c4=st.columns(4)
            maxlv=max(1,min(x.shape[1],len(x)-2,30))
            nlv=c1.number_input('Variáveis latentes',min_value=1,max_value=maxlv,value=min(2,maxlv),step=1,key=f'pls_lv_{rev}')
            folds=c2.number_input('Folds de CV',min_value=2,max_value=min(10,len(x)-1),value=min(5,len(x)-1),key=f'pls_folds_{rev}')
            scale=c3.selectbox('Pré-processamento', ['Centrar','Autoescalar'],index=1,key=f'pls_scale_{rev}')
            cvmode=c4.selectbox('Validação', ['K-fold aleatória','Por grupo'],disabled=data['groups'] is None,key=f'pls_cv_{rev}')
            groups=data['groups'] if cvmode=='Por grupo' else None
            result=validate_pls(x,y,int(nlv),int(folds),scale=='Autoescalar',groups)
            metric_df=pd.DataFrame(result['metrics'])
            metric_df['Y']=[str(y.columns[int(j)]) for j in metric_df['resposta']]
            st.dataframe(metric_df[['Y','R2_cal','RMSEC','R2_CV','RMSECV']],use_container_width=True,hide_index=True)
            st.caption('RMSEC e RMSECV nas unidades originais da resposta. CV por grupo mantém todas as replicatas de um grupo juntas.')
            yname=st.selectbox('Resposta a visualizar',list(y.columns),key=f'pls_y_{rev}')
            j=list(y.columns).index(yname)
            plot=go.Figure()
            plot.add_trace(go.Scatter(x=y.iloc[:,j],y=result['prediction'][:,j],mode='markers',name='Calibração',text=data['ids']))
            plot.add_trace(go.Scatter(x=y.iloc[:,j],y=result['cv_prediction'][:,j],mode='markers',name='Validação cruzada',text=data['ids']))
            limits=[float(min(y.iloc[:,j].min(),result['prediction'][:,j].min(),result['cv_prediction'][:,j].min())),float(max(y.iloc[:,j].max(),result['prediction'][:,j].max(),result['cv_prediction'][:,j].max()))]
            plot.add_trace(go.Scatter(x=limits,y=limits,mode='lines',line=dict(dash='dash',color='gray'),name='1:1'))
            plot.update_layout(template='plotly_white',height=450,xaxis_title=f'{yname} · referência',yaxis_title=f'{yname} · previsto')
            st.plotly_chart(plot,use_container_width=True)
            t1,t2=st.tabs(['VIP e scores','Predições e exportação'])
            with t1:
                vipfig=go.Figure(go.Scatter(x=data['axis'],y=result['vip'],mode='lines',name='VIP'))
                vipfig.add_hline(y=1,line_dash='dash')
                vipfig.update_layout(template='plotly_white',height=340,xaxis_title=cfg['axis_label'],yaxis_title='VIP (global PLS2)')
                st.plotly_chart(vipfig,use_container_width=True)
                score=result['scores']
                sf=go.Figure(go.Scatter(x=score[:,0],y=score[:,1] if score.shape[1]>1 else np.zeros(len(score)),mode='markers',text=data['ids']))
                sf.update_layout(template='plotly_white',height=340,xaxis_title='LV1',yaxis_title='LV2' if score.shape[1]>1 else '(uma LV)')
                st.plotly_chart(sf,use_container_width=True)
            with t2:
                export=pd.DataFrame({'Amostra':data['ids']})
                for k,col in enumerate(y.columns):
                    export[f'{col} · referência']=y.iloc[:,k].to_numpy()
                    export[f'{col} · calibração']=result['prediction'][:,k]
                    export[f'{col} · CV']=result['cv_prediction'][:,k]
                st.dataframe(export,use_container_width=True,hide_index=True)
                st.download_button('Baixar predições (CSV)',export.to_csv(index=False).encode('utf-8-sig'),file_name='PLS_predicoes.csv',mime='text/csv')
                vipcsv=pd.DataFrame({'X':x.columns,'VIP':result['vip']})
                st.download_button('Baixar VIP (CSV)',vipcsv.to_csv(index=False).encode('utf-8-sig'),file_name='PLS_VIP.csv',mime='text/csv')
            st.info('Os modelos são calculados nesta sessão e ainda não são persistidos no arquivo de projeto. A curva RMSECV e teste externo serão adicionados em outra etapa.')
        except (ValueError, TypeError) as e:
            st.warning(str(e))
    elif page=='Classificação':
        method=st.radio('Método de classificação',['PLS-DA','SIMCA','DD-SIMCA'],horizontal=True,key=f'class_method_{rev}')
        if method=='PLS-DA':
            st.subheader('PLS-DA · Classificação supervisionada')
            st.caption('Matrizes indicadoras (one-hot), modelo PLS e decisão pela maior resposta. A validação cruzada estratificada reajusta todo o modelo em cada fold.')
            try:
                data=build_dataset(df,cfg)
                x=data['X'];labels=np.asarray(data['classes'],dtype=str)
                classes,counts=np.unique(labels,return_counts=True)
                if any(v=='' for v in classes):
                    st.warning('Todas as amostras precisam ter uma classe definida.');st.stop()
                if len(classes)<2 or min(counts)<2:
                    st.warning('Defina ≥2 classes e ≥2 amostras em cada uma.');st.stop()
                a,b,c=st.columns(3)
                lv=a.number_input('Variáveis latentes',1,min(x.shape[1],len(x)-2),min(2,x.shape[1],len(x)-2),key=f'da_lv_{rev}')
                folds=b.number_input('Folds (CV estratificada)',2,int(min(counts)),min(5,int(min(counts))),key=f'da_cv_{rev}')
                scale=c.checkbox('Autoescalar X',value=True,key=f'da_scale_{rev}')
                r=pls_da(x,labels,lv,folds,scale)
                m1,m2=st.columns(2)
                m1.metric('Acurácia de CV',f'{r["accuracy"]:.1%}')
                m2.metric('Acurácia balanceada de CV',f'{r["balanced_accuracy"]:.1%}')
                st.caption('Métricas obtidas de predições fora do fold. Classes desbalanceadas: priorize acurácia balanceada, sensibilidade e especificidade por classe.')
                fig=px.imshow(r['matrix'],x=list(r['classes']),y=list(r['classes']),text_auto=True,
                              labels={'x':'Classe prevista (CV)','y':'Classe verdadeira','color':'Amostras'},aspect='auto')
                st.plotly_chart(fig,use_container_width=True)
                rows=pd.DataFrame({'Amostra':data['ids'],'Classe real':labels,'Calibração':r['pred'],'Predição CV':r['cv_pred']})
                for j,name in enumerate(r['classes']):rows[f'CV score · {name}']=r['cv_score'][:,j]
                st.dataframe(rows,use_container_width=True,hide_index=True)
                st.download_button('Baixar PLS-DA (CSV)',rows.to_csv(index=False).encode('utf-8-sig'),file_name='PLS_DA_CV.csv',mime='text/csv')
                st.caption('Scores PLS-DA não são probabilidades calibradas. CV é exploratória; validação externa permanece recomendada.')
            except (ValueError,TypeError) as exc:
                st.warning(str(exc))
        else:
            st.subheader(f'{method} · Modelagem de uma classe')
            st.caption('Calibra exclusivamente com a classe-alvo. As demais classes são usadas somente para avaliar rejeição.')
            try:
                data=build_dataset(df,cfg)
                x=data['X']
                if x.isna().any().any() or not np.isfinite(x.to_numpy(dtype=float)).all():
                    st.error('X contém valores ausentes ou infinitos. Corrija antes do ajuste.')
                    st.stop()
                classes=np.asarray(data['classes'],dtype=str)
                labels=sorted(set(classes)-{''})
                if not labels:
                    st.warning('Defina uma coluna como Classe na área Organizar.')
                    st.stop()
                c1,c2,c3,c4=st.columns(4)
                target=c1.selectbox('Classe-alvo',labels,key=f'dds_target_{rev}')
                trainmask=classes==target
                n=int(trainmask.sum())
                maximum=min(n-1,x.shape[1]-1)
                if n<8 or maximum<1:
                    st.error('Classe-alvo precisa ter ≥8 amostras e pelo menos duas variáveis X.')
                    st.stop()
                pc=c2.number_input('Componentes principais',1,maximum,value=min(2,maximum),key=f'dds_pc_{rev}')
                alpha=c3.number_input('Alpha (erro tipo I)',0.001,0.5,0.05,0.005,format='%.3f',key=f'dds_alpha_{rev}')
                gamma=c4.number_input('Gamma (outliers)',0.0001,0.5,0.01,0.001,format='%.4f',key=f'dds_gamma_{rev}')
                scaling=st.checkbox('Autoescalar X (caso contrário, apenas centrar)',value=False,key=f'dds_scale_{rev}')
                validation=simca_cross_validation(x.to_numpy(),classes,target,int(pc),float(alpha),scaling,folds=min(5,n),method=method,gamma=float(gamma))
                model=validation['model']
                result=validation['apparent']
                names=np.asarray(data['ids'])
                dfresult=pd.DataFrame({'Predição CV':np.where(validation['cv_pred'],'Aceita','Rejeitada'),'Amostra':names,'Classe declarada':classes,'Conjunto':np.where(trainmask,'Calibração (alvo)','Não-alvo / avaliação'),
                    'SD':result['SD'],'OD':result['OD'],'Distância combinada':result.get('distance',np.full(len(x),np.nan)),
                    'Aceita':result['accepted'],'Extrema':result.get('extreme',np.zeros(len(x),dtype=bool)),'Outlier':result.get('outlier',np.zeros(len(x),dtype=bool)),'Decisão':result.get('status',np.where(result['accepted'],'Aceita','Rejeitada'))})
                st.write(f'Calibração: {n} amostras da classe **{target}** · Variância PCA explicada: {model.pca_.explained_variance_ratio_.sum():.1%}')
                st.caption('Alpha: taxa nominal de rejeição de alvos. Gamma: parâmetro usado apenas pelo DD-SIMCA.')
                fig=go.Figure()
                for label,color in [('Calibração (alvo)','#246a92'),('Não-alvo / avaliação','#d45a33')]:
                    selection=dfresult['Conjunto']==label
                    if selection.any():
                        fig.add_trace(go.Scatter(x=dfresult.loc[selection,'SD'],y=dfresult.loc[selection,'OD'],mode='markers',name=label,
                            text=dfresult.loc[selection,'Amostra'],customdata=dfresult.loc[selection,'Decisão'],
                            hovertemplate='%{text}<br>SD=%{x:.3g}<br>OD=%{y:.3g}<br>%{customdata}<extra></extra>'))
                for out,label,color in ([(False,'Limite de aceitação','#238754'),(True,'Limite de outliers','#bd3b3b')] if method=='DD-SIMCA' else []):
                    bx,by=model.acceptance_boundary(outlier=out)
                    fig.add_trace(go.Scatter(x=bx,y=by,mode='lines',line=dict(color=color,dash='dash' if out else 'solid'),name=label))
                if method=='SIMCA':
                    fig.add_vline(x=model.sd_limit_,line_color='#238754')
                    fig.add_hline(y=model.od_limit_,line_color='#238754')
                fig.update_layout(template='plotly_white',height=510,xaxis_title='SD · distância de scores (quadrática)',yaxis_title='OD · distância ortogonal (quadrática)')
                st.plotly_chart(fig,use_container_width=True)
                c1,c2,c3=st.columns(3)
                c1.metric('Sensibilidade CV',f'{validation["sensitivity"]:.1%}')
                c2.metric('Especificidade CV',f'{validation["specificity"]:.1%}')
                c3.metric('Alvos aceitos no ajuste',f'{sum(result["accepted"] & trainmask)}/{n}')
                st.dataframe(dfresult,use_container_width=True,hide_index=True)
                st.download_button(f'Baixar classificação {method} (CSV)',dfresult.to_csv(index=False).encode('utf-8-sig'),file_name=f'{method.replace("-","_")}_resultados.csv',mime='text/csv')
                st.info('Sensibilidade/especificidade estimadas por CV exploratória; o conjunto de calibração ainda não é validação externa. SIMCA clássico: limites separados de SD e OD; DD-SIMCA: limite combinado.')
            except (ValueError,TypeError) as exc:
                st.warning(str(exc))
