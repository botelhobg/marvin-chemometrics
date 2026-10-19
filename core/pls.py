"""PLS regression with leakage-free cross validation and complete-case policy."""
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold, GroupKFold, cross_val_predict
from sklearn.metrics import r2_score


def prepare(X, Y):
    x=np.asarray(X,dtype=float); y=np.asarray(Y,dtype=float)
    if y.ndim==1: y=y[:,None]
    if x.ndim!=2 or y.ndim!=2 or x.shape[0]!=y.shape[0] or y.shape[1]<1:
        raise ValueError('As matrizes X e Y devem ter o mesmo número de amostras.')
    if x.shape[0]<5 or x.shape[1]<1:
        raise ValueError('São necessárias pelo menos cinco amostras e uma variável X.')
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('PLS não admite valores ausentes ou infinitos. Exclua ou trate essas linhas antes.')
    if np.any(np.std(x,axis=0)==0):
        raise ValueError('Remova variáveis X constantes antes de ajustar o modelo.')
    if np.any(np.std(y,axis=0)==0):
        raise ValueError('Remova respostas Y constantes antes de ajustar o modelo.')
    return x,y


def validate_pls(X,Y,n_components=2,n_splits=5,scale=True,groups=None):
    x,y=prepare(X,Y)
    if groups is not None:
        g=np.asarray(groups,dtype=str)
        if len(g)!=len(x) or np.any(g==''):
            raise ValueError('Todos os grupos precisam estar preenchidos para CV por grupo.')
        n_groups=len(np.unique(g))
        if n_groups<n_splits: raise ValueError('Número de grupos menor que o número de folds.')
        cv=GroupKFold(n_splits=n_splits)
        split=list(cv.split(x,y,g))
    else:
        if n_splits>len(x): raise ValueError('Número de folds excede o número de amostras.')
        split=list(KFold(n_splits=n_splits,shuffle=True,random_state=1729).split(x))
    # PLSRegression centers internally; fitting occurs separately in each CV fold.
    mintrain=min(len(tr) for tr,_ in split)
    max_components=min(x.shape[1],mintrain-1)
    if not 1<=n_components<=max_components:
        raise ValueError(f'Número de componentes inválido para esta CV (máximo {max_components}).')
    fitted=PLSRegression(n_components=n_components,scale=scale).fit(x,y)
    pred=fitted.predict(x).reshape(y.shape)
    cv_pred=cross_val_predict(PLSRegression(n_components=n_components,scale=scale),x,y,cv=split).reshape(y.shape)
    metrics=[]
    for j in range(y.shape[1]):
        metrics.append({'resposta':j,'R2_cal':float(r2_score(y[:,j],pred[:,j])),
                        'RMSEC':float(np.sqrt(np.mean((y[:,j]-pred[:,j])**2))),
                        'R2_CV':float(r2_score(y[:,j],cv_pred[:,j])),
                        'RMSECV':float(np.sqrt(np.mean((y[:,j]-cv_pred[:,j])**2)))})
    # Measured VIP: normalize total explained Y variance weighted by X score energy.
    t=fitted.x_scores_; w=fitted.x_weights_; q=fitted.y_loadings_
    ss=np.sum(t*t,axis=0)*np.sum(q*q,axis=0)
    denominator=np.sum(ss)
    weights=(w*w)/np.maximum(np.sum(w*w,axis=0),np.finfo(float).eps)
    vip=np.sqrt(x.shape[1]*(weights@ss)/denominator) if denominator>0 else np.zeros(x.shape[1])
    return {'model':fitted,'prediction':pred,'cv_prediction':cv_pred,'metrics':metrics,
            'vip':vip,'max_components':max_components,'scores':t}


def component_curve(X,Y,max_components=10,n_splits=5,scale=True,groups=None):
    x,y=prepare(X,Y)
    upper=min(max_components,x.shape[1],len(x)-2)
    records=[]
    for n in range(1,upper+1):
        try:
            result=validate_pls(x,y,n,n_splits,scale,groups)
        except ValueError:
            break
        records.append({'Componentes':n,'RMSECV médio':float(np.mean([m['RMSECV'] for m in result['metrics']]))})
    return records
