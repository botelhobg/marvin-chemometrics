"""Leakage-safe spectral preprocessing, regression and classification validation."""
import numpy as np
from scipy.signal import savgol_filter
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold, GroupKFold, StratifiedKFold
from sklearn.metrics import r2_score, balanced_accuracy_score, confusion_matrix

class SpectralTransformer:
    def __init__(self, baseline=False, window=0, polyorder=2, derivative=0, snv=False, msc=False):
        self.baseline=baseline; self.window=int(window); self.polyorder=int(polyorder)
        self.derivative=int(derivative); self.snv=snv; self.msc=msc

    def _row_ops(self, X):
        z=np.asarray(X,float).copy()
        if z.ndim!=2 or z.shape[1]<2 or not np.isfinite(z).all():
            raise ValueError('X deve conter espectros finitos, com pelo menos duas variáveis.')
        if self.baseline:
            z-=np.linspace(0,1,z.shape[1])[None,:]*(z[:,-1]-z[:,0])[:,None]+z[:,0,None]
        if self.window or self.derivative:
            w=self.window
            if w<3 or w%2!=1 or w>z.shape[1] or not 0<=self.derivative<=self.polyorder<w:
                raise ValueError('Janela/ordem Savitzky-Golay incompatíveis.')
            z=savgol_filter(z,w,polyorder=self.polyorder,deriv=self.derivative,axis=1)
        if self.snv:
            std=z.std(axis=1,ddof=1)
            if (std<1e-12).any(): raise ValueError('SNV não admite espectros constantes.')
            z=(z-z.mean(axis=1,keepdims=True))/std[:,None]
        return z

    def fit(self,X):
        z=self._row_ops(X)
        if self.msc:
            self.reference_=z.mean(axis=0).copy()
            if np.var(self.reference_)<1e-20: raise ValueError('Referência MSC degenerada.')
        self.n_features_=z.shape[1]
        return self

    def transform(self,X):
        z=self._row_ops(X)
        if z.shape[1]!=self.n_features_: raise ValueError('Número de variáveis mudou.')
        if self.msc:
            ref=self.reference_; refc=ref-ref.mean(); slope=((z-z.mean(axis=1,keepdims=True))@refc)/(refc@refc)
            if (np.abs(slope)<1e-12).any(): raise ValueError('Inclinação MSC nula.')
            intercept=z.mean(axis=1)-slope*ref.mean()
            z=(z-intercept[:,None])/slope[:,None]
        return z

    def fit_transform(self,X): return self.fit(X).transform(X)

def validate_pipeline(X,y,*,task='PLS',n_components=2,folds=5,groups=None,scale=True,settings=None,random_state=1729):
    X=np.asarray(X,float); y=np.asarray(y)
    if X.ndim!=2 or len(X)<6 or not np.isfinite(X).all() or len(y)!=len(X):
        raise ValueError('Dados incompatíveis ou não finitos.')
    if folds<2: raise ValueError('São necessários pelo menos dois folds.')
    if groups is not None:
        g=np.asarray(groups)
        if len(g)!=len(X) or len(np.unique(g))<folds: raise ValueError('Grupos insuficientes para CV.')
        splits=list(GroupKFold(n_splits=folds).split(X,y,g))
    elif task=='PLS-DA':
        _,cnt=np.unique(y,return_counts=True)
        if cnt.min()<folds: raise ValueError('A menor classe possui menos amostras que folds.')
        splits=list(StratifiedKFold(folds,shuffle=True,random_state=random_state).split(X,y))
    else:
        splits=list(KFold(folds,shuffle=True,random_state=random_state).split(X))
    maxlv=min(X.shape[1],min(len(tr) for tr,_ in splits)-1)
    if not 1<=n_components<=maxlv: raise ValueError(f'Componentes válidos: 1 a {maxlv}.')
    settings=settings or {}
    classes=np.unique(y) if task=='PLS-DA' else None
    targets=(y[:,None]==classes).astype(float) if classes is not None else np.asarray(y,float).reshape(len(X),-1)
    oof=np.empty((len(X),targets.shape[1]),float)
    fold_info=[]
    for i,(tr,te) in enumerate(splits,1):
        processor=SpectralTransformer(**settings).fit(X[tr]); xa=processor.transform(X[tr]); xb=processor.transform(X[te])
        if np.any(xa.std(axis=0)<1e-12): raise ValueError('Alguma variável ficou constante após o pré-processamento.')
        model=PLSRegression(n_components=n_components,scale=scale).fit(xa,targets[tr]); oof[te]=model.predict(xb)
        fold_info.append({'fold':i,'calibração':len(tr),'validação':len(te),'grupos_cal':len(set(groups[j] for j in tr)) if groups is not None else None})
    processor=SpectralTransformer(**settings).fit(X); xp=processor.transform(X)
    model=PLSRegression(n_components=n_components,scale=scale).fit(xp,targets)
    fit=model.predict(xp)
    if classes is not None:
        cv_label=classes[np.argmax(oof,axis=1)]; fit_label=classes[np.argmax(fit,axis=1)]
        return dict(cv_pred=cv_label,pred=fit_label,cv_scores=oof,classes=classes,confusion=confusion_matrix(y,cv_label,labels=classes),balanced_accuracy=float(balanced_accuracy_score(y,cv_label)),model=model,processor=processor,folds=fold_info)
    metrics=[{'Y':j+1,'R2_CV':float(r2_score(targets[:,j],oof[:,j])),'RMSECV':float(np.sqrt(np.mean((targets[:,j]-oof[:,j])**2))),'R2_cal':float(r2_score(targets[:,j],fit[:,j])),'RMSEC':float(np.sqrt(np.mean((targets[:,j]-fit[:,j])**2)))} for j in range(targets.shape[1])]
    return dict(cv_pred=oof,pred=fit,metrics=metrics,model=model,processor=processor,folds=fold_info)
