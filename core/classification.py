"""Supervised PLS-DA and classical SIMCA, with validation that refits every fold."""
import numpy as np
from scipy.stats import chi2
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold


def check_xy(X, labels):
    x=np.asarray(X,dtype=float); y=np.asarray(labels,dtype=str)
    if x.ndim!=2 or y.ndim!=1 or len(x)!=len(y) or not np.isfinite(x).all():
        raise ValueError('X e classes incompatíveis ou com valores não finitos.')
    classes, counts=np.unique(y,return_counts=True)
    if len(classes)<2 or np.any(counts<2) or np.any(y==''):
        raise ValueError('São necessárias duas classes ou mais, sem rótulos vazios, com pelo menos duas amostras por classe.')
    return x,y,classes,counts


def _pls_predict(model,x,classes):
    values=np.asarray(model.predict(x))
    return classes[np.argmax(values,axis=1)],values


def pls_da(X, labels, components=2, folds=5, scale=True, random_state=1729):
    x,y,classes,counts=check_xy(X,labels)
    folds=int(folds); components=int(components)
    if folds<2 or folds>counts.min():
        raise ValueError(f'Folds devem ficar entre 2 e {counts.min()} (menor classe).')
    cv=StratifiedKFold(n_splits=folds,shuffle=True,random_state=random_state)
    splits=list(cv.split(x,y)); mintrain=min(len(tr) for tr,_ in splits)
    maxlv=min(x.shape[1],mintrain-1)
    if components<1 or components>maxlv:
        raise ValueError(f'Componentes devem ficar entre 1 e {maxlv}.')
    onehot=(y[:,None]==classes[None,:]).astype(float)
    model=PLSRegression(n_components=components,scale=scale).fit(x,onehot)
    fit_classes,fit_scores=_pls_predict(model,x,classes)
    cv_classes=np.empty(len(x),dtype=object); cv_scores=np.zeros((len(x),len(classes)))
    for tr,te in splits:
        fold=PLSRegression(n_components=components,scale=scale).fit(x[tr],onehot[tr])
        cv_classes[te],cv_scores[te]=_pls_predict(fold,x[te],classes)
    return dict(classes=classes,pred=fit_classes,cv_pred=cv_classes,score=fit_scores,cv_score=cv_scores,
                model=model,components=components,folds=folds,
                accuracy=float(accuracy_score(y,cv_classes)),balanced_accuracy=float(balanced_accuracy_score(y,cv_classes)),
                matrix=confusion_matrix(y,cv_classes,labels=classes),
                report=classification_report(y,cv_classes,labels=classes,output_dict=True,zero_division=0))


class SIMCA:
    """Classical SIMCA: class PCA, separate score (T²) and residual Q cutoffs.

    Limits are approximate chi-square moment-matched estimates, not DD-SIMCA.
    """
    def __init__(self,n_components=2,alpha=0.05,scale=False):
        self.n_components=int(n_components);self.alpha=float(alpha);self.scale=bool(scale)

    def fit(self,X):
        x=np.asarray(X,dtype=float)
        if x.ndim!=2 or not np.isfinite(x).all() or len(x)<8 or x.shape[1]<2:
            raise ValueError('SIMCA requer ≥8 amostras de referência, ≥2 variáveis X e dados finitos.')
        if not 0<self.alpha<1 or not 1<=self.n_components<min(x.shape):
            raise ValueError('Alpha ou componentes PCA inválidos.')
        self.mean_=x.mean(axis=0)
        self.std_=x.std(axis=0,ddof=1) if self.scale else np.ones(x.shape[1])
        if (self.std_<1e-12).any():raise ValueError('Variáveis constantes impedem autoescalamento.')
        z=(x-self.mean_)/self.std_
        self.pca_=PCA(n_components=self.n_components,svd_solver='full').fit(z)
        sd,od=self.distances(x)
        def limit(a):
            m=a.mean();v=a.var(ddof=1)
            if m<=1e-14 or v<=1e-14:raise ValueError('Distâncias degeneradas; altere número de PCs.')
            dof=2*m*m/v
            return float((m/dof)*chi2.ppf(1-self.alpha,dof))
        self.sd_limit_=limit(sd);self.od_limit_=limit(od)
        return self

    def distances(self,X):
        x=np.asarray(X,dtype=float)
        if x.ndim==1:x=x[None,:]
        if x.ndim!=2 or x.shape[1]!=len(self.mean_) or not np.isfinite(x).all():
            raise ValueError('Matriz de avaliação incompatível.')
        z=(x-self.mean_)/self.std_;t=self.pca_.transform(z)
        sd=np.sum(t*t/self.pca_.explained_variance_,axis=1)
        od=np.sum((z-self.pca_.inverse_transform(t))**2,axis=1)
        return sd,od

    def predict_detail(self,X):
        sd,od=self.distances(X)
        return dict(SD=sd,OD=od,accepted=(sd<=self.sd_limit_)&(od<=self.od_limit_))


def simca_cross_validation(X,labels,target,n_components=2,alpha=0.05,scale=False,folds=5,method='SIMCA',gamma=0.01):
    """Refit PCA separately on target-class training folds; evaluate other classes with models trained only on targets."""
    from .dd_simca import DDSIMCA
    x,y,classes,counts=check_xy(X,labels)
    mask=y==target;n=mask.sum();folds=int(folds)
    if folds<2 or folds>n or n<10:raise ValueError('Classe-alvo precisa de ≥10 amostras e folds ≤ tamanho da classe.')
    splits=list(StratifiedKFold(n_splits=folds,shuffle=True,random_state=1729).split(x,mask.astype(int))) if (~mask).sum()>=folds else None
    if splits is None:
        from sklearn.model_selection import KFold
        splits=list(KFold(n_splits=folds,shuffle=True,random_state=1729).split(x))
    predictions=np.zeros(len(x),dtype=bool)
    for tr,te in splits:
        target_train=x[tr][mask[tr]]
        model=(DDSIMCA(n_components,alpha,gamma,scale) if method=='DD-SIMCA' else SIMCA(n_components,alpha,scale)).fit(target_train)
        predictions[te]=model.predict_detail(x[te])['accepted']
    model=(DDSIMCA(n_components,alpha,gamma,scale) if method=='DD-SIMCA' else SIMCA(n_components,alpha,scale)).fit(x[mask])
    apparent=model.predict_detail(x)
    return dict(model=model,apparent=apparent,cv_pred=predictions,
        sensitivity=float(np.mean(predictions[mask])),specificity=float(np.mean(~predictions[~mask])),
        confusion=confusion_matrix(mask,predictions,labels=[False,True]))
