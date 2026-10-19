import numpy as np
import pytest
from core.pipeline import SpectralTransformer, validate_pipeline

def make():
    rng=np.random.default_rng(41)
    x=rng.normal(0,0.5,(36,21))+np.sin(np.arange(21)/3)
    y=x[:,4]+x[:,8]*2
    return x,y

def test_msc_fit_reference_unaffected_by_test_data():
    x,_=make()
    tr=SpectralTransformer(msc=True).fit(x[:20]); old=tr.reference_.copy()
    tr.transform(x[20:]*12)
    np.testing.assert_array_equal(old,tr.reference_)

def test_group_folds():
    x,y=make(); g=np.repeat(np.arange(12),3)
    result=validate_pipeline(x,y,groups=g,folds=4,settings={'msc':True})
    assert result['cv_pred'].shape==(36,1)
    assert all(f['grupos_cal']==9 for f in result['folds'])

def test_plsda():
    x,y=make(); result=validate_pipeline(x,np.where(y>np.median(y),'alto','baixo'),task='PLS-DA',folds=3)
    assert result['confusion'].sum()==len(x)

def test_derivative_with_invalid_window():
    x,_=make()
    with pytest.raises(ValueError): SpectralTransformer(window=4,derivative=1).fit(x)
