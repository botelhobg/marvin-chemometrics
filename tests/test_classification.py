import numpy as np
import pytest
from core.classification import pls_da, simca_cross_validation

def dataset():
    rng=np.random.default_rng(22)
    return np.vstack((rng.normal(0,1,(45,12)),rng.normal(1.8,1,(45,12)))),np.array(['a']*45+['b']*45)

def test_plsda_cv():
    x,y=dataset();out=pls_da(x,y,2,5)
    assert out['matrix'].shape==(2,2)
    assert len(out['cv_pred'])==len(y)
    assert out['accuracy']>.95

def test_simca_cv():
    x,y=dataset()
    for method in ('SIMCA','DD-SIMCA'):
        out=simca_cross_validation(x,y,'a',2,method=method)
        assert out['sensitivity']>.6
        assert out['specificity']>.9
        assert len(out['cv_pred'])==len(y)

def test_empty_class_rejected():
    x,y=dataset();y[0]=''
    with pytest.raises(ValueError):pls_da(x,y,2,5)
