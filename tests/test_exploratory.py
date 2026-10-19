import unittest
import numpy as np
from core.exploratory import preprocess,exploratory_pca

class Tests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(71)
        self.X=rng.normal(size=(32,41))@np.diag(np.linspace(0.3,1,41))
    def test_no_mutation(self):
        old=self.X.copy()
        out=preprocess(self.X,snv=True)
        np.testing.assert_allclose(self.X,old)
        np.testing.assert_allclose(out.mean(axis=1),0,atol=1e-12)
    def test_msc(self):
        z=preprocess(self.X,msc=True)
        self.assertEqual(z.shape,self.X.shape)
    def test_sg(self):
        z=preprocess(self.X,smooth=True,window=9,polyorder=2,derivative=1)
        self.assertTrue(np.isfinite(z).all())
    def test_pca(self):
        r=exploratory_pca(self.X,3)
        self.assertEqual(r['scores'].shape,(32,3))
        self.assertTrue((r['q']>=0).all())
        self.assertTrue((r['t2']>=0).all())
    def test_reconstruction(self):
        r=exploratory_pca(self.X,3)
        np.testing.assert_allclose(r['q'],((self.X-r['reconstructed'])**2).sum(axis=1))
    def test_invalid_window(self):
        with self.assertRaises(ValueError): preprocess(self.X,smooth=True,window=8)

if __name__=='__main__':unittest.main()
