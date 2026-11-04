import unittest
import torch
from refrec.recal import multigroup_kl_shifts


class MultiGroupProjectionTests(unittest.TestCase):
    def test_exact_mass_and_within_group_order(self):
        logits=torch.tensor([[2.,1.,0.,-1.]])
        groups=torch.tensor([0,0,1,2])
        p=logits.softmax(1); m=torch.stack([p[:,groups==k].sum(1) for k in range(3)],1)
        t=torch.tensor([[.2,.3,.5]])
        shift=multigroup_kl_shifts(t,m)
        adjusted=logits+shift[:,groups]
        q=adjusted.softmax(1)
        got=torch.stack([q[:,groups==k].sum(1) for k in range(3)],1)
        self.assertTrue(torch.allclose(got,t,atol=1e-6))
        self.assertGreater(float(adjusted[0,0]),float(adjusted[0,1]))

    def test_identity(self):
        m=torch.tensor([[.2,.3,.5]])
        self.assertTrue(torch.allclose(multigroup_kl_shifts(m,m),torch.zeros_like(m)))

    def test_empty_group_rejected(self):
        with self.assertRaises(ValueError):
            multigroup_kl_shifts(torch.tensor([[.2,.3,.5]]),torch.tensor([[.0,.4,.6]]))


if __name__=='__main__': unittest.main()
