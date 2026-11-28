"""Synthetic software checks only: no pretrained model, research inference or GPU."""
import unittest
import numpy as np
import torch
from types import SimpleNamespace
from geometry import groups,addresses,render,token_field,IDENTITY
from vq_readback import validate_loading,hard_field,read_rgb,token_diagnostics,clean_geometry


class Tests(unittest.TestCase):
    def test_strict_loading(self):
        good={k:[] for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')}
        validate_loading(good)
        for key in good:
            bad=dict(good);bad[key]=['encoder.weight']
            with self.assertRaises(RuntimeError):validate_loading(bad)
        with self.assertRaises(RuntimeError):validate_loading({})

    def test_clean_field_coordinates(self):
        tokens=np.arange(1024)
        np.testing.assert_array_equal(hard_field(tokens).reshape(4096,16),render(token_field(tokens),IDENTITY))

    def test_rank_ties_and_error_attribution(self):
        g=groups();same=np.flatnonzero(g==g[0])[1];cross=np.flatnonzero(g!=g[0])[0]
        pred=np.zeros(1024,dtype=int);truth=np.zeros(1024,dtype=int)
        truth[0]=same;truth[1]=cross
        distances=np.full((1024,4096),10.,dtype=np.float32);distances[:,0]=0.
        distances[0,same]=1.;distances[1,cross]=1.
        counts=np.zeros(4096,int);counts[0]=3
        row,diag=token_diagnostics(pred,truth,distances,counts)
        self.assertEqual(row['token_errors'],2)
        self.assertEqual(row['same_group_token_errors'],1)
        self.assertEqual(row['cross_group_token_errors'],1)
        self.assertEqual(row['cross_group_given_token_error'],.5)
        self.assertEqual(row['true_token_top4'],1)
        self.assertEqual(row['seen_positions'],1022)
        self.assertEqual(row['unseen_positions'],2)
        self.assertTrue(np.all(diag['true_token_rank'][:2]==2))
        # Tied codebook entries use the same lower-index argmin as native quantize.
        distances[0,same]=0.
        row,diag=token_diagnostics(pred,truth,distances,counts)
        self.assertEqual(diag['true_token_rank'][0],2)

    def test_blind_interface_one_encode(self):
        from diffusers.models.autoencoders.vae import VectorQuantizer
        torch.set_num_threads(2)
        class Dummy:
            def __init__(self):
                self.calls=0
                self.quantize=VectorQuantizer(4096,128,.25)
                with torch.no_grad():
                    self.quantize.embedding.weight.fill_(2.)
                    self.quantize.embedding.weight[0].zero_()
            def encode(self,rgb):
                self.calls+=1
                assert rgb.shape==(1,3,256,256)
                assert not torch.is_grad_enabled()
                return SimpleNamespace(latents=torch.zeros(1,128,32,32))
        model=Dummy()
        pred,distances=read_rgb(model,np.zeros((256,256,3),np.uint8),'cpu')
        self.assertEqual(model.calls,1)
        self.assertTrue(np.all(pred==0))
        self.assertEqual(distances.shape,(1024,4096))

    def test_geometry_uses_clean_field_without_warp(self):
        g=groups();tokens=np.array([np.flatnonzero(g==a)[0] for a in addresses()])
        field=hard_field(tokens)
        d=dict(unit='synthetic',source_id='synthetic',arm='P',split='heldout',target=g[tokens].reshape(32,32))
        rows=clean_geometry([d],{'synthetic':field},{'synthetic':field})
        self.assertEqual(len(rows),24)
        for name in ('ideal_tokens','native_vq','saved_unet'):
            values=[r['margin'] for r in rows if r['method']==name]
            self.assertTrue(min(values)>0)
        self.assertEqual([r['margin'] for r in rows[:8]],[r['margin'] for r in rows[8:16]])


if __name__=='__main__':unittest.main()
