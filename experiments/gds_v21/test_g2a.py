"""Software tests only. Synthetic arrays; no research training or GPU."""
import json
from pathlib import Path
import unittest
import numpy as np
import torch
from g2a_reader import (Reader, source_split, make_priors, metrics, target64,
                       token_probs, evaluate, signal_flags, load_saved_g1, digest)


class Tests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(Path(__file__).with_name('g2a_config.json').read_text())
        self.split = source_split([f'g1-{i:03}' for i in range(8)],self.config)
        self.data = []
        rng = np.random.default_rng(42)
        for s, split in self.split.items():
            for arm in ('N','B0','B1','B2','P','P-E'):
                self.data.append(dict(source_id=s,arm=arm,split=split,unit=s+'_'+arm,
                                      target=rng.integers(0,16,(32,32))))

    def test_source_isolation_and_priors(self):
        self.assertEqual(list(self.split.values()).count('train'),4)
        self.assertEqual(list(self.split.values()).count('dev'),2)
        self.assertEqual(list(self.split.values()).count('heldout'),2)
        self.assertEqual(self.split, source_split(list(reversed(self.split)),self.config))
        original=make_priors(self.data)
        for d in self.data:
            if d['split']!='train' or d['arm'] not in ('N','B0'):
                d['target'][:]=15
        changed=make_priors(self.data)
        for k in original: np.testing.assert_array_equal(original[k],changed[k])

    def test_probability_grid_and_metrics(self):
        y=self.data[0]['target']
        p64=np.eye(16)[target64(y)]
        p32=token_probs(p64)
        np.testing.assert_array_equal(p32,np.eye(16)[y])
        m,c=metrics(p32,y)
        self.assertEqual(m['accuracy'],1)
        self.assertEqual(m['balanced_accuracy'],1)
        self.assertEqual(m['ce'],0)
        self.assertEqual(c.sum(),1024)
        m,_=metrics(np.full((32,32,16),1/16),y)
        self.assertAlmostEqual(m['balanced_accuracy'],1/16)
        self.assertAlmostEqual(m['ce'],np.log(16))
        bad=p32.copy();bad[0,0,0]=np.nan
        with self.assertRaises(ValueError):metrics(bad,y)

    def test_shortcut_controls_and_signal(self):
        outputs={d['unit']:np.eye(16)[target64(d['target'])] for d in self.data}
        rows,confusions=evaluate(self.data,outputs,make_priors(self.data),42)
        self.assertEqual(len(rows),48*7)
        self.assertEqual(len(confusions),48*7)
        for row in rows:
            if row['method']=='shuffled_rgb':
                self.assertNotEqual(row['unit'],row['control_peer'])
                peer=next(d for d in self.data if d['unit']==row['control_peer'])
                self.assertEqual(row['split'],peer['split'])
                self.assertEqual(row['arm'],peer['arm'])
        self.assertTrue(all(f['signal'] for f in signal_flags(rows)))
        outputs={k:np.full((64,64,16),1/16) for k in outputs}
        rows,_=evaluate(self.data,outputs,make_priors(self.data),42)
        self.assertFalse(any(f['signal'] for f in signal_flags(rows)))

    def test_architecture_and_finite_gradients(self):
        torch.set_num_threads(2)
        torch.manual_seed(3)
        model=Reader()
        # Single synthetic forward/backward for software shape/autograd checking;
        # no optimizer, weights update, research data, or scientific training.
        x=torch.zeros(1,3,256,256)
        logits=model(x)
        self.assertEqual(tuple(logits.shape),(1,16,64,64))
        loss=torch.nn.functional.cross_entropy(logits,torch.zeros(1,64,64,dtype=torch.long))
        loss.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()))
        self.assertFalse(any(isinstance(m,(torch.nn.BatchNorm2d,torch.nn.Dropout)) for m in model.modules()))

    def test_saved_input_validation(self):
        import tempfile
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);attempts=[]
            rng=np.random.default_rng(43)
            for d in self.data:
                folder=root/d['unit'];folder.mkdir()
                Image.fromarray(rng.integers(0,256,(256,256,3),dtype=np.uint8)).save(folder/'image.png')
                np.savez_compressed(folder/'private_tokens.npz',timesteps=np.arange(99,-1,-1),
                                    tokens=np.zeros((100,1024),dtype=np.int32))
                attempts.append(dict(unit=d['unit'],source_id=d['source_id'],arm=d['arm'],
                                     generator_forwards=100,status='complete'))
            report=dict(stage='G1',status='complete',attempts=attempts,generation_attempts_started=48,
                        files={'geometry.py':digest(Path(__file__).with_name('geometry.py'))})
            (root/'report.json').write_text(json.dumps(report))
            data,hashes,split=load_saved_g1(root,self.config)
            self.assertEqual(len(data),48)
            self.assertEqual(len(hashes),97)
            self.assertEqual(split,self.split)
            # Invalid/missing labels stop rather than exclusion or regeneration.
            (root/attempts[0]['unit']/'private_tokens.npz').unlink()
            with self.assertRaises(ValueError):load_saved_g1(root,self.config)


if __name__=='__main__':unittest.main()
