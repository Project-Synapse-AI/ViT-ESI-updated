"""CUDA model compatibility and independently derived matrix-count checks."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.cnn_1d import simple_1dCNN_v2
from models.lstm import HeckerLSTM, HeckerLSTMpl
from models.vit import EEGViT
from count_flops import a5_matrix_macs
from ablation.evaluate import build_model


class MatchedChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise RuntimeError("HARD STOP: CUDA required")

    def test_parameter_targets_and_neighbours(self):
        target = 3565538
        cnn = [sum(p.numel() for p in simple_1dCNN_v2([75,h,994],5,False).parameters()) for h in (2603,2604,2605)]
        lstm = [sum(p.numel() for p in HeckerLSTM(75,h,994,False).parameters()) for h in (294,295,296)]
        self.assertEqual(cnn[1], 3564876)
        self.assertEqual(lstm[1], 3557700)
        for counts in (cnn, lstm):
            self.assertLess(abs(counts[1]-target), min(abs(counts[0]-target),abs(counts[2]-target)))
            self.assertLess(abs(counts[1]/target-1), .05)

    def test_original_lstm_default_and_strict_weights(self):
        torch.manual_seed(0)
        original = HeckerLSTMpl(75,85,994,False).cuda().eval()
        torch.manual_seed(0)
        current = build_model("LSTM", SimpleNamespace(),75,994,500).cuda().eval()
        current.load_state_dict(original.state_dict(),strict=True)
        for key,value in current.state_dict().items():
            self.assertTrue(torch.equal(value,original.state_dict()[key]))
        x=torch.linspace(-1,1,75*17,device="cuda").reshape(1,75,17)
        with torch.no_grad():self.assertTrue(torch.equal(original(x),current(x)))
        matched=build_model("LSTM",SimpleNamespace(lstm_hidden_size=295),75,994,500).cuda()
        self.assertEqual(matched.model.lstm.hidden_size,295)
        with self.assertRaises(RuntimeError):matched.load_state_dict(original.state_dict(),strict=True)

    def test_matched_forward_backward(self):
        options=SimpleNamespace(inter_layer=2604,kernel_size=5,lstm_hidden_size=295)
        x=torch.linspace(-1,1,75*17,device="cuda").reshape(1,75,17)
        for name in ("1DCNN","LSTM"):
            model=build_model(name,options,75,994,17).cuda()
            y=model(x);self.assertEqual(y.shape,(1,994,17));y.square().mean().backward()
            for key,p in model.named_parameters():
                self.assertIsNotNone(p.grad,key);self.assertTrue(torch.isfinite(p.grad).all(),key)

    def test_independent_matrix_convention(self):
        cnn=simple_1dCNN_v2([75,2604,994],5,False)
        lstm=HeckerLSTM(75,295,994,False)
        vit=EEGViT(75,994)
        self.assertEqual(a5_matrix_macs(cnn)["matrix_macs"],3564876*500)
        # Two bidirectional LSTM layers carry32h bias scalars; all other
        # parameters are matrix entries used once per timepoint.
        self.assertEqual(a5_matrix_macs(lstm)["matrix_macs"],(3557700-32*295)*500)
        expected=500*(75*256+256*994+6*(4*256**2+2*256*512))+6*2*500**2*256
        self.assertEqual(a5_matrix_macs(vit)["matrix_macs"],expected)
        for model in (cnn,lstm,vit):
            r=a5_matrix_macs(model)
            self.assertEqual(r['matrix_flops_estimate'],2*r['matrix_macs'])
            self.assertEqual(a5_matrix_macs(model,batch=2)['matrix_macs'],2*r['matrix_macs'])
        # Doubling token count doubles affine work, quadruples attention products.
        a,b=a5_matrix_macs(vit,timepoints=250),a5_matrix_macs(vit,timepoints=500)
        self.assertEqual(b['affine_macs'],2*a['affine_macs'])
        self.assertEqual(b['attention_products_macs'],4*a['attention_products_macs'])


if __name__ == "__main__":unittest.main()
