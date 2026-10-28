import torch
import torch.nn as nn
import torch.nn.functional as F

from Vmamba.models.encoders.dual_vmamba import vssm_small as backbone
from option import get_args_parser

class Model(nn.Module):
    def __init__(self, cfg):
        super(Model, self).__init__()
        self.cfg = cfg
        self.backbone = backbone(cfg=self.cfg)
        self.channels = self.channels = [96, 192, 384, 768]

    def forward(self, rgb, modal_x):
        orisize = rgb.shape
        outs = self.backbone(rgb, modal_x)
        return outs
    
    def load_pretrain_model(self, model_path):
        pretrain_dict = torch.load(model_path)
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)

if __name__ == '__main__':
    cfg = get_args_parser()
    model = Model(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image = flow = modality = torch.ones([2, 3, 448, 448]).to(device)
    out = model(image, flow)
    print(model)
    print(out.shape)