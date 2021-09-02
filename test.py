import numpy as np
import torch
from torch import nn

#mk = np.zeros((32,70,70))
#for b in range(32):
#    for r in range(40):
#        mk[b][r][r] = 1
#        for c in range(r+1, 41):
#            t = np.random.rand()
#            if t<0.1 :
#                mk[b][r][c] = 1
#                mk[b][c][r] = 1
#mk = torch.Tensor(mk).to('cuda').unsqueeze(-1)
ew = np.random.randn(32,70,70,512)
ew = torch.Tensor(ew).to('cuda')
#ew = torch.mul(ew, mk)
aw = np.random.randn(32,70,512)
aw = torch.Tensor(aw).to('cuda')
ew_0 = ew.unsqueeze(-2)
ew_1 = ew.unsqueeze(-1)
ew_s = torch.matmul(ew_0, ew_1).squeeze(-2)
aw_0 = aw.unsqueeze(2).expand_as(ew).unsqueeze(-1)
aw_0s = torch.matmul(ew_0, aw_0).squeeze(-2)
aw_1 = aw_0.permute((0,2,1,3,4))
aw_1s = torch.matmul(ew_0, aw_1).squeeze(-2)
weight = torch.cat((ew_s, aw_0s, aw_1s), dim=-1)
weight = weight/np.sqrt(512)

#path_0:
weight = nn.Softmax(dim=-1)(weight).unsqueeze(-2)
hidden = torch.cat((ew_1, aw_0, aw_1), dim=-1).permute((0,1,2,4,3))
new_e = torch.matmul(weight, hidden).squeeze(-2)
