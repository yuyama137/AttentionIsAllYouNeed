import numpy as np
import torch
import torch.nn as nn


# MultiHeadAttentionのtransposeの理由
# a = np.arange(36).reshape(1,3,12)
# a = a.repeat(3,axis=0).reshape(3,3,12)
# a = torch.tensor(a)

# # a = torch.tensor(a)
# # t = a.transpose(-2,-1)
# print(a.reshape(3, 3, 3, 4).transpose(1,2))




# print(a.reshape(3, 3, 3, 4))

# a = torch.tensor(np.arange(10).reshape(2,5))
# b = torch.tensor(np.arange(10).reshape(2,5))

# print(a)
# print(torch.cat([a,b],1))

# l = nn.Linear(10,2)
# a = torch.tensor(np.arange(60, dtype=np.float32).reshape(3,2,10))

# print(l(a))

# a = np.arange(36).reshape(1,3,12)
# a = a.repeat(3,axis=0).reshape(3,3,12)
# a = torch.tensor(a)

# d = a.reshape(3, 3, 3, 4).transpose(1,2)
# print(d)

# print(d[:,0,:,:])
# print(d[:,1,:,:])
# print(d[:,2,:,:])

#%%
class PositionalEncoding(nn.Module):
    """
    位置エンコーディング
    
    args:
      - d_model (int) : ベクトルの次元数
      - dropout (float)
    """
    def __init__(self, d_model, dropout, device="cpu"):
        super().__init__()
        max_len = 10000 # must be over 512
        self.pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0,max_len).unsqueeze(1).type(torch.float32)
        tmp = torch.arange(0,d_model,2)
        den = 1/torch.pow(torch.ones(int(d_model/2))*10000,2*tmp/d_model)
        den = den.unsqueeze(0)
        self.pe[:,0::2] = torch.sin(torch.matmul(pos,den))
        self.pe[:,1::2] = torch.cos(torch.matmul(pos,den))
        self.pe = self.pe.to(device)

    def forward(self, x):
        return x + self.pe[:x.shape[1],:]

pe = PositionalEncoding(512,0.0)

data = torch.tensor(np.random.rand(3,10,512))
print(pe(data).shape)



# %%
