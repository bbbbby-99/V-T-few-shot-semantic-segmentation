import torch
import torch.nn as nn
import torch.nn.functional as F


class FMM(nn.Module):
    def __init__(self, in_features, out_features):
        super(FMM, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.conv_rgb = nn.Conv2d(in_channels= in_features, out_channels= in_features, kernel_size=1)
        self.conv_th = nn.Conv2d(in_channels=in_features, out_channels= in_features, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)
        self.value_conv_rgb = nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=1)
        self.value_conv_th = nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=1)

        self.gamma_rgb = nn.Parameter(torch.zeros(1))
        self.gamma_th = nn.Parameter(torch.zeros(1))
        self.gamma_2_rgb = nn.Parameter(torch.zeros(1))
        self.gamma_2_th = nn.Parameter(torch.zeros(1))
    def forward(self,x,x_th):
        '''x:b,256,w,h
        x_th:b,256,w,h'''
        '''position'''
        b,c,w,h=x.size()
        x_k = self.conv_rgb(x)
        x_th_k = self.conv_th(x_th)
        x_v =self.value_conv_rgb(x)
        x_v_th = self.value_conv_th(x_th)

        x_v = x_v.view(b, -1, w * h)
        x_v_th = x_v_th.view(b, -1, w * h)
        #
        x_k  = x_k.view(b, -1, w * h)
        x_th_k = x_th_k.view(b, -1, w * h)#b,c1 wh
        x_th_k_t = x_th_k.permute(0,2,1)#b,wh,c1

        sim = torch.bmm(x_th_k_t,x_k)#b,wh,wh
        sim = self.softmax(sim)
        x_1 = torch.bmm(x_v,sim)
        x_th_1 = torch.bmm(x_v_th,sim)#b,c,wh

        x_1 =x_1.view(b,c,w,h)
        x_th_1 = x_th_1.view(b, c, w, h)

        x_1 = x_1 * self.gamma_rgb + x
        x_th_1 = x_th_1 * self.gamma_th + x_th
        '''C'''
        x_1_1  = x_v.view(b, -1, w * h)
        x_th_1_1 = x_v_th.view(b, -1, w * h)#b,c wh
        x_1_t = x_1_1.permute(0,2,1)
        energy = torch.bmm(x_th_1_1, x_1_t)#b,c,c
        sim_c = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        sim_c = self.softmax(sim_c)

        x_2 = torch.bmm(sim_c,x_1_1)
        x_th_2 = torch.bmm(sim_c,x_th_1_1)  # b,c,wh

        x_2 =x_2.view(b,c,w,h)
        x_th_2 = x_th_2.view(b, c, w, h)

        x_2 = self.gamma_2_rgb * x_2 + x_1
        x_th_2 = self.gamma_2_th * x_th_2 + x_th_1


        return  x_2,x_th_2