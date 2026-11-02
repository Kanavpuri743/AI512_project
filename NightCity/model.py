import math
import torch
from torch import nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

class CustomMamba(nn.Module):

    def __init__(
        self,
        d_model,
        d_state=8,
        d_conv=2,
        expand=1,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        bias=False,
        conv_bias=True,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(expand * d_model)
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank

        # --- input projection: splits into main branch (x) and gate (z) ---
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=bias)

        # --- depthwise causal conv over the sequence axis ---
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,   # causal: trim the right side after conv
            bias=conv_bias,
        )

        # --- project x -> (Delta_rank, B, C) , all input-dependent ---
        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + 2 * d_state, bias=False
        )

        # --- low-rank Delta -> full d_inner Delta ---
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # initialize dt_proj so implied timesteps span [dt_min, dt_max]
        dt_init_std = self.dt_rank ** -0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=1e-4)
        inv_softplus_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_softplus_dt)

        # --- diagonal state matrix A, stored in log-space, always negative ---
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))  # (d_inner, d_state)

        # --- skip/feedthrough term ---
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # --- output projection back to d_model ---
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)

    def forward(self, x):
        """
        x: (B, L, d_model)  -- same call convention as mamba_ssm.Mamba
        returns: (B, L, d_model)
        """
        B, L, _ = x.shape

        # split into main path (x) and gating path (z)
        xz = self.in_proj(x)                      # (B, L, 2*d_inner)
        x_in, z = xz.chunk(2, dim=-1)              # each (B, L, d_inner)

        # causal depthwise conv (needs channel-first for Conv1d)
        x_in = x_in.transpose(1, 2)                # (B, d_inner, L)
        x_in = self.conv1d(x_in)[:, :, :L]         # trim causal padding
        x_in = F.silu(x_in)
        x_in = x_in.transpose(1, 2)                # (B, L, d_inner)

        # input-dependent Delta, B, C
        dbc = self.x_proj(x_in)                    # (B, L, dt_rank + 2*d_state)
        dt, Bmat, Cmat = torch.split(
            dbc, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        delta = F.softplus(self.dt_proj(dt))        # (B, L, d_inner), > 0

        y = self._selective_scan_parallel(x_in, delta, Bmat, Cmat)

        y = y * F.silu(z)                          # gate
        return self.out_proj(y)                    # (B, L, d_model)

    def _selective_scan_parallel(self, x, delta, Bmat, Cmat):
        """
        Parallel prefix scan (Hillis-Steele doubling) for the same
        diagonal linear recurrence as `_selective_scan`, but using only
        ceil(log2(L)) sequential steps instead of L.

        Why this matters on Jetson: `_selective_scan` below launches a
        handful of tiny CUDA kernels *per timestep* in a Python loop.
        Fixed per-launch overhead dominates on edge GPUs (weak host
        dispatch, small SM count, and TensorRT unrolls the loop into L
        sequential layers rather than fusing it). This version does the
        same math with O(log L) fully-vectorized steps -- no per-step
        Python overhead, and no division (so no numerical instability),
        only mul/add/cat, all of which have stable ONNX/TensorRT support.

        x     : (B, L, d_inner)
        delta : (B, L, d_inner)
        Bmat  : (B, L, d_state)
        Cmat  : (B, L, d_state)
        returns (B, L, d_inner)
        """
        Bsz, L, d_inner = x.shape
        N = self.d_state
        A = -torch.exp(self.A_log)                  # (d_inner, N), negative

        # per-step affine map: h_t = a_t * h_{t-1} + b_t
        a = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))       # (B,L,d_inner,N)
        b = delta.unsqueeze(-1) * Bmat.unsqueeze(2) * x.unsqueeze(-1)          # (B,L,d_inner,N)

        shift = 1
        while shift < L:
            # identity-padded shifted copies via constant-value F.pad
            # (a=1, b=0) is the no-op element; pad dim is L, which sits
            # at index 1 of (B,L,d_inner,N) -> F.pad pads last dims first,
            # so permute L to the end, pad, permute back.
            # pad tuple counts from the last dim inward: (N: none, d_inner: none, L: shift on the left)
            a_shift = F.pad(a[:, :-shift], (0, 0, 0, 0, shift, 0), value=1.0)
            b_shift = F.pad(b[:, :-shift], (0, 0, 0, 0, shift, 0), value=0.0)

            # compose (a_shift,b_shift) [earlier] then (a,b) [later]:
            # new_b must use OLD a, so compute b (fused multiply-add) before overwriting a
            b = torch.addcmul(b, a, b_shift)   # b = b + a * b_shift, one fused kernel
            a = a * a_shift
            shift *= 2

        state = b                                    # (B, L, d_inner, N) == h_t for every t
        y = torch.einsum("bldn,bln->bld", state, Cmat)
        y = y + x * self.D
        return y

    def _selective_scan(self, x, delta, Bmat, Cmat):
        """
        Sequential (ONNX/TensorRT-friendly) selective scan.

        x     : (B, L, d_inner)
        delta : (B, L, d_inner)
        Bmat  : (B, L, d_state)
        Cmat  : (B, L, d_state)
        returns (B, L, d_inner)
        """
        Bsz, L, d_inner = x.shape
        N = self.d_state
        A = -torch.exp(self.A_log)                 # (d_inner, N), negative

        # discretize: Abar = exp(delta * A), Bbar_x = delta * B * x
        # shapes broadcast to (B, L, d_inner, N)
        deltaA = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))       # (B,L,d_inner,N)
        deltaB_x = (
            delta.unsqueeze(-1) * Bmat.unsqueeze(2) * x.unsqueeze(-1)
        )                                                                          # (B,L,d_inner,N)

        state = x.new_zeros(Bsz, d_inner, N)
        ys = []
        for t in range(L):
            state = deltaA[:, t] * state + deltaB_x[:, t]      # (B, d_inner, N)
            y_t = torch.einsum("bdn,bn->bd", state, Cmat[:, t])  # (B, d_inner)
            ys.append(y_t)

        y = torch.stack(ys, dim=1)                  # (B, L, d_inner)
        y = y + x * self.D                          # skip/feedthrough
        return y

class MCHLayer(nn.Module):
    def __init__(self, input_dim, output_dim,
                 d_state=8):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.norm = nn.LayerNorm(input_dim)

        # reduced Mamba (only 1/4 channels)
        self.mamba = CustomMamba(
            d_model=input_dim // 4,
            d_state=d_state,
            d_conv=2,
            expand=1
        )

        # cheap branches
        self.dw = nn.Conv2d(input_dim//4, input_dim//4, 3,
                            padding=1, groups=input_dim//4)

        self.dw_dil = nn.Conv2d(input_dim//4, input_dim//4, 3,
                                padding=2, dilation=2,
                                groups=input_dim//4)

        # channel projection (IMPORTANT)
        self.proj = nn.Linear(input_dim, output_dim)

        self.skip_scale = nn.Parameter(torch.ones(1))

    def forward(self, x):

        if x.dtype == torch.float16:
            x = x.float()

        B, C, H, W = x.shape
        assert C == self.input_dim

        N = H * W

        x_flat = x.reshape(B, C, N).transpose(1,2)
        x_norm = self.norm(x_flat)

        # split channels
        x1, x2, x3, x4 = torch.chunk(x_norm, 4, dim=2)

        # helpers
        def to_2d(t):
            return t.transpose(1,2).reshape(B, C//4, H, W)

        def to_seq(t):
            return t.reshape(B, C//4, N).transpose(1,2)

        # branches
        b1 = x1  # identity

        b2 = to_seq(self.dw(to_2d(x2)))  # local conv
        #b2 = x2
        b3 = self.mamba(x3) + self.skip_scale * x3  # global
        #b3 = x3

        b4 = to_seq(self.dw_dil(to_2d(x4)))  # larger RF
        #b4 = x4
        # fuse
        out = torch.cat([b1, b2, b3, b4], dim=2)

        out = self.norm(out)
        out = self.proj(out)

        return out.transpose(1,2).reshape(B, self.output_dim, H, W)



class Model(nn.Module):
    
    def __init__(self, num_classes=1, input_channels=3, c_list=[8,16,24,32,48,64],
                split_att='fc'):
                #, bridge=True
        super().__init__()
        self.num_classes = num_classes
        
        self.encoder1 = nn.Sequential(
            nn.Conv2d(input_channels, c_list[0], 3, stride=1, padding=1),
        )
        self.encoder2 =nn.Sequential(
            nn.Conv2d(c_list[0], c_list[1], 3, stride=1, padding=1),
        ) 
        self.encoder3 = nn.Sequential(
            nn.Conv2d(c_list[1], c_list[2], 3, stride=1, padding=1),
        )
        self.encoder4 = nn.Sequential(
            MCHLayer(input_dim=c_list[2], output_dim=c_list[3])
        )
        self.encoder5 = nn.Sequential(
            MCHLayer(input_dim=c_list[3], output_dim=c_list[4])
        )
        self.encoder6 = nn.Sequential(
            MCHLayer(input_dim=c_list[4], output_dim=c_list[5])
        )

        #if bridge: 
        #   self.scab = SC_Att_Bridge(c_list, split_att)
        #    print('SC_Att_Bridge was used')
        
        self.decoder1 = nn.Sequential(
            MCHLayer(input_dim=c_list[5], output_dim=c_list[4])
        ) 
        self.decoder2 = nn.Sequential(
            MCHLayer(input_dim=c_list[4], output_dim=c_list[3])
        ) 
        self.decoder3 = nn.Sequential(
            MCHLayer(input_dim=c_list[3], output_dim=c_list[2])
        )  
        self.decoder4 = nn.Sequential(
            nn.Conv2d(c_list[2], c_list[1], 3, stride=1, padding=1),
        )  
        self.decoder5 = nn.Sequential(
            nn.Conv2d(c_list[1], c_list[0], 3, stride=1, padding=1),
        )  
        self.ebn1 = nn.GroupNorm(4, c_list[0])
        self.ebn2 = nn.GroupNorm(4, c_list[1])
        self.ebn3 = nn.GroupNorm(4, c_list[2])
        self.ebn4 = nn.GroupNorm(4, c_list[3])
        self.ebn5 = nn.GroupNorm(4, c_list[4])
        self.dbn1 = nn.GroupNorm(4, c_list[4])
        self.dbn2 = nn.GroupNorm(4, c_list[3])
        self.dbn3 = nn.GroupNorm(4, c_list[2])
        self.dbn4 = nn.GroupNorm(4, c_list[1])
        self.dbn5 = nn.GroupNorm(4, c_list[0])

        self.final = nn.Conv2d(c_list[0], num_classes, kernel_size=1)
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        
        
   
        
    def forward(self, x):
        #prior = self.prior_moe(x)
        #hvi = self.trans.HVIT(x)

        out = F.gelu(F.max_pool2d(self.ebn1(self.encoder1(x)),2,2))
        t1 = out # b, c0, H/2, W/2

        out = F.gelu(F.max_pool2d(self.ebn2(self.encoder2(out)),2,2))
        t2 = out # b, c1, H/4, W/4 

        out = F.gelu(F.max_pool2d(self.ebn3(self.encoder3(out)),2,2))
        t3 = out # b, c2, H/8, W/8
        
        out = F.gelu(F.max_pool2d(self.ebn4(self.encoder4(out)),2,2))
        t4 = out # b, c3, H/16, W/16
        
        out = F.gelu(F.max_pool2d(self.ebn5(self.encoder5(out)),2,2))
        t5 = out # b, c4, H/32, W/32

        
        out = F.gelu(self.encoder6(out)) # b, c5, H/32, W/32
        
        out5 = F.gelu(self.dbn1(self.decoder1(out))) # b, c4, H/32, W/32
        out5 = torch.add(out5, t5) # b, c4, H/32, W/32
        
        out4 = F.gelu(F.interpolate(self.dbn2(self.decoder2(out5)),scale_factor=(2,2),mode ='bilinear',align_corners=True)) # b, c3, H/16, W/16
        out4 = torch.add(out4, t4) # b, c3, H/16, W/16
        
        out3 = F.gelu(F.interpolate(self.dbn3(self.decoder3(out4)),scale_factor=(2,2),mode ='bilinear',align_corners=True)) # b, c2, H/8, W/8
        out3 = torch.add(out3, t3) # b, c2, H/8, W/8
        
        out2 = F.gelu(F.interpolate(self.dbn4(self.decoder4(out3)),scale_factor=(2,2),mode ='bilinear',align_corners=True)) # b, c1, H/4, W/4
        out2 = torch.add(out2, t2) # b, c1, H/4, W/4 
        
        out1 = F.gelu(F.interpolate(self.dbn5(self.decoder5(out2)),scale_factor=(2,2),mode ='bilinear',align_corners=True)) # b, c0, H/2, W/2
        out1 = torch.add(out1, t1) # b, c0, H/2, W/2
        
        out0 = F.interpolate(self.final(out1),scale_factor=(2,2),mode ='bilinear',align_corners=True) # b, num_class, H, W
        
        if self.num_classes == 1:
            return torch.sigmoid(out0)
        return out0