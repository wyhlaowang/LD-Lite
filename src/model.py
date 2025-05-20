import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, in_channels=3, dim=64, n_residual=3, n_downsample=2):
        super(Encoder, self).__init__()
        self.content_encoder1 = ContentEncoder(in_channels, dim, n_residual, n_downsample)
        self.content_encoder2 = ContentEncoder(1, dim, n_residual, n_downsample)

    def forward(self, x1, x2):
        content_code1 = self.content_encoder1(x1)
        content_code2 = self.content_encoder2(x2)
        return content_code1, content_code2


class Decoder(nn.Module):
    def __init__(self, out_channels=1, dim=64, n_residual=3, n_upsample=2):
        super().__init__()

        layers_up = []
        for _ in range(n_upsample):
            layers_up += [nn.ConvTranspose2d(dim, dim//2, 5, 2, 0),
                          LayerNorm(dim//2),
                          nn.ReLU(inplace=True)]
            dim = dim // 2

        layers_up += [nn.ReflectionPad2d(3), 
                      nn.Conv2d(dim, out_channels, 7), 
                      nn.Sigmoid()]

        self.model_up = nn.Sequential(*layers_up)

    def forward(self, vi_content_code, ir_content_code):
        content_code = torch.cat([vi_content_code, ir_content_code], dim=1)
        im = self.model_up(content_code)
        return im.clamp(0,1)


class ContentEncoder(nn.Module):
    def __init__(self, in_channels=3, dim=64, n_residual=3, n_downsample=2):
        super().__init__()
        layers = [nn.ReflectionPad2d(3),
                  nn.Conv2d(in_channels, dim, 7),
                  nn.InstanceNorm2d(dim),
                  nn.ReLU(inplace=True)]

        for _ in range(n_downsample):
            layers += [nn.Conv2d(dim, dim*2, 4, stride=2, padding=1),
                       nn.InstanceNorm2d(dim*2),
                       nn.ReLU(inplace=True)]
            dim *= 2

        for _ in range(n_residual):
            layers += [ResidualBlock(dim)]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class ResidualBlock(nn.Module):
    def __init__(self, features):
        super().__init__()
        norm_layer = nn.InstanceNorm2d
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(features, features, 3),
            norm_layer(features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(features, features, 3),
            norm_layer(features)
        )

    def forward(self, x):
        return x + self.block(x)


class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        mean = x.view(x.size(0), -1).mean(1).view(*shape)
        std = x.view(x.size(0), -1).std(1).view(*shape)
        x = (x - mean) / (std + self.eps)

        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
        x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x


class Model(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, dim=16, sample=3, residual=2, load_weight=True):
        super().__init__()
        self.encoder = Encoder(in_channels=in_channels, dim=dim, n_residual=residual, n_downsample=sample)
        self.decoder = Decoder(out_channels=out_channels, dim=2*16*2**3, n_residual=residual, n_upsample=sample)

        if load_weight:
            ew = torch.load('./weight/Enc_00069.pt')
            self.encoder.load_state_dict(ew, strict=False)
            dw = torch.load('./weight/Dec_00069.pt')
            self.decoder.load_state_dict(dw, strict=False)
            print('=== Pretrained models loaded ===')
        
    def forward(self, x1, x2):
        output_image = self.decoder(*self.encoder(x1, x2))
        return output_image





