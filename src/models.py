import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# modify first convolution layer to handle different number of input channels
def modify_first_conv_layer(model, in_channels=1):
    first_conv = model.conv1
    # create new conv layer with desired in_channels
    model.conv1 = nn.Conv2d(in_channels, out_channels=first_conv.out_channels, kernel_size=first_conv.kernel_size,
                            stride=first_conv.stride, padding=first_conv.padding, bias=first_conv.bias)

    # if using pretrained weights, compute new weights for the first layer
    if in_channels != 3 and first_conv.weight.data.size(1) == 3:
        # average the weights across the 3 channels to get weights for 1 channel
        model.conv1.weight.data = torch.mean(first_conv.weight.data, dim=1, keepdim=True)

    return model


# ========== SSL framework ==========
# mlp head with configurable number of layers
class MLPHead(nn.Module):
    def __init__(self, in_dim=2048, hidden_dim=2048, out_dim=128, num_layers=2):
        super().__init__()
        layers = []

        # first layer
        layers.extend([
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        ])

        # middle layers
        for _ in range(num_layers - 2):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True)
            ])

        # final layer (with batchnorm in simclr v2)
        layers.extend([
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(out_dim)
        ])

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        x = self.mlp(x)

        return F.normalize(x, dim=1)


# projection head with configurable number of layers for simclr
class ProjectionHead(nn.Module):
    def __init__(self, in_dim=2048, hidden_dim=2048, out_dim=128, num_layers=3):
        super().__init__()
        layers = []

        # first layer
        layers.extend([
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        ])

        # middle layers
        for _ in range(num_layers - 2):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True)
            ])

        # final layer (with batchnorm)
        layers.extend([
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(out_dim)
        ])

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        x = self.mlp(x)

        return F.normalize(x, dim=1)


# simclr model implementation
class SimCLR(nn.Module):
    def __init__(self, feature_dim=128, in_channels=1, pretrained='imagenet', encoder_name='resnet50', proj_layers=3):
        super().__init__()
        # initialize encoder based on pretrained type
        if pretrained == 'imagenet':
            print(f"loading imagenet pretrained {encoder_name} for SimCLR...")
            if encoder_name == 'resnet50':
                weights = models.ResNet50_Weights.IMAGENET1K_V1
                resnet = models.resnet50(weights=weights)
            elif encoder_name == 'resnet101':
                weights = models.ResNet101_Weights.IMAGENET1K_V1
                resnet = models.resnet101(weights=weights)
            else:
                raise ValueError(f"encoder {encoder_name} is not supported")
        else:
            print(f"initializing {encoder_name} with random weights for SimCLR...")
            if encoder_name == 'resnet50':
                resnet = models.resnet50(weights=None)
            elif encoder_name == 'resnet101':
                resnet = models.resnet101(weights=None)
            else:
                raise ValueError(f"encoder {encoder_name} is not supported")

        # modify first conv layer for input channels
        resnet = modify_first_conv_layer(resnet, in_channels)

        # encoder without classification head
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])

        # simclr v2 projection head
        self.projector = ProjectionHead(
            in_dim=2048,
            hidden_dim=2048,
            out_dim=feature_dim,
            num_layers=proj_layers
        )

    def forward(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        z = self.projector(h)

        return z


# moco model implementation
class MoCo(nn.Module):
    def __init__(self, dim=128, K=65536, m=0.999, T=0.07, in_channels=1, pretrained='imagenet', encoder_name='resnet50', proj_layers=2):
        super().__init__()
        self.K = K
        self.m = m
        self.T = T

        # initialize encoders based on pretrained type
        if pretrained == 'imagenet':
            print(f"loading imagenet pretrained {encoder_name} for MoCo...")
            if encoder_name == 'resnet50':
                weights = models.ResNet50_Weights.IMAGENET1K_V1
                resnet_q = models.resnet50(weights=weights)
                resnet_k = models.resnet50(weights=weights)
            elif encoder_name == 'resnet101':
                weights = models.ResNet101_Weights.IMAGENET1K_V1
                resnet_q = models.resnet101(weights=weights)
                resnet_k = models.resnet101(weights=weights)
            else:
                raise ValueError(f"encoder {encoder_name} is not supported")
        else:
            print(f"initializing {encoder_name} with random weights for MoCo...")
            if encoder_name == 'resnet50':
                resnet_q = models.resnet50(weights=None)
                resnet_k = models.resnet50(weights=None)
            elif encoder_name == 'resnet101':
                resnet_q = models.resnet101(weights=None)
                resnet_k = models.resnet101(weights=None)
            else:
                raise ValueError(f"encoder {encoder_name} is not supported")

        # modify first conv layer for both encoders
        resnet_q = modify_first_conv_layer(resnet_q, in_channels)
        resnet_k = modify_first_conv_layer(resnet_k, in_channels)

        # create encoder without fc layer
        self.encoder_q = nn.Sequential(*list(resnet_q.children())[:-1])
        self.encoder_k = nn.Sequential(*list(resnet_k.children())[:-1])

        # moco v2 projection head
        self.projection_q = MLPHead(
            in_dim=2048,
            hidden_dim=2048,
            out_dim=dim,
            num_layers=proj_layers
        )
        self.projection_k = MLPHead(
            in_dim=2048,
            hidden_dim=2048,
            out_dim=dim,
            num_layers=proj_layers
        )

        # initialize key encoder and projection as copy of query encoder
        for param_k, param_q in zip(self.encoder_k.parameters(), self.encoder_q.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        for param_k, param_q in zip(self.projection_k.parameters(), self.projection_q.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        # momentum update of the key encoder
        for param_k, param_q in zip(self.encoder_k.parameters(), self.encoder_q.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
        for param_k, param_q in zip(self.projection_k.parameters(), self.projection_q.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # update dictionary queue
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)

        if ptr + batch_size > self.K:
            batch_size = self.K - ptr
            keys = keys[:batch_size]

        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K

        self.queue_ptr[0] = ptr

    def forward(self, im_q, im_k):
        # compute features and contrasting logits
        q = self.encoder_q(im_q)
        q = q.view(q.size(0), -1)
        q = self.projection_q(q)

        with torch.no_grad():
            self._momentum_update_key_encoder()

            k = self.encoder_k(im_k)
            k = k.view(k.size(0), -1)
            k = self.projection_k(k)

        # compute logits
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return logits, labels


def get_simclr_loss(features, temperature=0.5):
    labels = torch.cat([torch.arange(features.shape[0] // 2) for _ in range(2)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.cuda()

    features = F.normalize(features, dim=1)

    similarity_matrix = torch.matmul(features, features.T)

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).cuda()
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    # select only the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

    logits = logits / temperature

    return F.cross_entropy(logits, labels)


def get_moco_loss(logits, labels):
    return F.cross_entropy(logits, labels)


# ========== UNet ==========
class UNet(nn.Module):
    def __init__(self, n_classes=1, in_channels=1, pretrained='imagenet', encoder_name='resnet50', ssl_weights_path=None):
        super().__init__()

        # initialize encoder based on pretrained type
        if pretrained == 'imagenet':
            print(f"loading imagenet pretrained {encoder_name}...")
            if encoder_name == 'resnet50':
                weights = models.ResNet50_Weights.IMAGENET1K_V1
                resnet = models.resnet50(weights=weights)
            elif encoder_name == 'resnet101':
                weights = models.ResNet101_Weights.IMAGENET1K_V1
                resnet = models.resnet101(weights=weights)
            else:
                raise ValueError(f"encoder {encoder_name} is not supported")

            resnet = modify_first_conv_layer(resnet, in_channels)

        elif pretrained in ['moco', 'simclr']:
            if ssl_weights_path is None:
                raise ValueError(f"ssl_weights_path must be provided for {pretrained} pretrained model")

            print(f"loading {pretrained} pretrained weights from {ssl_weights_path}...")

            # initialize base model
            if encoder_name == 'resnet50':
                resnet = models.resnet50(weights=None)
            elif encoder_name == 'resnet101':
                resnet = models.resnet101(weights=None)
            else:
                raise ValueError(f"encoder {encoder_name} is not supported")

            # modify first conv layer for input channels
            resnet = modify_first_conv_layer(resnet, in_channels)

            # remove fc layer
            base_model = nn.Sequential(*list(resnet.children())[:-1])

            # load ssl pretrained weights
            state_dict = torch.load(ssl_weights_path)
            print(f"loading {pretrained} weights...")

            if pretrained == 'moco':
                # handle moco weights
                new_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith('encoder_q.'):
                        # remove encoder_q prefix and load only backbone weights
                        k = k.replace('encoder_q.', '')
                        new_state_dict[k] = v
                state_dict = new_state_dict

            elif pretrained == 'simclr':
                # handle simclr weights
                new_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith('encoder.'):
                        # remove encoder prefix and load only backbone weights
                        k = k.replace('encoder.', '')
                        new_state_dict[k] = v
                state_dict = new_state_dict

            # load weights to base model
            msg = base_model.load_state_dict(state_dict, strict=False)
            print(f"loading ssl weights: {msg}")

            # extract layers from base model
            layers = list(base_model.children())
            resnet.conv1 = layers[0]
            resnet.bn1 = layers[1]
            resnet.relu = layers[2]
            resnet.maxpool = layers[3]
            resnet.layer1 = layers[4]
            resnet.layer2 = layers[5]
            resnet.layer3 = layers[6]
            resnet.layer4 = layers[7]

        elif pretrained is None:
            print(f"initializing {encoder_name} with random weights...")
            if encoder_name == 'resnet50':
                resnet = models.resnet50(weights=None)
            elif encoder_name == 'resnet101':
                resnet = models.resnet101(weights=None)
            else:
                raise ValueError(f"encoder {encoder_name} is not supported")

            resnet = modify_first_conv_layer(resnet, in_channels)

        else:
            raise ValueError(f"unsupported pretrained type: {pretrained}")

        # encoder layers
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        # decoder layers
        self.decoder4 = DecoderBlock(2048, 1024, 512)
        self.decoder3 = DecoderBlock(512, 512, 256)
        self.decoder2 = DecoderBlock(256, 256, 128)
        self.decoder1 = DecoderBlock(128, 64, 64)

        # final layers
        self.upconv = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.final_conv = nn.Conv2d(32, n_classes, kernel_size=1)

    def forward(self, x):
        # encoder
        x1 = self.firstconv(x)
        x1 = self.firstbn(x1)
        x1 = self.firstrelu(x1)

        x2 = self.firstmaxpool(x1)
        x2 = self.encoder1(x2)

        x3 = self.encoder2(x2)
        x4 = self.encoder3(x3)
        x5 = self.encoder4(x4)

        # decoder
        d4 = self.decoder4(x5, x4)
        d3 = self.decoder3(d4, x3)
        d2 = self.decoder2(d3, x2)
        d1 = self.decoder1(d2, x1)

        up = self.upconv(d1)
        out = self.final_conv(up)

        return out


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, x, skip):
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)

        return x


# ========== DeepLabV3Plus ==========
class DeepLabV3plus(nn.Module):
    def __init__(self, n_classes=1, in_channels=1, pretrained='imagenet', encoder_name='resnet50', ssl_weights_path=None, output_stride=16):
        super().__init__()

        # initialize encoder based on pretrained type
        if pretrained == 'imagenet':
            print(f"loading imagenet pretrained {encoder_name}...")
            if encoder_name == 'resnet50':
                weights = models.ResNet50_Weights.IMAGENET1K_V1
                self.backbone = models.resnet50(weights=weights)
            elif encoder_name == 'resnet101':
                weights = models.ResNet101_Weights.IMAGENET1K_V1
                self.backbone = models.resnet101(weights=weights)
            else:
                raise ValueError(f"encoder {encoder_name} is not supported")

            self.backbone = modify_first_conv_layer(self.backbone, in_channels)

        elif pretrained in ['moco', 'simclr']:
            if ssl_weights_path is None:
                raise ValueError(f"ssl_weights_path must be provided for {pretrained} pretrained model")

            print(f"loading {pretrained} pretrained weights from {ssl_weights_path}...")

            # initialize base model
            if encoder_name == 'resnet50':
                self.backbone = models.resnet50(weights=None)
            elif encoder_name == 'resnet101':
                self.backbone = models.resnet101(weights=None)
            else:
                raise ValueError(f"encoder {encoder_name} is not supported")

            # modify first conv layer for input channels
            self.backbone = modify_first_conv_layer(self.backbone, in_channels)

            # remove fc layer
            base_model = nn.Sequential(*list(self.backbone.children())[:-1])

            # load ssl pretrained weights
            state_dict = torch.load(ssl_weights_path)
            print(f"loading {pretrained} weights...")

            if pretrained == 'moco':
                # handle moco weights
                new_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith('encoder_q.'):
                        # remove encoder_q prefix and load only backbone weights
                        k = k.replace('encoder_q.', '')
                        new_state_dict[k] = v
                state_dict = new_state_dict

            elif pretrained == 'simclr':
                # handle simclr weights
                new_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith('encoder.'):
                        # remove encoder prefix and load only backbone weights
                        k = k.replace('encoder.', '')
                        new_state_dict[k] = v
                state_dict = new_state_dict

            # load weights to base model
            msg = base_model.load_state_dict(state_dict, strict=False)
            print(f"loading ssl weights: {msg}")

            # extract layers from base model
            layers = list(base_model.children())
            self.backbone.conv1 = layers[0]
            self.backbone.bn1 = layers[1]
            self.backbone.relu = layers[2]
            self.backbone.maxpool = layers[3]
            self.backbone.layer1 = layers[4]
            self.backbone.layer2 = layers[5]
            self.backbone.layer3 = layers[6]
            self.backbone.layer4 = layers[7]

        elif pretrained is None:
            print(f"initializing {encoder_name} with random weights...")
            if encoder_name == 'resnet50':
                self.backbone = models.resnet50(weights=None)
            elif encoder_name == 'resnet101':
                self.backbone = models.resnet101(weights=None)
            else:
                raise ValueError(f"encoder {encoder_name} is not supported")

            self.backbone = modify_first_conv_layer(self.backbone, in_channels)

        else:
            raise ValueError(f"unsupported pretrained type: {pretrained}")

        # set up dilated convolutions based on output_stride
        if output_stride == 16:
            replace_stride_with_dilation = [False, False, True]
            aspp_dilate = [6, 12, 18]
        elif output_stride == 8:
            replace_stride_with_dilation = [False, True, True]
            aspp_dilate = [12, 24, 36]
        else:
            raise ValueError(f"Output stride {output_stride} is not supported, use 8 or 16")

        # modify backbone for dilated convolutions if needed
        if encoder_name in ['resnet50', 'resnet101']:
            if replace_stride_with_dilation[0]:
                self.backbone.layer2[0].conv2.stride = (1, 1)
                self.backbone.layer2[0].downsample[0].stride = (1, 1)
                for block in self.backbone.layer2:
                    if hasattr(block, 'conv2'):
                        block.conv2.dilation = (2, 2)
                        block.conv2.padding = (2, 2)

            if replace_stride_with_dilation[1]:
                self.backbone.layer3[0].conv2.stride = (1, 1)
                self.backbone.layer3[0].downsample[0].stride = (1, 1)
                for block in self.backbone.layer3:
                    if hasattr(block, 'conv2'):
                        block.conv2.dilation = (2, 2)
                        block.conv2.padding = (2, 2)

            if replace_stride_with_dilation[2]:
                self.backbone.layer4[0].conv2.stride = (1, 1)
                self.backbone.layer4[0].downsample[0].stride = (1, 1)
                for block in self.backbone.layer4:
                    if hasattr(block, 'conv2'):
                        block.conv2.dilation = (2, 2)
                        block.conv2.padding = (2, 2)

        # get the backbone output channels
        if encoder_name in ['resnet50', 'resnet101']:
            low_level_channels = 256  # After layer1
            high_level_channels = 2048  # After layer4
        else:
            raise NotImplementedError("Only ResNet models are supported")

        # ASPP Module
        self.aspp = ASPP(high_level_channels, 256, aspp_dilate)

        # low level feature processing
        self.low_level_conv = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )

        # decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, n_classes, 1)
        )

    def forward(self, x):
        input_size = x.size()[2:]

        # extract features from backbone
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        low_level_feat = self.backbone.layer1(x)
        x = self.backbone.layer2(low_level_feat)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        # ASPP module
        x = self.aspp(x)

        # process low level features
        low_level_feat = self.low_level_conv(low_level_feat)

        # upsample ASPP features
        x = F.interpolate(x, size=low_level_feat.shape[2:], mode='bilinear', align_corners=False)

        # concatenate low level and ASPP features
        x = torch.cat([x, low_level_feat], dim=1)

        # decoder
        x = self.decoder(x)

        # final upsampling to original input size
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=False)

        return x


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)

        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rates):
        super(ASPP, self).__init__()
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ))

        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)

        return self.project(res)


# ========== get model function ==========
def get_model(model_name, encoder_name, pretrained=True, n_classes=1, in_channels=1, ssl_weights_path=None):
    if isinstance(pretrained, bool):
        pretrained = 'imagenet' if pretrained else None

    if model_name == 'UNet':
        model = UNet(n_classes=n_classes, in_channels=in_channels,
                     pretrained=pretrained, encoder_name=encoder_name, ssl_weights_path=ssl_weights_path)

        return model
    elif model_name == 'DeepLabV3plus':
        model = DeepLabV3plus(n_classes=n_classes, in_channels=in_channels,
                              pretrained=pretrained, encoder_name=encoder_name, ssl_weights_path=ssl_weights_path)

        return model
    else:
        raise ValueError(f"Model {model_name} is not supported")


# test code
if __name__ == "__main__":
    configs = [
        ('UNet', 'resnet50', True),
        ('UNet', 'resnet50', False),
        ('UNet', 'resnet101', True),
        ('UNet', 'resnet101', False),
        ('DeepLabV3plus', 'resnet50', True),
        ('DeepLabV3plus', 'resnet50', False),
        ('DeepLabV3plus', 'resnet101', True),
        ('DeepLabV3plus', 'resnet101', False)
    ]

    x = torch.randn(1, 1, 224, 224)

    for model_name, encoder_name, pretrained in configs:
        print(f"\nTesting {model_name} with {encoder_name} (pretrained={pretrained})")
        model = get_model(model_name, encoder_name, pretrained=pretrained)
        model.eval()

        with torch.no_grad():
            output = model(x)

        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

        # verify output size matches input size
        assert output.shape[2:] == x.shape[2:], "Output size doesn't match input size!"