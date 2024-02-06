from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .adapters import BackboneAdapter, PreFusionAdapter, PostFusionAdapter

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.squeeze(0)


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.relu3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(2)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


# class that defines each single block of the transformer
# we added the adapters
class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, n_layer: int = 0, layers: int = 0, type_input: str = None):
        super().__init__()

        self.n_layer = n_layer
        self.type_input = type_input

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor, back_adap_MHSA, back_adap_MLP):
        # in the forward pass we provide the adapters to use in the current layer
        # an adapter is applied after the MHSA and one after the MLP
        x = x + self.attention(self.ln_1(x))
        x = x + back_adap_MHSA(x)
        x = x + self.mlp(self.ln_2(x))
        x = x + back_adap_MLP(x)
        return x


# transformer module
# we modified the ResidualAttentionBlock to add the adapters
class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, type_input: str = None):
        super().__init__()
        self.type_input = type_input
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask, n_layer=n_layer, layers=layers, type_input=type_input) for n_layer in range(layers)])

    def forward(self, x: torch.Tensor):
        # print(f'type input: {self.type_input}\tinput shape: {x.shape}')
        return self.resblocks(x)


# visual encoder of CLIP
# we modified the self.transformer to add the adapters to the layers
class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads, type_input='image')

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x


class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int
                 ):
        super().__init__()

        # various model dimensions
        self.vision_layers = vision_layers
        self.transformer_layers = transformer_layers
        self.vision_width = vision_width
        self.transformer_width = transformer_width
        self.context_length = context_length

        # output of first 4 layers of visual encoder
        # used later as input for the refiner
        self.fv1 = None
        self.fv2 = None
        self.fv3 = None
        self.fv4 = None

        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
            )
        else:
            # image encoder in the ViT version
            vision_heads = vision_width // 64
            self.visual = VisionTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim
            )

        # text encoder
        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask(),
            type_input='text'
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype
    
    def init_adapters(self):
        # initialize the adapters after CLIP has loaded the weights
        vis_hidden_dim = int(self.vision_width / 2) # hidden dimension chosen as half of input embedding
        txt_hidden_dim = int(self.transformer_width / 2) # hidden dimension chosen as half of input embedding
        
        # 24 backbone apapters for vision encoder and 24 for text, 2 for each layer
        self.backbone_adapters_MHSA_vis = nn.Sequential(*[BackboneAdapter(self.vision_width, vis_hidden_dim) for _ in range(self.vision_layers)]).to(self.dtype)
        self.backbone_adapters_MLP_vis = nn.Sequential(*[BackboneAdapter(self.vision_width, vis_hidden_dim) for _ in range(self.vision_layers)]).to(self.dtype)
        self.backbone_adapters_MHSA_txt = nn.Sequential(*[BackboneAdapter(self.transformer_width, txt_hidden_dim) for _ in range(self.transformer_layers)]).to(self.dtype)
        self.backbone_adapters_MLP_txt = nn.Sequential(*[BackboneAdapter(self.transformer_width, txt_hidden_dim) for _ in range(self.transformer_layers)]).to(self.dtype)

        # 6 pre-fusion adapters (for the last 6 layers of encoders) and 6 post-fusion adapters attached after CLIP
        self.prefusion_adapters = nn.Sequential(*[PreFusionAdapter(self.vision_width, self.transformer_width, shared_dim=512, n_head=8) for _ in range(self.vision_layers-6)]).to(self.dtype)
        self.postfusion_adapters = nn.Sequential(*[PostFusionAdapter(shared_dim=self.visual.proj.shape[1], CA_n_head=8, MHSA_n_head=8, MLP_hidden_dim=256) for _ in range(6)]).to(self.dtype)
    
    def freeze_for_training(self):
        # freeze the model except the adapters added to CLIP
        for param in self.parameters():
            param.requires_grad = False
        for param in self.postfusion_adapters.parameters():
            param.requires_grad = True
        for param in self.prefusion_adapters.parameters():
            param.requires_grad = True
        for param in self.backbone_adapters_MHSA_vis.parameters():
            param.requires_grad = True
        for param in self.backbone_adapters_MHSA_txt.parameters():
            param.requires_grad = True
        for param in self.backbone_adapters_MLP_vis.parameters():
            param.requires_grad = True
        for param in self.backbone_adapters_MLP_txt.parameters():
            param.requires_grad = True
    
    def load_parameters(self, path):
        state_dict = torch.load(path)
        try:
            self.load_state_dict(state_dict)
        except RuntimeError:
            print(f"Failed to load state dict. N keys found: {len(state_dict.keys())}. N keys model: {len(self.state_dict().keys())}")

    def save_state_dict(self, path):
        torch.save(self.state_dict(), path)

    def encode(self, image, text):
        assert(isinstance(self.visual, VisionTransformer)) # check if the model is the ViT version
        assert(self.vision_layers == self.transformer.layers)# check if the number of layers in image and text encoder is the same
        assert(self.visual.proj.shape[1] ==  self.text_projection.shape[1]) # check if the final embedding dimension is the same

        # processing of image before the transformer is applied
        x_image = image.type(self.dtype)
        x_image = self.visual.conv1(x_image)
        x_image = x_image.reshape(x_image.shape[0], x_image.shape[1], -1)
        x_image = x_image.permute(0, 2, 1)
        x_image = torch.cat([self.visual.class_embedding.to(x_image.dtype) + torch.zeros(x_image.shape[0], 1, x_image.shape[-1], dtype=x_image.dtype, device=x_image.device), x_image], dim=1)
        x_image = x_image + self.visual.positional_embedding.to(x_image.dtype)
        x_image = self.visual.ln_pre(x_image)
        x_image = x_image.permute(1, 0, 2)

        # processing of text before the transformer is applied
        x_text = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]
        x_text = x_text + self.positional_embedding.type(self.dtype)
        x_text = x_text.permute(1, 0, 2)

        # apply each layer in both image and text encoders one at a time
        # because we need the output of each layer for both modalities to apply pre-fusion adapters
        # (prefusion adapters fuse information from modalities at each layer)
        for i in range(self.vision_layers):
            if i > 5:
                # apply pre-fusion adapters only for the last 6 layers
                v, t = self.prefusion_adapters[i-6](x_image, x_text)
                x_image = x_image + v
                x_text = x_text + t
                # always apply backbone adapters
                x_image = self.visual.transformer.resblocks[i](x_image, self.backbone_adapters_MHSA_vis[i], self.backbone_adapters_MLP_vis[i])
                x_text = self.transformer.resblocks[i](x_text, self.backbone_adapters_MHSA_txt[i], self.backbone_adapters_MLP_txt[i])
            else:
                # for the first 4 layers save the output of visual transformer to use it later in the refiner
                if i == 1:
                    self.fv1 = x_image
                elif i == 2:
                    self.fv2 = x_image
                elif i == 3:
                    self.fv3 = x_image
                elif i == 4:
                    self.fv4 = x_image
                # always apply backbone adapters
                x_image = self.visual.transformer.resblocks[i](x_image, self.backbone_adapters_MHSA_vis[i], self.backbone_adapters_MLP_vis[i])
                x_text = self.transformer.resblocks[i](x_text, self.backbone_adapters_MHSA_txt[i], self.backbone_adapters_MLP_txt[i])

        # then perform the last operations in both encoders to obtain the final token embeddings

        x_image = x_image.permute(1, 0, 2) # batch, CLS+patches, features
        x_text = x_text.permute(1, 0, 2) # batch, seq, features
        patch_tokens = x_image # from now on we care only about the patch tokens
        

        ##### this is not really needed anymore, since we don't use the CLS token ##############
        x_image = self.visual.ln_post(x_image[:, 0, :]) # take CLS token and layer norm
        if self.visual.proj is not None:
            x_image = x_image @ self.visual.proj # final proj into shared space
        #######################################################################################


        x_text = self.ln_final(x_text).type(self.dtype) # layer norm
        # text tokens projected in shared space
        text_tokens = x_text[torch.arange(x_text.shape[0])] @ self.text_projection
        # for each batch, take the last token (EOT) and project it into the shared space
        x_text = x_text[torch.arange(x_text.shape[0]), text.argmax(dim=-1)] @ self.text_projection


        # layer norm and projection into shared space for patch tokens
        patch_tokens = self.visual.ln_post(patch_tokens)
        if self.visual.proj is not None:
            patch_tokens = patch_tokens @ self.visual.proj
        
        patch_tokens = patch_tokens.permute(1, 0, 2)
        text_tokens = text_tokens.permute(1, 0, 2)
        for i in range(len(self.postfusion_adapters)):
            v, t = self.postfusion_adapters[i](patch_tokens, text_tokens)
            patch_tokens = v + patch_tokens
            text_tokens = t + text_tokens

        
        # take patch_tokens[:, 1:, :] and compute similarity with out_text
        # then take the similarity vector for each batch and reshape it in 14x14 and return it in a list with all the maps of the batch

        tokens = patch_tokens.permute(1, 0, 2)[:, 1:, :]
        out_text = text_tokens.permute(1, 0, 2)[torch.arange(text_tokens.shape[1]), text.argmax(dim=-1)]

        maps = []
        for i in range(tokens.shape[0]):
            # map = 1 - torch.cosine_similarity(tokens[i], out_text[i])
            map = torch.cosine_similarity(tokens[i], out_text[i])
            map = map.reshape(14, 14)
            maps.append(map)
        
        maps = torch.stack(maps)

        # return the probability maps for the current batch
        # and the features of the first 4 layers of the visual encoder
        # used later in the refiner
        return maps, [self.fv1.permute(1, 0, 2), 
                      self.fv1.permute(1, 0, 2), 
                      self.fv1.permute(1, 0, 2), 
                      self.fv1.permute(1, 0, 2)]
    
    def encode_image(self, image):
        return self.visual(image.type(self.dtype))

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def build_model(state_dict: dict):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks")))

    model = CLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    convert_weights(model)
    model.load_state_dict(state_dict)
    return model.eval()
