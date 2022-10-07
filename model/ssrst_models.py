import math
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import os
import wget
os.environ['TORCH_HOME'] = '../pretrained_models'
import timm
from timm.models.layers import to_2tuple,trunc_normal_

# override the timm package to relax the input shape constraint.
# 即简单的整除。。。
class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):  #默认参数在实际中都用不到，除了patch_size默认16以外。 如in_chans在语音中通常是1
        super().__init__()

        img_size = to_2tuple(img_size)      #只用来算默认的num_patches，没啥实际作用
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])  
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches   #在外面会被实际的num_patches修改

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)  #  在外面也会被修改， self.v.patch_embed.proj = new_proj

    def forward(self, x):
        # print("-------------doing patchembed----------------")
        # print("self.patch_size: ", self.patch_size)
        # print("self.num_patches: ", self.num_patches)
        # print("self.img_size: ", self.img_size)
        # x 的输入是 (batch_size, 1, frequency_bins, time_frame_num)
        x = self.proj(x).flatten(2).transpose(1, 2) #flatten(2)把第二维之后的数据拉直为1维， 即（a, b, c, d) >> (a, b, c * d)，即把所有的patches拉直,再和通道数交换，即x的shape(b, num_patches, bins)
        return x


class ASTModel(nn.Module):
    """
    The AST model.
    :param label_dim: the label dimension, i.e., the number of total classes, it is 527 for AudioSet, 50 for ESC-50, and 35 for speechcommands v2-35
    :param fstride: the stride of patch spliting on the frequency dimension, for 16*16 patchs, fstride=16 means no overlap, fstride=10 means overlap of 6
    :param tstride: the stride of patch spliting on the time dimension, for 16*16 patchs, tstride=16 means no overlap, tstride=10 means overlap of 6
    :param input_fdim: the number of frequency bins of the input spectrogram, mel_bin, spec_h
    :param input_tdim: the number of time frames of the input spectrogram
    :param imagenet_pretrain: if use ImageNet pretrained model
    :param audioset_pretrain: if use full AudioSet and ImageNet pretrained model
    :param model_size: the model size of AST, should be in [tiny224, small224, base224, base384], base224 and base 384 are same model, but are trained differently during ImageNet pretraining.
    """
    def __init__(self, label_dim=527, fstride=16, tstride=16, input_fdim=80, input_tdim=301, imagenet_pretrain=False, audioset_pretrain=False, model_size='base384', verbose=True, **kw):

        super(ASTModel, self).__init__()
        assert timm.__version__ == '0.4.5', 'Please use timm == 0.4.5, the code might not be compatible with newer versions.'

        self.spec_w, self.spec_h = input_tdim, input_fdim
        if verbose == True:
            print('---------------SSRST Model Summary---------------')
            print('ImageNet pretraining: {:s}, AudioSet pretraining: {:s}'.format(str(imagenet_pretrain),str(audioset_pretrain)))
        # override timm input shape restriction
        timm.models.vision_transformer.PatchEmbed = PatchEmbed

        # if AudioSet pretraining is not used (but ImageNet pretraining may still apply)
        if audioset_pretrain == False:
            #问题1,如何看到timm定义的模型源码？ 还有怎么看有哪些模型？
            if model_size == 'tiny224':
                self.v = timm.create_model('vit_deit_tiny_distilled_patch16_224', pretrained=imagenet_pretrain)
            elif model_size == 'small224':
                self.v = timm.create_model('vit_deit_small_distilled_patch16_224', pretrained=imagenet_pretrain)
            elif model_size == 'base224':
                self.v = timm.create_model('vit_deit_base_distilled_patch16_224', pretrained=imagenet_pretrain)
            elif model_size == 'base384':
                self.v = timm.create_model('vit_deit_base_distilled_patch16_384', pretrained=imagenet_pretrain)
            else:
                raise Exception('Model size must be one of tiny224, small224, base224, base384.')
            self.original_num_patches = self.v.patch_embed.num_patches  #所有的patches个数，model类的self.patch_embed是timm.models.vision_transformer.PatchEmbed类
            #print('self.original_num_patches', self.original_num_patches)
            self.oringal_hw = int(self.original_num_patches ** 0.5) # 一行或列有多少个patches，因为图片是正方形处理，所以函数默认的hw是开方。之后计算新的位置编码有用，多还少补。
            #print('self.oringal_hw', self.oringal_hw)
            self.original_embedding_dim = self.v.pos_embed.shape[2] # vit 源代码 self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim)), 1是位置编码，embed_dim由预先链模型定义
 
            self.head = nn.Sequential(nn.LayerNorm(self.original_embedding_dim), nn.Linear(self.original_embedding_dim, label_dim))

            # automatcially get the intermediate shape
            f_dim, t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim)
            num_patches = f_dim * t_dim
            self.v.patch_embed.num_patches = num_patches

            if verbose == True:
                print('frequncey stride={:d}, time stride={:d}'.format(fstride, tstride))
                print('number of patches={:d}'.format(num_patches))

            # the linear projection layer, 通道数self.original_embedding_dim不变，tranformer的设计思想, 这里patches是重叠的！！！
            new_proj = torch.nn.Conv2d(1, self.original_embedding_dim, kernel_size=(16, 16), stride=(fstride, tstride))  #改变移动步长

            if imagenet_pretrain == True:
                # 因为图像是3通道，所以weight的对应也是3, (b, 3, 16, 16) >> (b, 1, 16, 16), 这里声音是单通道
                new_proj.weight = torch.nn.Parameter(torch.sum(self.v.patch_embed.proj.weight, dim=1).unsqueeze(1)) #融合3个通道的参数
 
                new_proj.bias = self.v.patch_embed.proj.bias # shape[192], 和输出维度一样 self.original_embedding_dim

            self.v.patch_embed.proj = new_proj

            # the positional embedding
            if imagenet_pretrain == True:
                # get the positional embedding from deit model, skip the first two tokens (cls token and distillation token), reshape it to original 2D shape (24*24).
                #print(self.v.pos_embed.shape) # torch.Size([1, 198, 192]), 198 = 2 + 196, 196 = 14 * 14， 14 = 224 / 16，这个是预训练初始模型的位置编码，所以用的是默认参数,即shape为 （1, 2 + original_num_patches， original_embedding_dim）

                new_pos_embed = self.v.pos_embed[:, 2:, :].detach().reshape(1, self.original_num_patches, self.original_embedding_dim).transpose(1, 2).reshape(1, self.original_embedding_dim, self.oringal_hw, self.oringal_hw)
                
                #因为位置编码也是训练过的，所以我们要适当改动契合我们的新输入，包括上面的new_proj.weight也是这个道理
                #和x = (batch_size, 1, frequency_bins, time_frame_num)对应，t_dim是最后一维
                # cut (from middle) or interpolate the second dimension of the positional embedding
                if t_dim <= self.oringal_hw:
                    #从self.oringal_hw中间取t_dim长度
                    new_pos_embed = new_pos_embed[:, :, :, int(self.oringal_hw / 2) - int(t_dim / 2): int(self.oringal_hw / 2) - int(t_dim / 2) + t_dim]
                else:
                    new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(self.oringal_hw, t_dim), mode='bilinear') # 双线性插值到指定维度，前两维不变

                # cut (from middle) or interpolate the first dimension of the positional embedding
                if f_dim <= self.oringal_hw:
                    new_pos_embed = new_pos_embed[:, :, int(self.oringal_hw / 2) - int(f_dim / 2): int(self.oringal_hw / 2) - int(f_dim / 2) + f_dim, :]
                else:
                    new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(f_dim, t_dim), mode='bilinear')
                # flatten the positional embedding
                new_pos_embed = new_pos_embed.reshape(1, self.original_embedding_dim, num_patches).transpose(1,2)
                # concatenate the above positional embedding with the cls token and distillation token of the deit model.
                self.v.pos_embed = nn.Parameter(torch.cat([self.v.pos_embed[:, :2, :].detach(), new_pos_embed], dim=1))
                #新的位置编码shape (1, num_patches + 2, bins)

            else:
                # if not use imagenet pretrained model, just randomly initialize a learnable positional embedding
                # TODO can use sinusoidal positional embedding instead
                new_pos_embed = nn.Parameter(torch.zeros(1, self.v.patch_embed.num_patches + 2, self.original_embedding_dim))
                self.v.pos_embed = new_pos_embed
                trunc_normal_(self.v.pos_embed, std=.02)  #利用正态分布生成一个点，如果这个点在[a,b]区间之外，那么就重新生成，直到在区间为止。

        # now load a model that is pretrained on both ImageNet and AudioSet
        elif audioset_pretrain == True:
            if audioset_pretrain == True and imagenet_pretrain == False:
                raise ValueError('currently model pretrained on only audioset is not supported, please set imagenet_pretrain = True to use audioset pretrained model.')
            if model_size != 'base384':
                raise ValueError('currently only has base384 AudioSet pretrained model.')
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if os.path.exists('../../pretrained_models/audioset_10_10_0.4593.pth') == False:
                # this model performs 0.4593 mAP on the audioset eval set
                audioset_mdl_url = 'https://www.dropbox.com/s/cv4knew8mvbrnvq/audioset_0.4593.pth?dl=1'
                wget.download(audioset_mdl_url, out='../../pretrained_models/audioset_10_10_0.4593.pth')
            sd = torch.load('../../pretrained_models/audioset_10_10_0.4593.pth', map_location=device)

            #循环定义ASTModel ？ audioset_pretrain=False 相当于不会再进入这个逻辑里，所以不会死循环
            audio_model = ASTModel(label_dim=527, fstride=10, tstride=10, input_fdim=128, input_tdim=1024, imagenet_pretrain=False, audioset_pretrain=False, model_size='base384', verbose=False)
            audio_model = torch.nn.DataParallel(audio_model)
            audio_model.load_state_dict(sd, strict=False)           #载入 strict=False
            self.v = audio_model.module.v               #module 是  torch.nn.DataParallel的属性，其实就是ASTModel类
            self.original_embedding_dim = self.v.pos_embed.shape[2]
            self.head = nn.Sequential(nn.LayerNorm(self.original_embedding_dim), nn.Linear(self.original_embedding_dim, label_dim))

            #测试经过PatchEmbed函数划分pathes，实际有多少个patch
            f_dim, t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim)
            num_patches = f_dim * t_dim
            self.v.patch_embed.num_patches = num_patches
            if verbose == True:
                print('frequncey stride={:d}, time stride={:d}'.format(fstride, tstride))
                print('number of patches={:d}'.format(num_patches))

            # 和上面类似了，只不过这里是针对audio set专用的，所以shape可以用硬编码
            new_pos_embed = self.v.pos_embed[:, 2:, :].detach().reshape(1, 1212, 768).transpose(1, 2).reshape(1, 768, 12, 101)
            # if the input sequence length is larger than the original audioset (10s), then cut the positional embedding
            if t_dim < 101:
                new_pos_embed = new_pos_embed[:, :, :, 50 - int(t_dim/2): 50 - int(t_dim/2) + t_dim]
            # otherwise interpolate
            else:
                new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(12, t_dim), mode='bilinear')

            if f_dim < 12:
                new_pos_embed = new_pos_embed[:, :, 6 - int(f_dim/2): 6 - int(f_dim/2) + f_dim, :]
            # otherwise interpolate
            elif f_dim > 12:
                new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(f_dim, t_dim), mode='bilinear')

            new_pos_embed = new_pos_embed.reshape(1, 768, num_patches).transpose(1, 2)
            self.v.pos_embed = nn.Parameter(torch.cat([self.v.pos_embed[:, :2, :].detach(), new_pos_embed], dim=1))

    def get_shape(self, fstride, tstride, input_fdim=128, input_tdim=1024):
        #模拟看形输出状, 计算实际的num_patches个数
        test_input = torch.randn(1, 1, input_fdim, input_tdim)
        test_proj = nn.Conv2d(1, self.original_embedding_dim, kernel_size=(16, 16), stride=(fstride, tstride))
        test_out = test_proj(test_input)
        f_dim = test_out.shape[2]
        t_dim = test_out.shape[3]
        return f_dim, t_dim
    
    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 2
        N = self.v.pos_embed.shape[1] - 2     #deit是2个
        if npatch == N and w == h: #等于不用插值，如果小于呢？
            return self.v.pos_embed
        class_pos_embed = self.v.pos_embed[:, :2]  #deit是2个
        patch_pos_embed = self.v.pos_embed[:, 2:]
        dim = x.shape[-1]
        w0 = w // self.v.patch_embed.patch_size[1]   #patch_embed.patch_size 是 （16，16）, patch后的宽度
        h0 = h // self.v.patch_embed.patch_size[0]
        #print(w0, h0, "w h", patch_pos_embed.shape, x.shape)
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        #print(patch_pos_embed.shape, self.spec_h // self.v.patch_embed.patch_size[0], self.spec_w // self.v.patch_embed.patch_size[1])
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, self.spec_h // self.v.patch_embed.patch_size[0], self.spec_w // self.v.patch_embed.patch_size[1], dim).permute(0, 3, 1, 2),
            scale_factor=(h0 / (self.spec_h // self.v.patch_embed.patch_size[0]), w0 / (self.spec_w  // self.v.patch_embed.patch_size[1])),
            mode='bicubic',
        )  #插值方式可能有影响，如果patch间有重叠，计算方式就不会是self.spec_h // self.v.patch_embed.patch_size[0]了
        # 插值 （b, c, h, w）-> (b, c, h * scale_factor[0], w * scale_factor[1])
        #print("after interploate",patch_pos_embed.shape)
        assert int(h0) == patch_pos_embed.shape[-2] and int(w0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed, patch_pos_embed), dim=1)

    def prepare_tokens(self, x):

        #x = x.transpose(2, 3)     #感觉这操作无关紧要？ 反正self.v.patch_embed(x)都是后两个维度作卷积

        B, _, h, w = x.shape
        x = self.v.patch_embed(x)      #论文中的划分patches和linear projection
        #print("self.v.patch_embed(x)", x.shape)    #x (b, num_patches, bins) ，bins 即original_embedding_dim

        #print("self.v.cls_token.shape & self.v.dist_token.shape", self.v.cls_token.shape, self.v.dist_token.shape) # (1, 1, bins)
        cls_tokens = self.v.cls_token.expand(B, -1, -1)
        dist_token = self.v.dist_token.expand(B, -1, -1)
        #print("cls_tokens & dist_token", cls_tokens.shape, dist_token.shape)   # (b, 1, bins)

        x = torch.cat((cls_tokens, dist_token, x), dim=1)
        #print("x = torch.cat((cls_tokens, dist_token, x), dim=1)", x.shape)     # (b, num_patches + 2, bins)

        # add positional encoding to each token
        x = x + self.interpolate_pos_encoding(x, w, h)

        return self.v.pos_drop(x)

    def prepare_tokens_no_interpolate(self, x):

        #x = x.transpose(2, 3)     #感觉这操作无关紧要？ 反正self.v.patch_embed(x)都是后两个维度作卷积
        x = self.v.patch_embed(x)      #论文中的划分patches和linear projection
        #print("self.v.patch_embed(x)", x.shape)    #x (b, num_patches, bins) ，bins 即original_embedding_dim
        B, n, _ = x.shape
        #print("self.v.cls_token.shape & self.v.dist_token.shape", self.v.cls_token.shape, self.v.dist_token.shape) # (1, 1, bins)
        cls_tokens = self.v.cls_token.expand(B, -1, -1)
        dist_token = self.v.dist_token.expand(B, -1, -1)
        #print("cls_tokens & dist_token", cls_tokens.shape, dist_token.shape)   # (b, 1, bins)

        x = torch.cat((cls_tokens, dist_token, x), dim=1)
        print(x.shape, self.v.pos_embed.shape, n)
        x = x + self.v.pos_embed[:, :(n + 2)]   #这要求模型定义的输入特征图t是最大的情况，之后的输入t_dim不会比他大
        return x

    @autocast()
    def forward(self, x):
        """
        :param x: the input spectrogram, expected shape: (batch_size, frequency_bins, time_frame_num), e.g., (12, 80, 301)
        :return: prediction
        """
        # expect input x = (batch_size, frequency_bins, time_frame_num), e.g., (12, 1024, 128)
        #print("b f",x.shape)
        x = self.prepare_tokens(x)
        #x = self.prepare_tokens_no_interpolate(x)
        #tranformer block不改变shape， attention的输入形状是     B, N, C = x.shape, attention参数只需指定C，而C即bins或self.original_embedding_dim是模型定义时决定的，是不变的！会根据输入音频大小变化的只有num_patches + 2的维度，位置编码看情况插值，所以ast可以处理不定长的音频序列。
        # C之和预先训练模型有关, tiny22是192, base384是768
        #print("before transformer:", x.shape)
        for blk in self.v.blocks:
            x = blk(x)
        x = self.v.norm(x)
        #print("self.v.norm(x)", x.shape)    #不改变shape

        #融合 Deit两个cls toke的信息用作分类，其它都不要， 是不是有点草率了？！
        x = (x[:, 0] + x[:, 1]) / 2    
        #print('x = (x[:, 0] + x[:, 1]) / 2', x.shape, self.original_embedding_dim)  # （b, bins)
        x = self.head(x)                           # (b, class_nums)
        return x

    # def get_last_selfattention(self, x):
    #     x = self.prepare_tokens(x)
    #     for i, blk in enumerate(self.v.blocks):
    #         if i < len(self.v.blocks) - 1:
    #             x = blk(x)
    #         else:
    #             # return attention of the last block
    #             return blk(x, return_attention=True)   #这个要重写block, 才能返回atten

    def get_intermediate_layers(self, x, n=1):
        x = self.prepare_tokens(x)
        # we return the output tokens from the `n` last blocks
        output = []
        for i, blk in enumerate(self.v.blocks):
            x = blk(x)
            if len(self.v.blocks) - i <= n:
                output.append(self.v.norm(x))
        return output

    def get_first_layers(self, x, n=1):
            x = self.prepare_tokens(x)
            # we return the output tokens from the `n` first blocks
            output = []
            for i, blk in enumerate(self.v.blocks):
                x = blk(x)
                if i < n:
                    output.append(self.v.norm(x))
                else:
                    return output
            return output

class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True, nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False)) #归一化主要是在相应的维度上进行了模大小的规整，方向保持不变。防止过拟合 weight = weight_g * weight_v/|weight_v| 一个是决定大小的weight_g,一个是决定方向的weight_v
        self.last_layer.weight_g.data.fill_(1) #初始，可学习
        if norm_last_layer: # teacher肯定不需要 
            self.last_layer.weight_g.requires_grad = False #所以weight_g 永远是1？ 不变相当于没有做norm

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        #print(x.shape)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x



if __name__ == '__main__':
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    mel_bin = 80
    ast_mdl = ASTModel(input_fdim=mel_bin, input_tdim=301 , model_size='tiny224')
    # input a batch of 10 spectrogram, each with 100 time frames and 128 frequency bins
    test_input = torch.rand([2, mel_bin, 201])
    test_input, ast_mdl = test_input.to(device), ast_mdl.to(device)
    test_output = ast_mdl(test_input)
    print(ast_mdl.original_embedding_dim)
    # output should be in shape [10, 527], i.e., 10 samples, each with prediction of 527 classes.
    print(test_output.shape)

