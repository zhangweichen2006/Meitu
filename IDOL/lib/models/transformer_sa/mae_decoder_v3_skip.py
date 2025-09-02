import torch
import torch.nn as nn
import numpy as np
from timm.models.vision_transformer import PatchEmbed, Block, checkpoint_seq
from typing import Union



def get_2d_sincos_pos_embed(embed_dim, grid_size, add_cls_token=False):
    """
    Create 2D sin/cos positional embeddings.

    Args:
        embed_dim (`int`):
            Embedding dimension.
        grid_size (`int`):
            The grid height and width.
        add_cls_token (`bool`, *optional*, defaults to `False`):
            Whether or not to add a classification (CLS) token.

    Returns:
        (`torch.FloatTensor` of shape (grid_size*grid_size, embed_dim) or (1+grid_size*grid_size, embed_dim): the
        position embeddings (with or without classification token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if add_cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be even")

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position pos: a list of positions to be encoded: size (M,) out: (M, D)
    """
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be even")

    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


    
class neck_SA_v3_skip(nn.Module):
    def __init__(self, patch_size=4, in_chans=32, num_patches=196, embed_dim=1024, decoder_embed_dim=512, 
                 decoder_depth=8, decoder_num_heads=16, mlp_ratio=4, norm_layer=nn.LayerNorm, total_num_hidden_states=25, connect_mode:Union['uniform', 'zeros', 'shadow']='uniform', if_checkpoint_seq=False):
        super().__init__()
        self.num_patches = num_patches
        # Decoder-specific

        self.if_checkpoint_seq = if_checkpoint_seq # to save the memory


        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches, decoder_embed_dim), requires_grad=True)  # fixed sin-cos embedding

        self.decoder_blocks_depart = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True,  norm_layer=norm_layer) #qk_scale=None
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch

        if connect_mode == 'uniform':
            skip = total_num_hidden_states// (decoder_depth-1)
            self.select_hidden_states = [skip*i for i in range(decoder_depth)] # for 25
            self.select_hidden_states[-1] = total_num_hidden_states - 1 
            self.select_hidden_states = self.select_hidden_states[::-1] # inverse the order
        elif connect_mode == 'zeros':
            # print('!!!!!!!!!!! zeros !!!!!!!!!!!!!!!!')
            self.select_hidden_states = [0, 0, 0, 0, 0, 0]
        self.decoder_embed = nn.ModuleList([
            nn.Linear(embed_dim, decoder_embed_dim, bias=True)
            for _ in range(decoder_depth) 
        ])
        self.initialize_weights()

    def initialize_weights(self):
        # Initialization
        # Initialize (and freeze) pos_embed by sin-cos embedding
        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int((self.num_patches)**.5), add_cls_token=False)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # Initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)
       

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # We use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_decoder(self, in_features, ids_restore):
        # Embed tokens
        B, N_l, N_f, C = in_features.shape
        select_in_features = in_features[:, self.select_hidden_states, :, :]
        # parallelly embed the hidden states wiht jit
        forks = [torch.jit.fork(self.decoder_embed[i], select_in_features[:, i]) for i in range(len(self.select_hidden_states))]
        x_list = [torch.jit.wait(fork) for fork in forks]
        x_all_states = torch.stack(x_list) # N_l, B, N_feat, C


        # Add pos embed
        mask_tokens = self.mask_token.repeat(B, ids_restore.shape[1], 1) # B, N_q, C
        query_x = mask_tokens + self.decoder_pos_embed

        # Append mask tokens to sequence
        x = torch.zeros_like(x_all_states[0])
        x = torch.cat([x, query_x], dim=1)  # no cls token # B, N_q+N_f, C
        # # Apply Transformer blocks # v0

        # Apply Transformer blocks # v1
        for i, blk in enumerate( self.decoder_blocks_depart):
            x_add = x_all_states[i]
            x[:, :N_f, :] += x_add # add the hidden states
            x = blk(x)

        x = self.decoder_norm(x)


        
        x = x[:, -self.num_patches:, :]
        x_reshaped = x

        return x_reshaped

    def forward(self, encoded_latent, ids_restore):
        decoded_output = self.forward_decoder(encoded_latent, ids_restore)
        return decoded_output