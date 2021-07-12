# -*- coding: utf-8 -*-
"""
@file  : pinyin.py
@author: zijun
@contact : zijun_sun@shannonai.com
@date  : 2020/8/16 14:45
@version: 1.0
@desc  : pinyin embedding
"""

from torch import nn
from torch.nn import functional as F


class PinyinEmbedding(nn.Module):
    def __init__(self, pinyin_map_len: int, embedding_size: int, pinyin_out_dim: int):
        """
            Pinyin Embedding Module
        Args:
            embedding_size: the size of each embedding vector
            pinyin_out_dim: kernel number of conv
        """
        super(PinyinEmbedding, self).__init__()

        self.pinyin_out_dim = pinyin_out_dim
        self.embedding = nn.Embedding(pinyin_map_len, embedding_size)
        self.conv = nn.Conv1d(
            in_channels=embedding_size,
            out_channels=self.pinyin_out_dim,
            kernel_size=2,
            stride=1,
            padding=0,
        )

    def forward(self, pinyin_ids):
        """
        Args:
            pinyin_ids: (bs*sentence_length*pinyin_locs)

        Returns:
            pinyin_embed: (bs,sentence_length,pinyin_out_dim)
        """
        # input pinyin ids for 1-D conv
        embed = self.embedding(
            pinyin_ids
        )  # [bs,sentence_length,pinyin_locs,embed_size]

        bs, sentence_length, pinyin_locs, embed_size = embed.shape
        view_embed = embed.reshape(
            -1, pinyin_locs, embed_size
        )  # [(bs*sentence_length),pinyin_locs,embed_size]
        input_embed = view_embed.permute(
            0, 2, 1
        )  # [(bs*sentence_length), embed_size, pinyin_locs]
        # conv + max_pooling
        pinyin_conv = self.conv(input_embed)  # [(bs*sentence_length),pinyin_out_dim,H]
        pinyin_embed = F.max_pool1d(
            pinyin_conv, pinyin_conv.shape[-1]
        )  # [(bs*sentence_length),pinyin_out_dim,1]
        return pinyin_embed.reshape(
            bs, sentence_length, self.pinyin_out_dim
        )  # [bs,sentence_length,pinyin_out_dim]
