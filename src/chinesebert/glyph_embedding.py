# -*- coding: utf-8 -*-
"""
@author: Yuxian Meng
@contact: yuxian_meng@shannonai.com

@version: 1.0
@file: glyph_embedding
@time: 2020/8/4 15:04

"""


from torch import nn


class GlyphEmbedding(nn.Module):
    """Glyph2Image Embedding"""

    def __init__(self, num_embeddings, embedding_dim):
        super(GlyphEmbedding, self).__init__()
        self.embedding = nn.Embedding(
            num_embeddings=num_embeddings, embedding_dim=embedding_dim
        )

    def forward(self, input_ids):
        """
            get glyph images for batch inputs
        Args:
            input_ids: [batch, sentence_length]
        Returns:
            images: [batch, sentence_length, self.font_num*self.font_size*self.font_size]
        """
        # return self.embedding(input_ids).reshape([-1, self.font_num, self.font_size, self.font_size])
        return self.embedding(input_ids)
