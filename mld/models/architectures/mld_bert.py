import torch
import os

from typing import List, Union
from torch import nn, Tensor
from torch.distributions.distribution import Distribution

from mld.models.operator import PositionalEncoding
from mld.utils.temos_utils import lengths_to_mask


class MLDTextEncoder(nn.Module):
    def __init__(self,
                 cfg,
                 modelpath: str,
                 finetune: bool = False,
                 vae: bool = True,
                 latent_dim: int = 256,
                 ff_size: int = 1024,
                 num_layers: int = 6,
                 num_heads: int = 4,
                 dropout: float = 0.1,
                 activation: str = "gelu",
                 **kwargs) -> None:

        super().__init__()

        from transformers import AutoTokenizer, AutoModel
        from transformers import logging

        logging.set_verbosity_error()
        # Tokenizer
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        self.tokenizer = AutoTokenizer.from_pretrained(modelpath)

        # Text model
        self.text_model = AutoModel.from_pretrained(modelpath)
        # Don't train the model
        if not finetune:
            self.text_model.training = False
            for p in self.text_model.parameters():
                p.requires_grad = False

        # Then configure the model
        self.text_encoded_dim = self.text_model.config.hidden_size
        self.text_encoded_dim = latent_dim  # enable projection
        # self.save_hyperparameters(logger=False)

        encoded_dim = self.text_model.config.hidden_size

        # Projection of the text-outputs into the latent space
        self.projection = nn.Sequential(nn.ReLU(),
                                        nn.Linear(encoded_dim, latent_dim))

        # TransformerVAE adapted from ACTOR
        # Action agnostic: only one set of params

        vae = False
        if vae:
            self.mu_token = nn.Parameter(torch.randn(latent_dim))
            self.logvar_token = nn.Parameter(torch.randn(latent_dim))
        else:
            self.global_text_token = nn.Parameter(torch.randn(latent_dim))

        self.sequence_pos_encoding = PositionalEncoding(latent_dim, dropout)
        seq_trans_encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=ff_size,
            dropout=dropout,
            activation=activation)
        self.seqTransEncoder = nn.TransformerEncoder(seq_trans_encoder_layer,
                                                     num_layers=num_layers)

        # for action part
        # self.is_action_branch = cfg.model.DIST_ADAIN
        # self.is_cross_token = cfg.model.CROSS_TOKEN
        # if self.is_cross_token:
        #     self.mean_token = nn.Parameter(torch.randn(latent_dim))
        #     self.std_token = nn.Parameter(torch.randn(latent_dim))

        if self.is_action_branch:
            action_trans_encoder_layer = nn.TransformerEncoderLayer(
                d_model=latent_dim,
                nhead=num_heads,
                dim_feedforward=ff_size,
                dropout=dropout,
                activation=activation)
            self.actionTransEncoder = nn.TransformerEncoder(
                action_trans_encoder_layer, num_layers=num_layers)
            self.mean_token = nn.Parameter(torch.randn(latent_dim))
            self.std_token = nn.Parameter(torch.randn(latent_dim))

    def global_branch(self, x, mask):
        bs = x.shape[0]

        # Switch sequence and batch_size because the input of
        # Pytorch Transformer is [Sequence, Batch size, ...]
        x = x.permute(1, 0, 2)  # now it is [nframes, bs, latent_dim]

        # if self.vae:
        #     mu_token = torch.tile(self.mu_token, (bs,)).reshape(bs, -1)
        #     logvar_token = torch.tile(self.logvar_token, (bs,)).reshape(bs, -1)

        #     # adding the distribution tokens for all sequences
        #     xseq = torch.cat((mu_token[None], logvar_token[None], x), 0)

        #     # create a bigger mask, to allow attend to mu and logvar
        #     token_mask = torch.ones((bs, 2), dtype=bool, device=x.device)
        #     aug_mask = torch.cat((token_mask, mask), 1)
        # else:
        global_tokens = torch.tile(self.global_text_token,
                                   (bs, )).reshape(bs, -1)

        if self.is_cross_token:
            mean_tokens = torch.tile(self.mean_token, (bs, )).reshape(bs, -1)
            std_tokens = torch.tile(self.std_token, (bs, )).reshape(bs, -1)
            # adding the embedding token for all sequences
            xseq = torch.cat(
                (mean_tokens[None], std_tokens[None], global_tokens[None], x),
                0)

            # create a bigger mask, to allow attend to emb
            token_mask = torch.ones((bs, 3), dtype=bool, device=x.device)
            aug_mask = torch.cat((token_mask, mask), 1)
        else:
            # adding the embedding token for all sequences
            xseq = torch.cat((global_tokens[None], x), 0)

            # create a bigger mask, to allow attend to global
            token_mask = torch.ones((bs, 1), dtype=bool, device=x.device)
            aug_mask = torch.cat((token_mask, mask), 1)

        # add positional encoding
        xseq = self.sequence_pos_encoding(xseq)
        # content encode
        text_tokens = self.seqTransEncoder(xseq,
                                           src_key_padding_mask=~aug_mask)
        return text_tokens

    def action_branch(self, x, mask):
        bs = x.shape[0]
        mean_tokens = torch.tile(self.mean_token, (bs, )).reshape(bs, -1)
        std_tokens = torch.tile(self.std_token, (bs, )).reshape(bs, -1)

        # adding the embedding token for all sequences
        actionSeq = torch.cat((mean_tokens[None], std_tokens[None], x), 0)

        # create a bigger mask, to allow attend to emb
        token_mask = torch.ones((bs, 2), dtype=bool, device=x.device)
        aug_mask = torch.cat((token_mask, mask), 1)

        # Pass through the transformer decoder
        # with the latent vector for memory
        # add positional encoding
        actionSeq = self.sequence_pos_encoding(actionSeq)
        action_tokens = self.actionTransEncoder(actionSeq,
                                                src_key_padding_mask=~aug_mask)
        return action_tokens[0:2]

    def forward(self, texts: List[str]):
        text_encoded, mask = self.get_last_hidden_state(texts,
                                                        return_mask=True)
        text_emb = self.projection(text_encoded)

        # text_tokens = self.global_branch(text_emb, mask)

        # if self.is_action_branch:
        #     action_dist = self.action_branch(text_emb, mask)
        #     tokens = text_tokens
        # elif self.is_cross_token:
        #     action_dist = text_tokens[0:2]
        #     tokens = text_tokens[2:]
        # else:
        #     tokens = text_tokens
        #     action_dist = None

        # # content distribution
        # mu, logvar = content[0], content[1]
        # std = logvar.exp().pow(0.5)
        # dist = torch.distributions.Normal(mu, std)
        # return tokens[0], tokens[1:], action_dist, text_emb
        return text_emb

    def get_last_hidden_state(self,
                              texts: List[str],
                              return_mask: bool = False
                              ):  #-> Union[Tensor, tuple[Tensor, Tensor]]:
        encoded_inputs = self.tokenizer(texts,
                                        return_tensors="pt",
                                        padding=True)
        output = self.text_model(**encoded_inputs.to(self.text_model.device))
        if not return_mask:
            return output.last_hidden_state
        return output.last_hidden_state, encoded_inputs.attention_mask.to(
            dtype=bool)
