"""ResNet18 -> WiKG node communication -> the existing gated ABMIL/classifier.

WiKG reference: https://arxiv.org/html/2403.07719v1 (Eq. 6 and 8), and
WonderLandxD/WiKG model.py at eb3144fb8267f525c57e71ced5c5b2b6a1c42a93.
The Eq. 6 dot product is intentional: the upstream einsum uses two different
feature indices and therefore computes a product of sums instead.
"""

import math

import torch
from torch import nn

from downstream.shared_model import SharedE2EModel


class GatedAttentionMIL(nn.Module):
    """Attention over N region embeddings; output is one bag vector and N weights."""

    def __init__(self, input_dim=512, hidden_dim=128):
        super().__init__()
        self.attention_V = nn.Linear(input_dim, hidden_dim, bias=True)
        self.attention_U = nn.Linear(input_dim, hidden_dim, bias=True)
        self.attention_w = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, embeddings):
        gated = torch.tanh(self.attention_V(embeddings)) * torch.sigmoid(self.attention_U(embeddings))
        scores = self.attention_w(gated).squeeze(-1)
        weights = torch.softmax(scores, dim=0)
        pooled = (weights.unsqueeze(-1) * embeddings).sum(dim=0, keepdim=True)
        return pooled, weights


class WiKGCommunication(nn.Module):
    """Update all N nodes together, preserving [B,N,D]; self-loops are allowed."""

    def __init__(self, feature_dim=512, topk=6, dropout=0.3):
        super().__init__()
        if isinstance(topk, bool) or not isinstance(topk, int) or topk < 1:
            raise ValueError("wikg_topk must be a positive integer")
        if (isinstance(dropout, bool) or not isinstance(dropout, (int, float))
                or not math.isfinite(dropout) or not 0 <= dropout < 1):
            raise ValueError("wikg_dropout must be finite and in [0,1)")
        self.feature_dim = feature_dim
        self.topk = topk
        self.scale = feature_dim ** -0.5
        # Retain the official learned input transform without changing D.
        self.input_projection = nn.Sequential(nn.Linear(feature_dim, feature_dim), nn.LeakyReLU())
        self.W_head = nn.Linear(feature_dim, feature_dim)
        self.W_tail = nn.Linear(feature_dim, feature_dim)
        self.linear1 = nn.Linear(feature_dim, feature_dim)
        self.linear2 = nn.Linear(feature_dim, feature_dim)
        self.activation = nn.LeakyReLU()
        self.message_dropout = nn.Dropout(dropout)

    def forward(self, x, log_shapes=False):
        if x.ndim != 3 or x.shape[1] < 1 or x.shape[2] != self.feature_dim:
            raise ValueError("WiKG expects [B,N,D] with N >= 1 and D={}".format(self.feature_dim))
        x = self.input_projection(x)
        x = (x + x.mean(dim=1, keepdim=True)) * 0.5
        e_h, e_t = self.W_head(x), self.W_tail(x)
        attn_logit = (e_h * self.scale) @ e_t.transpose(-2, -1)
        k_eff = min(self.topk, x.shape[1])
        topk_weight, topk_index = torch.topk(attn_logit, k=k_eff, dim=-1)
        batch_index = torch.arange(x.shape[0], device=x.device).view(-1, 1, 1)
        neighbor_tails = e_t[batch_index, topk_index]
        topk_prob = torch.softmax(topk_weight, dim=-1).unsqueeze(-1)
        e_h_expand = e_h.unsqueeze(2).expand(-1, -1, k_eff, -1)
        relation = topk_prob * neighbor_tails + (1 - topk_prob) * e_h_expand
        gate = torch.tanh(e_h_expand + relation)
        # Paper Eq. (6): match the feature dimension in both operands.
        ka_weight = (neighbor_tails * gate).sum(dim=-1)
        ka_prob = torch.softmax(ka_weight, dim=-1)
        message = (ka_prob.unsqueeze(-1) * neighbor_tails).sum(dim=2)
        output = self.activation(self.linear1(e_h + message))
        output = output + self.activation(self.linear2(e_h * message))
        output = self.message_dropout(output)
        if log_shapes:
            print("WiKG input: {}; relation score: {}; effective topk: {}; "
                  "TopK: {}; neighbor features: {}; WiKG output: {}".format(
                      list(x.shape), list(attn_logit.shape), k_eff, list(topk_weight.shape),
                      list(neighbor_tails.shape), list(output.shape)), flush=True)
        return output


class ResNetWiKGABMIL(SharedE2EModel):
    model_name = "resnet_wikg_abmil"
    pooling_name = "wikg_gated_attention"
    downstream_description = (
        "WSI downstream: ResNet18 [N,512] -> WiKG [1,N,512] -> gated attention -> Linear(512,2)")

    @staticmethod
    def validate_downstream_args(args):
        required = {"wikg_topk", "wikg_dropout", "classification_gradpool"}
        if not isinstance(args, dict) or not required <= args.keys() or args.keys() - required - {"debug_shapes"}:
            raise ValueError("resnet_wikg_abmil args require wikg_topk, wikg_dropout, "
                             "classification_gradpool; optional debug_shapes")
        args = dict(args)
        args.setdefault("debug_shapes", False)
        if isinstance(args["wikg_topk"], bool) or not isinstance(args["wikg_topk"], int) or args["wikg_topk"] < 1:
            raise ValueError("wikg_topk must be a positive integer")
        dropout = args["wikg_dropout"]
        if (isinstance(dropout, bool) or not isinstance(dropout, (int, float))
                or not math.isfinite(dropout) or not 0 <= dropout < 1):
            raise ValueError("wikg_dropout must be finite and in [0,1)")
        for key in ("classification_gradpool", "debug_shapes"):
            if not isinstance(args[key], bool):
                raise ValueError("{} must be a YAML boolean".format(key))
        return args

    def __init__(self, sr, model_spec=None, downstream_args=None):
        args = self.validate_downstream_args(downstream_args)
        super().__init__(sr, model_spec)
        self.classifier = nn.Linear(512, 2)
        self.mil_head = GatedAttentionMIL(input_dim=512, hidden_dim=128)
        self.downstream_args = args
        self.classification_gradpool = args["classification_gradpool"]
        self.wikg = WiKGCommunication(512, args["wikg_topk"], args["wikg_dropout"])
        self._debug_shapes = args["debug_shapes"]
        self._encoder_shapes_logged = False
        self._bag_shapes_logged = False

    def encode_regions(self, lr, preserve_bn_stats=False, log_shapes=False):
        debug = self._debug_shapes and not self._encoder_shapes_logged
        result = super().encode_regions(lr, preserve_bn_stats, log_shapes or debug)
        if debug:
            self._encoder_shapes_logged = True
        return result

    def aggregate_with_attention(self, embeddings):
        debug = self._debug_shapes and not self._bag_shapes_logged
        if debug:
            print("HAT feature (logical full WSI; encoded in micro-batches): {}".format(
                [embeddings.shape[0], 64, 256, 256]), flush=True)
            print("ResNet region features (full WSI): {}".format(list(embeddings.shape)), flush=True)
        communicated = self.wikg(embeddings.unsqueeze(0), log_shapes=debug).squeeze(0)
        pooled, weights = self.mil_head(communicated)
        if debug:
            print("ABMIL output: {}".format(list(pooled.shape)), flush=True)
        return pooled, weights

    def aggregate_embeddings(self, embeddings):
        pooled, _ = self.aggregate_with_attention(embeddings)
        return pooled

    def classify(self, wsi_embedding):
        return self.classifier(wsi_embedding)

    def aggregate(self, embeddings):
        return self.aggregate_embeddings(embeddings)

    def forward_embeddings(self, embeddings):
        logits = super().forward_embeddings(embeddings)
        if self._debug_shapes and not self._bag_shapes_logged:
            print("logits: {}".format(list(logits.shape)), flush=True)
            self._bag_shapes_logged = True
        return logits

    def downstream_gradient_groups(self):
        return {"WSI classifier": self.classifier.parameters(),
                "Attention MIL": self.mil_head.parameters(),
                "WiKG communication": self.wikg.parameters()}
