import timm
from ..registry import BACKBONES

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm


@BACKBONES.register_module
class TimmBackbone(nn.Module):
	def __init__(self, model_name, pretrained=True, norm_eval=True,
				out_indices=(1,2,3,4), frozen_stages=-1, **kwargs):

		super(TimmBackbone, self).__init__()
		self.model = timm.create_model(
			model_name,
			pretrained=pretrained,
			features_only=True,
			out_indices=out_indices,
			**kwargs)

		# trick: eval have effect on BatchNorm only
		self.norm_eval = norm_eval
		if norm_eval:
			for m in self.model.modules():
				if isinstance(m, _BatchNorm):
					m.eval()

		self.frozen_stages = frozen_stages
		if frozen_stages >=0:
			self.freeze(frozen_stages)

	def forward(self, x):
		outs = self.model(x)
		return outs

	def init_weights(self, pretrained):
		if pretrained:
			self.load_state_dict(pretrained)

	def freeze(self, block_idx):
		assert block_idx >=0
		for p in self.model.conv_stem.parameters():
			p.requires_grad = False
		for p in self.model.bn1.parameters():
			p.requires_grad = False
		for ith in range(block_idx+1):
			for p in self.model.blocks[ith].parameters():
				p.requires_grad = False

	def train(self, mode=True):
		super(TimmBackbone, self).train(mode)
		if mode and self.norm_eval:
			for m in self.model.modules():
				# trick: eval have effect on BatchNorm only
				if isinstance(m, _BatchNorm):
					m.eval()
