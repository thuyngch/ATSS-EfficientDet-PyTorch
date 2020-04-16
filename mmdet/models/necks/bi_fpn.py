import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init

from mmdet.core import auto_fp16
from ..registry import NECKS
from mmdet.ops import ConvModule
from mmdet.ops.activation import build_activation_layer


def may_apply_conv_1x1(in_channel, out_channel,
		conv_cfg=None, norm_cfg=dict(type='BN'), act_cfg=None):
	if in_channel != out_channel:
		return ConvModule(
			in_channel, out_channel, 1,
			conv_cfg=conv_cfg, norm_cfg=norm_cfg,
			act_cfg=act_cfg, bias=True, inplace=False)
	else:
		return nn.Identity()


@NECKS.register_module
class BiFPN(nn.Module):
	def __init__(self,
				 in_channels,
				 out_channels,
				 num_outs,
				 stack,
				 start_level=0,
				 end_level=-1,
				 add_extra_convs=False,
				 extra_convs_on_inputs=True,
				 relu_before_extra_convs=False,
				 conv_cfg=dict(type='ConvDWS'),
				 norm_cfg=dict(type='BN'),
				 act_cfg=None,
				 activation=dict(type='ReLU'),
				 **kargs):

		super(BiFPN, self).__init__()
		assert isinstance(in_channels, list)
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.num_ins = len(in_channels)
		self.num_outs = num_outs
		self.stack = stack
		self.activation = activation
		self.relu_before_extra_convs = relu_before_extra_convs

		if end_level == -1:
			self.backbone_end_level = self.num_ins
			assert num_outs >= self.num_ins - start_level
		else:
			# if end_level < inputs, no extra level is allowed
			self.backbone_end_level = end_level
			assert end_level <= len(in_channels)
			assert num_outs == end_level - start_level

		self.start_level = start_level
		self.end_level = end_level
		self.add_extra_convs = add_extra_convs
		self.extra_convs_on_inputs = extra_convs_on_inputs

		# Extra convs
		self.fpn_convs = nn.ModuleList()
		extra_levels = num_outs - self.backbone_end_level + self.start_level
		if add_extra_convs and extra_levels >= 1:
			for i in range(extra_levels):
				if i == 0 and self.extra_convs_on_inputs:
					in_channels = self.in_channels[self.backbone_end_level - 1]
				else:
					in_channels = out_channels
				fpn_conv = may_apply_conv_1x1(
					in_channels, out_channels,
					conv_cfg=None, norm_cfg=norm_cfg, act_cfg=None)
				self.fpn_convs.append(fpn_conv)

		# Stacked BiFPN
		self.stack_bifpn_convs = nn.ModuleList()
		for idx in range(stack):
			in_channels = self.in_channels if idx==0 \
				else self.num_outs * [self.out_channels]

			self.stack_bifpn_convs.append(BiFPNModule(
				in_channels=in_channels,
				out_channel=self.out_channels, activation=activation,
				conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg))

		self.init_weights()

	def init_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				xavier_init(m, distribution='uniform')

	@auto_fp16()
	def forward(self, inputs):
		assert len(inputs) == len(self.in_channels)
		used_backbone_levels = len(inputs) - self.start_level
		outs = [inputs[idx] for idx in range(self.start_level, self.backbone_end_level)]

		# add extra levels
		if self.num_outs > used_backbone_levels:
			# use max pool to get more levels on top of outputs
			# (e.g., Faster R-CNN, Mask R-CNN)
			if not self.add_extra_convs:
				for i in range(self.num_outs - used_backbone_levels):
					outs.append(F.max_pool2d(outs[-1], 1, stride=2))
			# add conv layers on top of original feature maps (RetinaNet)
			else:
				if self.extra_convs_on_inputs:
					orig = inputs[self.backbone_end_level - 1]
					out = self.fpn_convs[0](orig)
					out = F.max_pool2d(out, kernel_size=3, stride=2, padding=1)
					outs.append(out)
				else:
					outs.append(self.fpn_convs[0](outs[-1]))

				for i in range(1, self.num_outs - used_backbone_levels):
					out = self.fpn_convs[i](F.relu(outs[-1])) \
						if self.relu_before_extra_convs else self.fpn_convs[i](outs[-1])
					out = F.max_pool2d(out, kernel_size=3, stride=2, padding=1)
					outs.append(out)

		# stacked bi-fpn
		for bifpn_module in self.stack_bifpn_convs:
			outs = bifpn_module(outs)
		return tuple(outs)


class BiFPNModule(nn.Module):
	def __init__(self,
				 in_channels,
				 out_channel,
				 start_level=1,
				 init=1.0,
				 eps=0.0001,
				 conv_cfg=dict(type='ConvDWS'),
				 norm_cfg=dict(type='BN'),
				 act_cfg=None,
				 activation=dict(type='ReLU'),
				 **kargs):

		super(BiFPNModule, self).__init__()
		self.eps = eps
		self.activation = build_activation_layer(activation)

		# 8 weights
		self.weights_2 = nn.Parameter(torch.Tensor(5,2).fill_(init)) # node 0 1 2 3 7
		self.weights_3 = nn.Parameter(torch.Tensor(3,3).fill_(init)) # node 4 5 6

		# 5 possible laterals: may exist at the first stack, but not at following stacks
		self.lateral_convs = nn.ModuleList()
		for path_idx in range(5):
			if path_idx <= 2: # direct connections from P3, P4, P5
				in_channel = in_channels[start_level+path_idx]
				l_conv = may_apply_conv_1x1(in_channel, out_channel,
					conv_cfg=None, norm_cfg=norm_cfg, act_cfg=None)
				self.lateral_convs.append(l_conv)
			else: # skip connections from P4, P5
				in_channel = in_channels[start_level+path_idx-2]
				l_conv = may_apply_conv_1x1(in_channel, out_channel,
					conv_cfg=None, norm_cfg=norm_cfg, act_cfg=None)
				self.lateral_convs.append(l_conv)

		# 8 node convs
		self.node_convs = nn.ModuleList()
		for _ in range(8):
			node_conv = ConvModule(out_channel, out_channel, 3, padding=1,
				conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg,
				bias=True, inplace=False)
			self.node_convs.append(node_conv)

	@auto_fp16()
	def forward(self, feats):
		# Normalize weights
		weights_2 = F.relu(self.weights_2)
		weights_2 = weights_2 / (weights_2.sum(dim=1, keepdim=True) + self.eps)
		weights_3 = F.relu(self.weights_3)
		weights_3 = weights_3 / (weights_3.sum(dim=1, keepdim=True) + self.eps)
		node_weights = [
			weights_2[0], weights_2[1], weights_2[2], weights_2[3],
			weights_3[0], weights_3[1], weights_3[2], weights_2[4],
		]
		nodes = []

		# Build node-0
		node_idx = 0
		inputs = [
			feats[3], # P6
			F.interpolate(feats[4], scale_factor=2, mode='nearest'), # P7
		]
		weights = node_weights[node_idx]
		node = inputs[0] * weights[0] + inputs[1] * weights[1]
		node = self.activation(node)
		node = self.node_convs[node_idx](node)
		nodes.append(node)

		# Build node-1
		node_idx = 1
		inputs = [
			self.lateral_convs[2](feats[2]), # P5
			F.interpolate(nodes[0], scale_factor=2, mode='nearest'), # node-0
		]
		weights = node_weights[node_idx]
		node = inputs[0] * weights[0] + inputs[1] * weights[1]
		node = self.activation(node)
		node = self.node_convs[node_idx](node)
		nodes.append(node)

		# Build node-2
		node_idx = 2
		inputs = [
			self.lateral_convs[1](feats[1]), # P4
			F.interpolate(nodes[1], scale_factor=2, mode='nearest'), # node-1
		]
		weights = node_weights[node_idx]
		node = inputs[0] * weights[0] + inputs[1] * weights[1]
		node = self.activation(node)
		node = self.node_convs[node_idx](node)
		nodes.append(node)

		# Build node-3
		node_idx = 3
		inputs = [
			self.lateral_convs[0](feats[0]), # P3
			F.interpolate(nodes[2], scale_factor=2, mode='nearest'), # node-2
		]
		weights = node_weights[node_idx]
		node = inputs[0] * weights[0] + inputs[1] * weights[1]
		node = self.activation(node)
		node = self.node_convs[node_idx](node)
		nodes.append(node)

		# Build node-4
		node_idx = 4
		inputs = [
			self.lateral_convs[3](feats[1]), # P4
			nodes[2], # node-2
			F.max_pool2d(nodes[3], kernel_size=3, stride=2, padding=1), # node-3
		]
		weights = node_weights[node_idx]
		node = inputs[0] * weights[0] + inputs[1] * weights[1] + inputs[2] * weights[2]
		node = self.activation(node)
		node = self.node_convs[node_idx](node)
		nodes.append(node)

		# Build node-5
		node_idx = 5
		inputs = [
			self.lateral_convs[4](feats[2]), # P5
			nodes[1], # node-1
			F.max_pool2d(nodes[4], kernel_size=3, stride=2, padding=1), # node-4
		]
		weights = node_weights[node_idx]
		node = inputs[0] * weights[0] + inputs[1] * weights[1] + inputs[2] * weights[2]
		node = self.activation(node)
		node = self.node_convs[node_idx](node)
		nodes.append(node)

		# Build node-6
		node_idx = 6
		inputs = [
			feats[3], # P6
			nodes[0], # node-0
			F.max_pool2d(nodes[5], kernel_size=3, stride=2, padding=1), # node-5
		]
		weights = node_weights[node_idx]
		node = inputs[0] * weights[0] + inputs[1] * weights[1] + inputs[2] * weights[2]
		node = self.activation(node)
		node = self.node_convs[node_idx](node)
		nodes.append(node)

		# Build node-7
		node_idx = 7
		inputs = [
			feats[4], # P7
			F.max_pool2d(nodes[6], kernel_size=3, stride=2, padding=1), # node-6
		]
		weights = node_weights[node_idx]
		node = inputs[0] * weights[0] + inputs[1] * weights[1]
		node = self.activation(node)
		node = self.node_convs[node_idx](node)
		nodes.append(node)

		# Return
		return nodes[3], nodes[4], nodes[5], nodes[6], nodes[7]

	def init_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				xavier_init(m, distribution='uniform')
