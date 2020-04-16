import torch
import torch.nn as nn
import torch.distributed as dist

import numpy as np
from mmcv.cnn import normal_init

from ..registry import HEADS
from ..builder import build_loss
from .anchor_head import AnchorHead
from ..utils import bias_init_with_prob

from mmdet.ops import ConvModule, BnModule, DepthwiseSeparableConv2d
from mmdet.core import (PseudoSampler, anchor_inside_flags, bbox2delta,
						build_assigner, delta2bbox, force_fp32,
						images_to_levels, multi_apply, multiclass_nms, unmap)


def reduce_mean(tensor):
	if not (dist.is_available() and dist.is_initialized()):
		return tensor
	tensor = tensor.clone()
	dist.all_reduce(tensor.div_(dist.get_world_size()), op=dist.reduce_op.SUM)
	return tensor


#------------------------------------------------------------------------------
#  EffDetHead
#------------------------------------------------------------------------------
@HEADS.register_module
class EffDetHead(AnchorHead):

	def __init__(self,
				 num_classes,
				 in_channels,
				 stacked_convs,
				 num_levels,
				 octave_base_scale=4,
				 scales_per_octave=3,
				 conv_cfg=dict(type='ConvDWS'),
				 norm_cfg=dict(type='BN'),
				 act_cfg=dict(type='ReLU'),
				 **kwargs):

		self.stacked_convs = stacked_convs
		self.num_levels = num_levels
		self.octave_base_scale = octave_base_scale
		self.scales_per_octave = scales_per_octave
		self.conv_cfg = conv_cfg
		self.norm_cfg = norm_cfg
		self.act_cfg = act_cfg

		octave_scales = np.array(
			[2**(i / scales_per_octave) for i in range(scales_per_octave)])
		anchor_scales = octave_scales * octave_base_scale
		super(EffDetHead, self).__init__(
			num_classes, in_channels, anchor_scales=anchor_scales, **kwargs)

	def _init_layers(self):
		# Build list of intermediate convolutions
		self.cls_convs = nn.ModuleList()
		self.reg_convs = nn.ModuleList()
		for i in range(self.stacked_convs):
			chn = self.in_channels if i == 0 else self.feat_channels
			self.cls_convs.append(ConvModule(
				chn, self.feat_channels, 3, padding=1,
				conv_cfg=self.conv_cfg, norm_cfg=None, act_cfg=None))
			self.reg_convs.append(ConvModule(
				chn, self.feat_channels, 3, padding=1,
				conv_cfg=self.conv_cfg, norm_cfg=None, act_cfg=None))

		# Build list of batchnorms
		self.cls_bns = nn.ModuleList()
		self.reg_bns = nn.ModuleList()
		for depth_idx in range(self.stacked_convs):
			cls_bns = nn.ModuleList()
			reg_bns = nn.ModuleList()
			chn = self.in_channels if (depth_idx == 0) else self.feat_channels
			for level_idx in range(self.num_levels):
				cls_bns.append(BnModule(chn, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg))
				reg_bns.append(BnModule(chn, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg))
			self.cls_bns.append(cls_bns)
			self.reg_bns.append(reg_bns)

		# Build final convolutions
		self.predict_cls = ConvModule(
			self.feat_channels, self.num_anchors * self.cls_out_channels,
			3, padding=1, conv_cfg=self.conv_cfg, norm_cfg=None, act_cfg=None)
		self.predict_reg = ConvModule(
			self.feat_channels, self.num_anchors * 4,
			3, padding=1, conv_cfg=self.conv_cfg, norm_cfg=None, act_cfg=None)

	def init_weights(self):
		for m in self.cls_convs:
			self._init_conv_weight(m)
		for m in self.reg_convs:
			self._init_conv_weight(m)
		bias_cls = bias_init_with_prob(0.01)
		self._init_conv_weight(self.predict_cls, bias=bias_cls)
		self._init_conv_weight(self.predict_reg)

	def _init_conv_weight(self, module, std=0.01, bias=0):
		if isinstance(module.conv, DepthwiseSeparableConv2d):
			normal_init(module.conv.depthwise, std=std, bias=bias)
			normal_init(module.conv.pointwise, std=std, bias=bias)
		else:
			normal_init(module.conv, std=std, bias=bias)

	def forward(self, feats):
		level_ids = range(len(feats))
		return multi_apply(self.forward_single, feats, level_ids)

	def forward_single(self, x, level_idx):
		cls_feat = x
		reg_feat = x
		for cls_conv, cls_bns, reg_conv, reg_bns in zip(
				self.cls_convs, self.cls_bns, self.reg_convs, self.reg_bns):

			# Cls feat
			cls_feat = cls_conv(cls_feat)	# Share between levels
			cls_feat = cls_bns[level_idx](cls_feat)	# Not share between levels

			# Reg feat
			reg_feat = reg_conv(reg_feat)	# Share between levels
			reg_feat = reg_bns[level_idx](reg_feat)	# Not share between levels

		cls_score = self.predict_cls(cls_feat)
		bbox_pred = self.predict_reg(reg_feat)
		return cls_score, bbox_pred

	@force_fp32(apply_to=('cls_scores', 'bbox_preds'))
	def get_bboxes(self, cls_scores, bbox_preds, img_metas, cfg, rescale=False):
		assert len(cls_scores) == len(bbox_preds)
		num_levels = len(cls_scores)

		device = cls_scores[0].device
		mlvl_anchors = [
			self.anchor_generators[i].grid_anchors(
				cls_scores[i].size()[-2:],
				self.anchor_strides[i],
				device=device) for i in range(num_levels)
		]
		result_list = []
		for img_id in range(len(img_metas)):
			cls_score_list = [
				cls_scores[i][img_id].detach() for i in range(num_levels)
			]
			bbox_pred_list = [
				bbox_preds[i][img_id].detach() for i in range(num_levels)
			]
			img_shape = img_metas[img_id]['img_shape']
			scale_factor = img_metas[img_id]['scale_factor']

			proposals = self.get_bboxes_single(cls_score_list, bbox_pred_list,
				mlvl_anchors, img_shape, scale_factor, cfg, rescale)

			result_list.append(proposals)
		return result_list

	def get_bboxes_single(self, cls_score_list, bbox_pred_list, mlvl_anchors,
						img_shape, scale_factor, cfg, rescale=False):
		"""
		Transform outputs for a single batch item into labeled boxes.
		"""
		assert len(cls_score_list) == len(bbox_pred_list) == len(mlvl_anchors)
		mlvl_bboxes = []
		mlvl_scores = []
		for cls_score, bbox_pred, anchors in zip(cls_score_list, bbox_pred_list, mlvl_anchors):
			assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
			cls_score = cls_score.permute(1, 2, 0).reshape(-1, self.cls_out_channels)

			if self.use_sigmoid_cls:
				scores = cls_score.sigmoid()
			else:
				scores = cls_score.softmax(-1)

			bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
			nms_pre = cfg.get('nms_pre', -1)

			if nms_pre > 0 and scores.shape[0] > nms_pre:
				# Get maximum scores for foreground classes.
				if self.use_sigmoid_cls:
					max_scores, _ = scores.max(dim=1)
				else:
					max_scores, _ = scores[:, 1:].max(dim=1)

				_, topk_inds = max_scores.topk(nms_pre)
				anchors = anchors[topk_inds, :]
				bbox_pred = bbox_pred[topk_inds, :]
				scores = scores[topk_inds, :]

			bboxes = delta2bbox(
				anchors, bbox_pred, self.target_means, self.target_stds, img_shape)
			
			mlvl_bboxes.append(bboxes)
			mlvl_scores.append(scores)

		mlvl_scores = torch.cat(mlvl_scores)
		mlvl_bboxes = torch.cat(mlvl_bboxes)
		if rescale:
			mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)

		if self.use_sigmoid_cls:
			# Add a dummy background class to the front when using sigmoid
			padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
			mlvl_scores = torch.cat([padding, mlvl_scores], dim=1)

		det_bboxes, det_labels = multiclass_nms(
			mlvl_bboxes, mlvl_scores, cfg.score_thr, cfg.nms, cfg.max_per_img)

		return det_bboxes, det_labels


#------------------------------------------------------------------------------
#  ATSSEffDetHead
#------------------------------------------------------------------------------
@HEADS.register_module
class ATSSEffDetHead(EffDetHead):

	def __init__(self,
				 num_classes,
				 in_channels,
				 stacked_convs,
				 num_levels,
				 octave_base_scale=4,
				 scales_per_octave=1,
				 conv_cfg=dict(type='ConvDWS'),
				 norm_cfg=dict(type='BN'),
				 act_cfg=dict(type='ReLU'),
				 loss_centerness=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
				 **kwargs):

		super(ATSSEffDetHead, self).__init__(
			num_classes, in_channels, stacked_convs, num_levels,
			octave_base_scale, scales_per_octave,
			conv_cfg, norm_cfg, act_cfg, **kwargs)

		self.loss_centerness = build_loss(loss_centerness)

	def _init_layers(self):
		super(ATSSEffDetHead, self)._init_layers()
		self.predict_centerness = ConvModule(
			self.feat_channels, self.num_anchors, 3,
			padding=1, conv_cfg=self.conv_cfg, norm_cfg=None, act_cfg=None)

	def init_weights(self):
		super(ATSSEffDetHead, self).init_weights()
		self._init_conv_weight(self.predict_centerness)

	def forward_single(self, x, level_idx):
		cls_feat = x
		reg_feat = x
		for cls_conv, cls_bns, reg_conv, reg_bns in zip(
				self.cls_convs, self.cls_bns, self.reg_convs, self.reg_bns):

			# Cls feat
			cls_feat = cls_conv(cls_feat)	# Share between levels
			cls_feat = cls_bns[level_idx](cls_feat)	# Not share between levels

			# Reg feat
			reg_feat = reg_conv(reg_feat)	# Share between levels
			reg_feat = reg_bns[level_idx](reg_feat)	# Not share between levels

		cls_score = self.predict_cls(cls_feat)
		bbox_pred = self.predict_reg(reg_feat)
		centerness = self.predict_centerness(reg_feat)
		return cls_score, bbox_pred, centerness

	def loss_single(self, anchors, cls_score, bbox_pred, centerness, labels,
					label_weights, bbox_targets, num_total_samples, cfg):

		anchors = anchors.reshape(-1, 4)
		cls_score = cls_score.permute(0, 2, 3,
									  1).reshape(-1, self.cls_out_channels)
		bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
		centerness = centerness.permute(0, 2, 3, 1).reshape(-1)
		bbox_targets = bbox_targets.reshape(-1, 4)
		labels = labels.reshape(-1)
		label_weights = label_weights.reshape(-1)

		# classification loss
		loss_cls = self.loss_cls(
			cls_score, labels, label_weights, avg_factor=num_total_samples)

		pos_inds = torch.nonzero(labels).squeeze(1)

		if len(pos_inds) > 0:
			pos_bbox_targets = bbox_targets[pos_inds]
			pos_bbox_pred = bbox_pred[pos_inds]
			pos_anchors = anchors[pos_inds]
			pos_centerness = centerness[pos_inds]

			centerness_targets = self.centerness_target(
				pos_anchors, pos_bbox_targets)
			pos_decode_bbox_pred = delta2bbox(pos_anchors, pos_bbox_pred,
											  self.target_means,
											  self.target_stds)
			pos_decode_bbox_targets = delta2bbox(pos_anchors, pos_bbox_targets,
												 self.target_means,
												 self.target_stds)

			# regression loss
			loss_bbox = self.loss_bbox(
				pos_decode_bbox_pred,
				pos_decode_bbox_targets,
				weight=centerness_targets,
				avg_factor=1.0)

			# centerness loss
			loss_centerness = self.loss_centerness(
				pos_centerness,
				centerness_targets,
				avg_factor=num_total_samples)

		else:
			loss_bbox = bbox_pred.sum() * 0
			loss_centerness = centerness.sum() * 0
			centerness_targets = torch.tensor(0).cuda()

		return loss_cls, loss_bbox, loss_centerness, centerness_targets.sum()

	@force_fp32(apply_to=('cls_scores', 'bbox_preds', 'centernesses'))
	def loss(self,
			 cls_scores,
			 bbox_preds,
			 centernesses,
			 gt_bboxes,
			 gt_labels,
			 img_metas,
			 cfg,
			 gt_bboxes_ignore=None):

		featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
		assert len(featmap_sizes) == len(self.anchor_generators)

		device = cls_scores[0].device
		anchor_list, valid_flag_list = self.get_anchors(
			featmap_sizes, img_metas, device=device)
		label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1

		cls_reg_targets = self.atss_target(
			anchor_list,
			valid_flag_list,
			gt_bboxes,
			img_metas,
			cfg,
			gt_bboxes_ignore_list=gt_bboxes_ignore,
			gt_labels_list=gt_labels,
			label_channels=label_channels)
		if cls_reg_targets is None:
			return None

		(anchor_list, labels_list, label_weights_list, bbox_targets_list,
		 bbox_weights_list, num_total_pos, num_total_neg) = cls_reg_targets

		num_total_samples = reduce_mean(
			torch.tensor(num_total_pos).cuda()).item()
		num_total_samples = max(num_total_samples, 1.0)

		losses_cls, losses_bbox, loss_centerness,\
			bbox_avg_factor = multi_apply(
				self.loss_single,
				anchor_list,
				cls_scores,
				bbox_preds,
				centernesses,
				labels_list,
				label_weights_list,
				bbox_targets_list,
				num_total_samples=num_total_samples,
				cfg=cfg)

		bbox_avg_factor = sum(bbox_avg_factor)
		bbox_avg_factor = reduce_mean(bbox_avg_factor).item()
		losses_bbox = list(map(lambda x: x / bbox_avg_factor, losses_bbox))
		return dict(
			loss_cls=losses_cls,
			loss_bbox=losses_bbox,
			loss_centerness=loss_centerness)

	def centerness_target(self, anchors, bbox_targets):
		# only calculate pos centerness targets, otherwise there may be nan
		gts = delta2bbox(anchors, bbox_targets, self.target_means,
						 self.target_stds)
		anchors_cx = (anchors[:, 2] + anchors[:, 0]) / 2
		anchors_cy = (anchors[:, 3] + anchors[:, 1]) / 2
		l_ = anchors_cx - gts[:, 0]
		t_ = anchors_cy - gts[:, 1]
		r_ = gts[:, 2] - anchors_cx
		b_ = gts[:, 3] - anchors_cy

		left_right = torch.stack([l_, r_], dim=1)
		top_bottom = torch.stack([t_, b_], dim=1)
		centerness = torch.sqrt(
			(left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) *
			(top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0]))
		assert not torch.isnan(centerness).any()
		return centerness

	@force_fp32(apply_to=('cls_scores', 'bbox_preds', 'centernesses'))
	def get_bboxes(self,
				   cls_scores,
				   bbox_preds,
				   centernesses,
				   img_metas,
				   cfg,
				   rescale=False):

		assert len(cls_scores) == len(bbox_preds)
		num_levels = len(cls_scores)
		device = cls_scores[0].device
		mlvl_anchors = [
			self.anchor_generators[i].grid_anchors(
				cls_scores[i].size()[-2:],
				self.anchor_strides[i],
				device=device).type_as(bbox_preds[0]) for i in range(num_levels)
		]

		result_list = []
		for img_id in range(len(img_metas)):
			cls_score_list = [
				cls_scores[i][img_id].detach() for i in range(num_levels)
			]
			bbox_pred_list = [
				bbox_preds[i][img_id].detach() for i in range(num_levels)
			]
			centerness_pred_list = [
				centernesses[i][img_id].detach() for i in range(num_levels)
			]
			img_shape = img_metas[img_id]['img_shape']
			scale_factor = img_metas[img_id]['scale_factor']
			proposals = self.get_bboxes_single(cls_score_list, bbox_pred_list,
											   centerness_pred_list,
											   mlvl_anchors, img_shape,
											   scale_factor, cfg, rescale)
			result_list.append(proposals)
		return result_list

	def get_bboxes_single(self,
						  cls_scores,
						  bbox_preds,
						  centernesses,
						  mlvl_anchors,
						  img_shape,
						  scale_factor,
						  cfg,
						  rescale=False):
		assert len(cls_scores) == len(bbox_preds) == len(mlvl_anchors)
		mlvl_bboxes = []
		mlvl_scores = []
		mlvl_centerness = []
		for cls_score, bbox_pred, centerness, anchors in zip(
				cls_scores, bbox_preds, centernesses, mlvl_anchors):
			assert cls_score.size()[-2:] == bbox_pred.size()[-2:]

			scores = cls_score.permute(1, 2, 0).reshape(
				-1, self.cls_out_channels).sigmoid()
			bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
			centerness = centerness.permute(1, 2, 0).reshape(-1).sigmoid()

			nms_pre = cfg.get('nms_pre', -1)
			if nms_pre > 0 and scores.shape[0] > nms_pre:
				max_scores, _ = (scores * centerness[:, None]).max(dim=1)
				_, topk_inds = max_scores.topk(nms_pre)
				anchors = anchors[topk_inds, :]
				bbox_pred = bbox_pred[topk_inds, :]
				scores = scores[topk_inds, :]
				centerness = centerness[topk_inds]

			bboxes = delta2bbox(anchors, bbox_pred, self.target_means,
								self.target_stds, img_shape)
			mlvl_bboxes.append(bboxes)
			mlvl_scores.append(scores)
			mlvl_centerness.append(centerness)

		mlvl_bboxes = torch.cat(mlvl_bboxes)
		if rescale:
			mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)

		mlvl_scores = torch.cat(mlvl_scores)
		padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
		mlvl_scores = torch.cat([padding, mlvl_scores], dim=1)
		mlvl_centerness = torch.cat(mlvl_centerness)

		det_bboxes, det_labels = multiclass_nms(
			mlvl_bboxes,
			mlvl_scores,
			cfg.score_thr,
			cfg.nms,
			cfg.max_per_img,
			score_factors=mlvl_centerness)
		return det_bboxes, det_labels

	def atss_target(self,
					anchor_list,
					valid_flag_list,
					gt_bboxes_list,
					img_metas,
					cfg,
					gt_bboxes_ignore_list=None,
					gt_labels_list=None,
					label_channels=1,
					unmap_outputs=True):
		"""
		almost the same with anchor_target, with a little modification,
		here we need return the anchor
		"""
		num_imgs = len(img_metas)
		assert len(anchor_list) == len(valid_flag_list) == num_imgs

		# anchor number of multi levels
		num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
		num_level_anchors_list = [num_level_anchors] * num_imgs

		# concat all level anchors and flags to a single tensor
		for i in range(num_imgs):
			assert len(anchor_list[i]) == len(valid_flag_list[i])
			anchor_list[i] = torch.cat(anchor_list[i])
			valid_flag_list[i] = torch.cat(valid_flag_list[i])

		# compute targets for each image
		if gt_bboxes_ignore_list is None:
			gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
		if gt_labels_list is None:
			gt_labels_list = [None for _ in range(num_imgs)]
		(all_anchors, all_labels, all_label_weights, all_bbox_targets,
		 all_bbox_weights, pos_inds_list, neg_inds_list) = multi_apply(
			 self.atss_target_single,
			 anchor_list,
			 valid_flag_list,
			 num_level_anchors_list,
			 gt_bboxes_list,
			 gt_bboxes_ignore_list,
			 gt_labels_list,
			 img_metas,
			 cfg=cfg,
			 label_channels=label_channels,
			 unmap_outputs=unmap_outputs)
		# no valid anchors
		if any([labels is None for labels in all_labels]):
			return None
		# sampled anchors of all images
		num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
		num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
		# split targets to a list w.r.t. multiple levels
		anchors_list = images_to_levels(all_anchors, num_level_anchors)
		labels_list = images_to_levels(all_labels, num_level_anchors)
		label_weights_list = images_to_levels(all_label_weights,
											  num_level_anchors)
		bbox_targets_list = images_to_levels(all_bbox_targets,
											 num_level_anchors)
		bbox_weights_list = images_to_levels(all_bbox_weights,
											 num_level_anchors)
		return (anchors_list, labels_list, label_weights_list,
				bbox_targets_list, bbox_weights_list, num_total_pos,
				num_total_neg)

	def atss_target_single(self,
						   flat_anchors,
						   valid_flags,
						   num_level_anchors,
						   gt_bboxes,
						   gt_bboxes_ignore,
						   gt_labels,
						   img_meta,
						   cfg,
						   label_channels=1,
						   unmap_outputs=True):
		inside_flags = anchor_inside_flags(flat_anchors, valid_flags,
										   img_meta['img_shape'][:2],
										   cfg.allowed_border)
		if not inside_flags.any():
			return (None, ) * 6
		# assign gt and sample anchors
		anchors = flat_anchors[inside_flags.type(torch.bool), :]

		num_level_anchors_inside = self.get_num_level_anchors_inside(
			num_level_anchors, inside_flags)
		bbox_assigner = build_assigner(cfg.assigner)
		assign_result = bbox_assigner.assign(anchors, num_level_anchors_inside,
											 gt_bboxes, gt_bboxes_ignore,
											 gt_labels)

		bbox_sampler = PseudoSampler()
		sampling_result = bbox_sampler.sample(assign_result, anchors,
											  gt_bboxes)

		num_valid_anchors = anchors.shape[0]
		bbox_targets = torch.zeros_like(anchors)
		bbox_weights = torch.zeros_like(anchors)
		labels = anchors.new_zeros(num_valid_anchors, dtype=torch.long)
		label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)

		pos_inds = sampling_result.pos_inds
		neg_inds = sampling_result.neg_inds
		if len(pos_inds) > 0:
			pos_bbox_targets = bbox2delta(sampling_result.pos_bboxes,
										  sampling_result.pos_gt_bboxes,
										  self.target_means, self.target_stds)
			bbox_targets[pos_inds, :] = pos_bbox_targets
			bbox_weights[pos_inds, :] = 1.0
			if gt_labels is None:
				labels[pos_inds] = 1
			else:
				labels[pos_inds] = gt_labels[
					sampling_result.pos_assigned_gt_inds]
			if cfg.pos_weight <= 0:
				label_weights[pos_inds] = 1.0
			else:
				label_weights[pos_inds] = cfg.pos_weight
		if len(neg_inds) > 0:
			label_weights[neg_inds] = 1.0

		# map up to original set of anchors
		if unmap_outputs:
			inside_flags = inside_flags.type(torch.bool)
			num_total_anchors = flat_anchors.size(0)
			anchors = unmap(anchors, num_total_anchors, inside_flags)
			labels = unmap(labels, num_total_anchors, inside_flags)
			label_weights = unmap(label_weights, num_total_anchors,
								  inside_flags)
			bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
			bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)

		return (anchors, labels, label_weights, bbox_targets, bbox_weights,
				pos_inds, neg_inds)

	def get_num_level_anchors_inside(self, num_level_anchors, inside_flags):
		split_inside_flags = torch.split(inside_flags, num_level_anchors)
		num_level_anchors_inside = [
			int(flags.sum()) for flags in split_inside_flags
		]
		return num_level_anchors_inside
