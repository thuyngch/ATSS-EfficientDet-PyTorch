import os, cv2
import numpy as np
from glob import glob
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import torch, mmcv
from mmcv.parallel import collate, scatter
from mmdet.datasets.pipelines import Compose
from mmdet.datasets.coco import CocoDataset as DATASET
from mmdet.apis.inference import init_detector, show_result, LoadImage


def inference_detector(model, img):
    cfg = model.cfg
    device = next(model.parameters()).device
    test_pipeline = [LoadImage()] + cfg.test_pipeline[1:]
    test_pipeline = Compose(test_pipeline)
    data = dict(img=img)
    data = test_pipeline(data)
    data = scatter(collate([data], samples_per_gpu=1), [device])[0]
    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **data)
    return result


if __name__ == "__main__":
	# Define
	threshold = 0.3
	config = "configs/effdet/atss_effdet_d0.py"
	checkpoint = "work_dirs/atss_effdet_d0/atss_effdet_d0.pth"

	out_dir = "demo/atss_effdet_d0"
	img_files = sorted(glob("demo/images/*.jpg"))
	# img_files = sorted(glob("datasets/coco/images/val2017/*.jpg"))[:10]

	# Setup
	model = init_detector(config, checkpoint=checkpoint, device='cuda')
	CLASSES = list(DATASET.CLASSES)
	os.makedirs(out_dir, exist_ok=True)

	# Inference
	for img_file in img_files:
		result = inference_detector(model, img_file)
		out_file = os.path.join(out_dir, '{}.jpg'.format(os.path.basename(img_file).split('.')[0]))
		show_result(img_file, result, CLASSES, score_thr=threshold, show=False, out_file=out_file)
		print("Result is saved at {}".format(out_file))
