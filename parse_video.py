# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import glob
import multiprocessing as mp
import numpy as np
import os
import tempfile
import time
import warnings
import cv2
import tqdm
import sys
import mss

import torch
from torch.utils.data import DataLoader, IterableDataset

from detectron2.config import get_cfg
from detectron2.engine.defaults import DefaultPredictor

from BYTE.byte_tracker import Tracker, NearestNeighborDistanceMetric
from BYTE.detection import Detection

sys.path.insert(0, 'third_party/CenterNet2/')
from centernet.config import add_centernet_config
from detic.config import add_detic_config

from detic.predictor import MetadataCatalog, get_clip_embeddings, reset_cls_test
from detic.predictor import BUILDIN_METADATA_PATH, BUILDIN_CLASSIFIER

import pickle

import colorsys


class VideoReader(IterableDataset):
    def __init__(self, video_fn, frame_rate=None, max_size=None):
        self.video_fn = video_fn
        container = cv2.VideoCapture(self.video_fn)
        self.num_frames = int(container.get(cv2.CAP_PROP_FRAME_COUNT))
        self.original_fps = container.get(cv2.CAP_PROP_FPS)
        self.skip = max(int(self.original_fps // frame_rate), 0)
        self.frame_rate = self.original_fps / self.skip
        self.max_size = max_size

    def _frame_from_video(self):
        container = cv2.VideoCapture(self.video_fn)
        frame_id = 0
        while container.isOpened():
            success, frame = container.read()
            ts = container.get(cv2.CAP_PROP_POS_MSEC) / 1000.

            if success:
                if frame_id % self.skip == 0:   # Skip frames to match desired framerate
                    # Resize
                    if self.max_size is not None:
                        max_size = max(frame.shape[0], frame.shape[1])
                        width = int(frame.shape[1] * self.max_size/max_size)
                        height = int(frame.shape[0] * self.max_size/max_size)
                        frame = cv2.resize(frame, (width, height))

                    yield frame, ts
                frame_id += 1
            else:
                break

    def __iter__(self):
        return self._frame_from_video()

    def __len__(self):
        return self.num_frames


class BatchPredictor(DefaultPredictor):
    def __init__(self, cfg):
        super().__init__(cfg)

    @torch.no_grad()
    def __call__(self, batch, return_feats=False):
        # Apply pre-processing to image.
        single_img = batch.ndim == 3
        if single_img:
            batch = batch[None]

        # whether the model expects BGR inputs or RGB
        if self.input_format == "RGB":
            batch = batch[:, :, :, ::-1]

        height, width = batch[0].shape[:2]
        transform = self.aug.get_transform(batch[0])
        batch = [transform.apply_image(img) for img in batch]
        batch = [torch.as_tensor(img.astype("float32").transpose(2, 0, 1)) for img in batch]
        inputs = [{"image": img, "height": height, "width": width} for img in batch]
        preds = self.model.inference(inputs, return_feats=return_feats)
        return preds[0] if single_img else preds

    @torch.no_grad()
    def get_feature_vectors(self, batch, results):
        single_img = batch.ndim == 3
        if single_img:
            batch = batch[None]

        # whether the model expects BGR inputs or RGB
        if self.input_format == "RGB":
            batch = batch[:, :, :, ::-1]

        height, width = batch[0].shape[:2]
        transform = self.aug.get_transform(batch[0])
        batch = [transform.apply_image(img) for img in batch]
        batch = [torch.as_tensor(img.astype("float32").transpose(2, 0, 1)) for img in batch]
        inputs = [{"image": img, "height": height, "width": width} for img in batch]
        return self.model.get_features(inputs, results)


class DETIC(object):
    def __init__(self, detic_cfg, vocabulary, custom=None):
        """
        Args:
            cfg (CfgNode):
            instance_mode (ColorMode):
            parallel (bool): whether to run the model in different processes from visualization.
                Useful since the visualization logic can be slow.
        """
        if vocabulary == 'custom':
            self.metadata = MetadataCatalog.get("__unused")
            self.metadata.thing_classes = custom.split(',')
            classifier = get_clip_embeddings(self.metadata.thing_classes)
        else:
            self.metadata = MetadataCatalog.get(
                BUILDIN_METADATA_PATH[vocabulary])
            classifier = BUILDIN_CLASSIFIER[vocabulary]

        num_classes = len(self.metadata.thing_classes)
        self.cpu_device = torch.device("cpu")

        self.predictor = BatchPredictor(detic_cfg)
        reset_cls_test(self.predictor.model, classifier, num_classes)


def setup_detic_cfg(config_file, confidence_threshold, pred_all_class=True, opts=[], cpu=False):
    cfg = get_cfg()
    if cpu:
        cfg.MODEL.DEVICE="cpu"
    add_centernet_config(cfg)
    add_detic_config(cfg)
    cfg.merge_from_file(config_file)
    cfg.merge_from_list(opts)

    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = confidence_threshold
    cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = 'rand' # load later
    if not pred_all_class:
        cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = True
    cfg.freeze()
    return cfg



def tracking():
    video_fn = 'kittens_short.mp4'
    batch_size = 64
    frame_rate = 30
    frame_max_size = 640

    model_cfg = 'configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml'
    model_weights = 'models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth'
    confidence_thr = 0.5
    dictionary = 'lvis'

    opts = ['MODEL.WEIGHTS', model_weights, 'MODEL.MASK_ON', False]
    if frame_max_size is not None:
        opts.extend(['INPUT.MIN_SIZE_TEST', frame_max_size, 'INPUT.MAX_SIZE_TEST', frame_max_size])
    cfg = setup_detic_cfg(
        config_file=model_cfg,
        opts=opts,
        confidence_threshold=confidence_thr
    )
    model = DETIC(cfg, dictionary)
    video = VideoReader(video_fn, frame_rate=frame_rate, max_size=frame_max_size)
    video_loader = DataLoader(video, batch_size=batch_size, num_workers=1)

    metric = NearestNeighborDistanceMetric(metric='cosine', matching_threshold=.7)
    tracker = Tracker(metric=metric, n_init=1)

    colors = getDistinctColors(50)

    tracked_data = {}

    out_video = cv2.VideoWriter('tracking_videos/' + video_fn[:-4] + '_' + str(frame_rate) + '_' + str(confidence_thr) + '_tracks.mp4', cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, (640, 360))

    ts = time.time()
    iteration = 0
    for batch, timestamps in tqdm.tqdm(video_loader, total=video.num_frames // batch_size):
        t_data = time.time() - ts

        ts = time.time()
        results = pickle.load(open(video_fn[:-4] + '_results_' + str(iteration) + '.txt', 'rb'))
        iteration += 1

        counter = 0

        model.predictor.get_feature_vectors(batch.numpy(), results)

        tracked_data[iteration] = []

        for frame in results:

            img = batch[counter].numpy()
            detections = []
            for i in range(len(frame['instances'].scores)):
                box = frame['instances'].pred_boxes[i].tensor.cpu()[0]

                tlwh = [box[0], box[1], box[2] - box[0], box[3] - box[1]]
                det = Detection(tlwh=tlwh, confidence=frame['instances'].scores[i],
                                feature=frame['instances'].feat[i].flatten())

                detections.append(det)
                # box = box.numpy()
                # start_point = (int(box[0]), int(box[1]))
                # end_point = (int(box[2]), int(box[3]))
                # cv2.rectangle(img, start_point, end_point, colors[i], 1)

            tracker.predict()
            tracker.update(detections, frame['instances'].pred_classes)

            tracks = tracker.get_current_tracks()

            tracked_data[iteration].append(tracks)

            for t in tracks:
                cv2.rectangle(img, (t[0], t[1]), (t[2], t[3]), colors[(t[4] * 5) % len(colors)], 1)

            out_video.write(img)


            # cv2.imshow('frame ' + str(iteration), img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            counter += 1

        t_model = time.time() - ts

        t_proc = batch_size / video.frame_rate
        eff_min_per_hour = (t_data + t_model) / (t_proc / 3600) / 60
        print(f'Data: {t_data:.3f} Model: {t_model:.3f} Processing Time: {eff_min_per_hour:.1f} processing minutes per input hour', flush=True)

        ts = time.time()

    pickle.dump(tracked_data, open(video_fn[:-4] + 'tracked.txt', 'wb'))
    out_video.release()

    return


def prediction():
    video_fn = 'kittens_short.mp4'
    batch_size = 64
    frame_rate = 30
    frame_max_size = 640

    model_cfg = 'configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml'
    model_weights = 'models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth'
    confidence_thr = 0.5
    dictionary = 'lvis'

    opts = ['MODEL.WEIGHTS', model_weights, 'MODEL.MASK_ON', False]
    if frame_max_size is not None:
        opts.extend(['INPUT.MIN_SIZE_TEST', frame_max_size, 'INPUT.MAX_SIZE_TEST', frame_max_size])
    cfg = setup_detic_cfg(
        config_file=model_cfg,
        opts=opts,
        confidence_threshold=confidence_thr
    )
    model = DETIC(cfg, dictionary)
    video = VideoReader(video_fn, frame_rate=frame_rate, max_size=frame_max_size)
    video_loader = DataLoader(video, batch_size=batch_size, num_workers=1)

    ts = time.time()
    iteration = 0
    for batch, timestamps in tqdm.tqdm(video_loader, total=video.num_frames // batch_size):
        t_data = time.time() - ts

        ts = time.time()
        results = model.predictor(batch.numpy(), return_feats=False)
        pickle.dump(results, open(video_fn[:-4] + '_results_' + str(iteration) + '.txt', 'wb'))
        iteration += 1
        t_model = time.time() - ts

        t_proc = batch_size / video.frame_rate
        eff_min_per_hour = (t_data + t_model) / (t_proc / 3600) / 60
        print(
            f'Data: {t_data:.3f} Model: {t_model:.3f} Processing Time: {eff_min_per_hour:.1f} processing minutes per input hour',
            flush=True)

        ts = time.time()


        ### Accessing results
        # len(results) -> number frames in batch
        # len(results[i]['instances']) -> number bounding boxes in frame i
        # results[i]['instances'].pred_boxes -> coordinates of bounding boxes in frame i (Ki x 4 Tensor)
        # results[i]['instances'].scores -> objectness score of bounding boxes in frame i (Ki-dim Tensor)
        # results[i]['instances'].pred_classes -> index of predicted class for each bounding box in frame i (Ki-dim Tensor)
        # results[i]['instances'].feat -> ROIPool'ed feature for each bounding box in frame i (Ki x D x 7 x 7 Tensor)
        # model.metadata.thing_classes -> Class names


def HSVToRGB(h, s, v):
    (r, g, b) = colorsys.hsv_to_rgb(h, s, v)
    return (int(255 * r), int(255 * g), int(255 * b))


def getDistinctColors(n):
    huePartition = 1.0 / (n + 1)
    return [HSVToRGB(huePartition * value, 1.0, 1.0) for value in range(0, n)]

if __name__ == '__main__':
    tracking()
