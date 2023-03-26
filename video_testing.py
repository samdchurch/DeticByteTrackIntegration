import colorsys
import math
import os
import pickle

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
import tqdm

from BYTE.byte_tracker import Tracker, NearestNeighborDistanceMetric
from BYTE.detection import Detection
from parse_video import VideoReader, DETIC, setup_detic_cfg

model_cfg = 'configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml'
model_weights = 'models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth'
dictionary = 'lvis'

def HSVToRGB(h, s, v):
    (r, g, b) = colorsys.hsv_to_rgb(h, s, v)
    return (int(255 * r), int(255 * g), int(255 * b))


def getDistinctColors(n):
    huePartition = 1.0 / (n + 1)
    return [HSVToRGB(huePartition * value, 1.0, 1.0) for value in range(0, n)]


def make_tracking_video(video_fn, save_folder, frame_max_size, frame_rate, tracks):
    vcap = cv2.VideoCapture(video_fn)
    success, frame = vcap.read()
    original_fps = vcap.get(cv2.CAP_PROP_FPS)
    skip = max(int(original_fps // frame_rate), 0)

    orig_width = vcap.get(cv2.CAP_PROP_FRAME_WIDTH)
    orig_height = vcap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    max_dim = max(orig_width, orig_height)
    width = int(frame.shape[1] * frame_max_size / max_dim)
    height = int(frame.shape[0] * frame_max_size / max_dim)

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    out_video = cv2.VideoWriter(save_folder + '/' + video_fn[:-4] + '_wfeat_tracking.mp4',
                    cv2.VideoWriter_fourcc(*'mp4v'), original_fps, (width, height))

    colors = getDistinctColors(100)


    frame_num = 0
    while vcap.isOpened():
        success, frame = vcap.read()

        if success:
            frame = cv2.resize(frame, (width, height))
            if frame_num % skip == 0:
                frame_tracks = tracks[int(frame_num / skip)]
                if len(frame_tracks) > 0:
                    for t in frame_tracks:
                        cv2.rectangle(frame, (t[0], t[1]), (t[2], t[3]), colors[(t[4] * 10) % len(colors)], 1)

            else:
                prev_index = math.floor(frame_num / skip)
                next_index = math.ceil(frame_num / skip)

                if prev_index != 0 and next_index < len(tracks):
                    weight = (frame_num % skip) / skip

                    prev_tracks = tracks[prev_index]
                    next_tracks = tracks[next_index]

                    for t_prev in prev_tracks:
                        track_id = t_prev[4]
                        for t_next in next_tracks:
                            if t_next[4] == track_id:
                                loc = (1-weight)*t_prev[:4] + weight*t_next[:4]
                                loc = np.asarray(loc, dtype=int)
                                cv2.rectangle(frame, (loc[0], loc[1]), (loc[2], loc[3]),
                                              colors[(track_id * 10) % len(colors)], 1)

            out_video.write(frame)
            frame_num += 1
        else:
            break

    out_video.release()


def make_predictions(video_loader, video, model, filename, batch_size, save_predictions):
    results = []
    for batch, timestamps in tqdm.tqdm(video_loader, total=video.num_frames // batch_size):
        batch_results = model.predictor(batch.numpy(), return_feats=False)

        results += batch_results

    if not save_predictions:
        return results
    else:
        results_fn = filename[:-4] + '_predictions.txt'
        pickle.dump(results, open(results_fn, 'wb'))
        return results_fn


def run_tracking(prediction_results, video_loader, video, model, batch_size):
    batch_index_begin = 0
    metric = NearestNeighborDistanceMetric(metric='cosine', matching_threshold=.7)
    tracker = Tracker(metric=metric, n_init=1)

    tracks = []

    for batch, timestamps in tqdm.tqdm(video_loader, total=video.num_frames // batch_size):
        batch_index_end = min(batch_index_begin + batch_size, len(prediction_results))
        batch_results = prediction_results[batch_index_begin:batch_index_end]
        batch_index_begin += batch_size

        model.predictor.get_feature_vectors(batch.numpy(), batch_results)

        for frame in batch_results:
            detections = []
            for i in range(len(frame['instances'].scores)):
                box = frame['instances'].pred_boxes[i].tensor.cpu()[0]

                tlwh = [box[0], box[1], box[2] - box[0], box[3] - box[1]]
                det = Detection(tlwh=tlwh, confidence=frame['instances'].scores[i],
                              feature=frame['instances'].feat[i].flatten())
                # det = Detection(tlwh=tlwh, confidence=frame['instances'].scores[i],
                #                 feature=torch.Tensor([1, 1]))

                detections.append(det)

            tracker.predict()
            tracker.update(detections, frame['instances'].pred_classes)

            tracks.append(tracker.get_current_tracks())

    return tracks


def run_videos(video_fns, save_folder, batch_size, frame_rate, frame_max_size, confidence_thr):

    opts = ['MODEL.WEIGHTS', model_weights, 'MODEL.MASK_ON', False]
    if frame_max_size is not None:
        opts.extend(['INPUT.MIN_SIZE_TEST', frame_max_size, 'INPUT.MAX_SIZE_TEST', frame_max_size])
    cfg = setup_detic_cfg(
        config_file=model_cfg,
        opts=opts,
        confidence_threshold=confidence_thr
    )
    model = DETIC(cfg, dictionary)

    for video_fn in video_fns:
        video = VideoReader(video_fn, frame_rate=frame_rate, max_size=frame_max_size)
        video_loader = DataLoader(video, batch_size=batch_size, num_workers=1)


        print('Num frames:', video.num_frames)

        results = make_predictions(video_loader=video_loader, video=video, model=model, filename=video_fn,
                                   batch_size=batch_size, save_predictions=False)

        tracks = run_tracking(prediction_results=results, video_loader=video_loader, video=video,
                              model=model, batch_size=batch_size)

        make_tracking_video(video_fn, save_folder, frame_max_size, frame_rate, tracks)



if __name__ == '__main__':
    video_fns = ['kittens_short.mp4']

    run_videos(video_fns, save_folder='output', batch_size=64, frame_rate=8, frame_max_size=640, confidence_thr=0.7)
