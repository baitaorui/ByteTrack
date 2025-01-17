import argparse
import os
import os.path as osp
import time
from turtle import down
import cv2
import numpy as np
import torch

from loguru import logger

from yolox.data.data_augment import preproc
from yolox.exp import get_exp
from yolox.utils import fuse_model
from yolox.utils.visualize import plot_tracking2
from yolox.tracker.byte_tracker import BYTETracker
from yolox.sort_tracker.sort import Sort
from tracker.newByteTrack import NewByteTrack
from mmdet.apis import inference_detector, init_detector
from tools.utils import CLASSES
from tools.utils import get_polygon, judge, plot_text


IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]

def make_parser():
    parser = argparse.ArgumentParser("ByteTrack Demo!")
    # parser.add_argument(
    #     "demo", default="video", help="demo type, eg. image, video and webcam"
    # )
    parser.add_argument(
        "config", help="mmdet yolox config file path"
    )
    parser.add_argument(
        "checkpoint", help="mmdet yolox ckpt file path"
    )
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    parser.add_argument(
        #"--path", default="./datasets/mot/train/MOT17-05-FRCNN/img1", help="path to images or video"
        "--path", default="./videos/palace.mp4", help="path to images or video"
    )
    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
    parser.add_argument(
        "--save_result",
        action="store_true",
        help="whether to save the inference result of image/video",
    )

    # exp file
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="pls input your expriment description file",
    )
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument(
        "--device",
        default="gpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    parser.add_argument("--conf", default=None, type=float, help="test conf")
    parser.add_argument("--nms", default=None, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument("--fps", default=30, type=int, help="frame rate (fps)")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    # tracking args
    parser.add_argument("--track_thresh", type=float, default=0.1, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=150, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.6, help="matching threshold for tracking")
    parser.add_argument(
        "--aspect_ratio_thresh", type=float, default=1.6,
        help="threshold for filtering out boxes of which aspect ratio are above the given value."
    )
    parser.add_argument('--min_box_area', type=float, default=15, help='filter out tiny boxes')
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")
    return parser


def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = osp.join(maindir, filename)
            ext = osp.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names


def write_results(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n'
    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids, scores in results:
            for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(frame=frame_id, id=track_id, x1=round(x1, 1), y1=round(y1, 1), w=round(w, 1), h=round(h, 1), s=round(score, 2))
                f.write(line)
    logger.info('save results to {}'.format(filename))


class Predictor(object):
    def __init__(
        self,
        model,
        exp,
        device=torch.device("cpu"),
        fp16=False, 
    ):
        self.model = model
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16

    def inference(self, img):
        ii = img
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = osp.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        img, ratio = preproc(img, self.test_size)
        img_info["ratio"] = ratio
        img = torch.from_numpy(img).unsqueeze(0).float().to(self.device)
        if self.fp16:
            img = img.half()  # to FP16

        outputs = inference_detector(self.model, ii)
        # outputs = post_pro(outputs)
        return outputs, img_info

def post_pro(outputs):
    res = []
    for i in range(0, len(outputs)):
        boxes = np.array(outputs[i])
        if boxes.shape[0] == 0:
            continue
        for box in boxes:
            box = np.append(box, i)
            res.append(box)
    if len(res) == 0:
        return [None]
    return [np.array(res)]

def imageflow_demo(predictor, vis_folder, current_time, args):
    videos = os.listdir(args.path)
    for video in videos:
        if ".mp4" not in video:
             continue
        cap = cv2.VideoCapture(args.path + video)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
        fps = cap.get(cv2.CAP_PROP_FPS)
        timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
        save_folder = osp.join(vis_folder, timestamp)
        os.makedirs(save_folder, exist_ok=True)
        os.makedirs(save_folder + "/result", exist_ok=True)
        save_path = osp.join(save_folder, video)
        logger.info(f"video save_path is {save_path}")
        # vid_writer = cv2.VideoWriter(
        #     save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
        # )
        mask, colo_img = get_polygon(width, height)
        trackers = [BYTETracker(args, frame_rate=30) for i in range(0,len(CLASSES))]
        # trackers = [Sort(args) for i in range(0,len(CLASSES))]
        frame_id = 0
        results = []
        # list 与蓝色polygon重叠
        list_overlapping_blue_polygon = [[] for i in range(0,len(CLASSES))]
        # list 与黄色polygon重叠
        list_overlapping_yellow_polygon = [[] for i in range(0,len(CLASSES))]
        # 进入数量
        down_count = [0 for i in range(0,len(CLASSES))]
        # 离开数量
        up_count = [0 for i in range(0,len(CLASSES))]

        while True:
            if frame_id % 20 == 0:
                logger.info('Processing frame {}'.format(frame_id))
            ret_val, frame = cap.read()
            if ret_val:
                outputs, img_info = predictor.inference(frame)
                online_im = img_info['raw_img']
                # online_im = cv2.add(online_im, colo_img)
                for i in range(0, len(outputs)):
                    if outputs[i] is not None and len(outputs[i] > 0):
                        online_targets = trackers[i].update2(outputs[i], [img_info['height'], img_info['width']], exp.test_size)
                        online_tlwhs = []
                        online_ids = []
                        online_scores = []
                        for t in online_targets:
                            tlwh = t.tlwh
                            tid = t.track_id
                            if tlwh[2] * tlwh[3] > args.min_box_area:
                                online_tlwhs.append(tlwh)
                                online_ids.append(tid)
                                online_scores.append(t.score)
                            # results.append(
                            #     f"{frame_id},{i},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                            # )
                            results.append(
                                f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},{i},-1\n"
                            )
                        # online_im = plot_tracking2(online_im, online_tlwhs, online_ids, frame_id=frame_id + 1)
                        # list_overlapping_blue_polygon[i], list_overlapping_yellow_polygon[i], down_count[i], up_count[i] = judge(
                        #     online_targets, list_overlapping_blue_polygon[i], list_overlapping_yellow_polygon[i], down_count[i], up_count[i], i, mask
                        #     )
                    # else :
                        # online_im = plot_tracking2(online_im, [], [],  frame_id=frame_id + 1)
                # online_im = plot_text(online_im, down_count)
                # if args.save_result:
                #     vid_writer.write(online_im)
                ch = cv2.waitKey(1)
                if ch == 27 or ch == ord("q") or ch == ord("Q"):
                    break
            else:
                break
            frame_id += 1
        if args.save_result:
            # res_file = osp.join(vis_folder, f"{timestamp + video}.txt")
            # with open(res_file, 'w') as f:
            #     f.writelines(results)
            # logger.info(f"save results to {res_file}")
            res_file = osp.join(vis_folder + "/" + timestamp + "/result" , f"{video.split('.')[0]}.txt")
            with open(res_file, 'w') as f:
                f.writelines(results)
            logger.info(f"save results to {res_file}")
    trackers = []

def main(exp, args):
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    output_dir = osp.join(exp.output_dir, args.experiment_name)
    os.makedirs(output_dir, exist_ok=True)

    if args.save_result:
        vis_folder = osp.join(output_dir, "track_vis")
        os.makedirs(vis_folder, exist_ok=True)

    args.device = torch.device("cuda" if args.device == "gpu" else "cpu")

    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = init_detector(config=args.config, checkpoint=args.checkpoint)
    model.eval()

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    if args.fp16:
        model = model.half()  # to FP16

    predictor = Predictor(model, exp, args.device, args.fp16)
    current_time = time.localtime()
    imageflow_demo(predictor, vis_folder, current_time, args)


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)

    main(exp, args)
