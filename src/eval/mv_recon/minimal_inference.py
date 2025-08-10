import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import time
import torch
import argparse
import numpy as np
# import open3d as o3d
import os.path as osp
from torch.utils.data import DataLoader
from add_ckpt_path import add_path_to_dust3r
from accelerate import Accelerator
from torch.utils.data._utils.collate import default_collate
import tempfile
from tqdm import tqdm
import uuid
import json
from collections import defaultdict

def create_fake_frames(num_frames=4, img_channels=3, img_height=518, img_width=392):
    images = torch.zeros((num_frames, img_channels, img_height, img_width), dtype=torch.bfloat16).to("cuda")
    
    frames = []
    for i in range(images.shape[0]):
        image = images[i].unsqueeze(0)
        frame = {
            "img": image 
        }
        frames.append(frame)
    
    return frames

def get_args_parser():
    parser = argparse.ArgumentParser("3D Reconstruction evaluation", add_help=False)
    parser.add_argument(
        "--weights",
        type=str,
        default="",
        help="ckpt name",
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="device")
    parser.add_argument("--model_name", type=str, default="")
    parser.add_argument(
        "--conf_thresh", type=float, default=0.0, help="confidence threshold"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="value for outdir",
    )
    parser.add_argument("--size", type=int, default=518)
    parser.add_argument("--revisit", type=int, default=1, help="revisit times")
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--use_proj", action="store_true")
    return parser


def main(args):
    add_path_to_dust3r(args.weights)
    #from eval.mv_recon.data import SevenScenes, NRGBD
    #from eval.mv_recon.utils import accuracy, completion

    if args.size == 512:
        resolution = (512, 384)
    elif args.size == 224:
        resolution = 224
    elif args.size == 518:
        resolution = (518, 392)
        # resolution = (518, 336)
    else:
        raise NotImplementedError

    accelerator = Accelerator()
    device = accelerator.device
    model_name = args.model_name
    if model_name == "StreamVGGT":
        from streamvggt.models.streamvggt import StreamVGGT
        from streamvggt.utils.pose_enc import pose_encoding_to_extri_intri
        from streamvggt.utils.geometry import unproject_depth_map_to_point_map
        from eval.mv_recon.criterion import Regr3D_t_ScaleShiftInv, L21
        from dust3r.utils.geometry import geotrf
        from copy import deepcopy
        model = StreamVGGT()
        ckpt = torch.load(args.weights, map_location=device)
        model.load_state_dict(ckpt, strict=True)
        model.eval()
        model = model.to("cuda").to(torch.bfloat16)
    del ckpt

    batch = create_fake_frames()
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype = torch.bfloat16):
            with torch.no_grad():
                print("#########", batch[0]["img"].dtype)
                results = model.export_memory(batch)

            # print("################")

            # preds, batch = results.ress, results.views 

            # if args.use_proj:
            #     pose_enc = torch.stack([preds[s]["camera_pose"] for s in range(len(preds))], dim=1)
            #     depth_map = torch.stack([preds[s]["depth"] for s in range(len(preds))], dim=1)
            #     depth_conf = torch.stack([preds[s]["depth_conf"] for s in range(len(preds))], dim=1)
            #     extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc,
            #                                                             batch[0]["img"].shape[-2:])




from collections import defaultdict
import re

pattern = r"""
    Idx:\s*(?P<scene_id>[^,]+),\s*
    Acc:\s*(?P<acc>[^,]+),\s*
    Comp:\s*(?P<comp>[^,]+),\s*
    NC1:\s*(?P<nc1>[^,]+),\s*
    NC2:\s*(?P<nc2>[^,]+)\s*-\s*
    Acc_med:\s*(?P<acc_med>[^,]+),\s*
    Compc_med:\s*(?P<comp_med>[^,]+),\s*
    NC1c_med:\s*(?P<nc1_med>[^,]+),\s*
    NC2c_med:\s*(?P<nc2_med>[^,]+)
"""

regex = re.compile(pattern, re.VERBOSE)


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()

    main(args)

                    # if model_name == "stream3r" or "VGGT":
                    #     revisit = args.revisit
                    #     update = not args.freeze
                    #     if revisit > 1:
                    #         # repeat input for 'revisit' times
                    #         new_views = []
                    #         for r in range(revisit):
                    #             for i in range(len(batch)):
                    #                 new_view = deepcopy(batch[i])
                    #                 new_view["idx"] = [
                    #                     (r * len(batch) + i)
                    #                     for _ in range(len(batch[i]["idx"]))
                    #                 ]
                    #                 new_view["instance"] = [
                    #                     str(r * len(batch) + i)
                    #                     for _ in range(len(batch[i]["instance"]))
                    #                 ]
                    #                 if r > 0:
                    #                     if not update:
                    #                         new_view["update"] = torch.zeros_like(
                    #                             batch[i]["update"]
                    #                         ).bool()
                    #                 new_views.append(new_view)
                    #         batch = new_views
                    #     dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
                    #     with torch.cuda.amp.autocast(dtype=dtype):
                    #         if isinstance(batch, dict) and "img" in batch:
                    #             batch["img"] = (batch["img"] + 1.0) / 2.0
                    #         elif isinstance(batch, list) and all(isinstance(v, dict) and "img" in v for v in batch):
                    #             for view in batch:
                    #                 view["img"] = (view["img"] + 1.0) / 2.0
