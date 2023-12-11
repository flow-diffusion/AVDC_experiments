from mypolicy import MyPolicy_CL
from metaworld_exp.utils import get_seg, get_cmat, collect_video, sample_n_frames
import sys
sys.path.append('core')
import imageio.v2 as imageio
import numpy as np
from myutils import get_flow_model, pred_flow_frame, get_transforms, get_transformation_matrix
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE as env_dict
from metaworld import policies
from tqdm import tqdm
import cv2
import imageio
import json
import os
from flowdiffusion.inference_utils import get_video_model, pred_video
import random
import torch
from argparse import ArgumentParser

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

def get_task_text(env_name):
    name = " ".join(env_name.split('-')[:-3])
    return name

def get_policy(env_name):
    name = "".join(" ".join(get_task_text(env_name)).title().split(" "))
    policy_name = "Sawyer" + name + "V2Policy"
    try:
        policy = getattr(policies, policy_name)()
    except:
        policy = None
    return policy

with open("name2maskid.json", "r") as f:
    name2maskid = json.load(f)

def run(args):
    result_root = args.result_root
    os.makedirs(result_root, exist_ok=True)

    n_exps = args.n_exps
    resolution = (320, 240)
    cameras = ['corner', 'corner2', 'corner3']
    max_replans = 5


    video_model = get_video_model(ckpts_dir=args.ckpt_dir, milestone=args.milestone)
    flow_model = get_flow_model()

    try:
        with open(f"{result_root}/result_dict.json", "r") as f:
            result_dict = json.load(f)
    except:
        result_dict = {}


    env_name = args.env_name
    print(env_name)
    seg_ids = name2maskid[env_name]
    benchmark_env = env_dict[env_name]

    succes_rates = []
    reward_means = []
    reward_stds = []
    replans_counters = []
    for camera in cameras:
        success = 0
        rewards = []
        replans_counter = {i: 0 for i in range(max_replans + 1)}
        for seed in tqdm(range(n_exps)):
            try: 
                env = benchmark_env(seed=seed)
                
                obs = env.reset()
                policy = MyPolicy_CL(env, env_name, camera, video_model, flow_model, max_replans=max_replans)

                # os.makedirs(f'{result_root}/plans/{env_name}', exist_ok=True)
                # imageio.mimsave(f'{result_root}/plans/{env_name}/{camera}_{seed}.mp4', images.transpose(0, 2, 3, 1))

                images, _, episode_return = collect_video(obs, env, policy, camera_name=camera, resolution=resolution)
                rewards.append(episode_return / len(images))

                used_replans = max_replans - policy.replans
                
                ### save sample video
                os.makedirs(f'{result_root}/videos/{env_name}', exist_ok=True)
                imageio.mimsave(f'{result_root}/videos/{env_name}/{camera}_{seed}.mp4', images)
                
                print("test eplen: ", len(images))
                if len(images) <= 500:
                    success += 1
                    replans_counter[used_replans] += 1
                    print("success, used replans: ", used_replans)
            except Exception as e:
                print(e)
                print("something went wrong, skipping this seed")
                continue
        rewards = rewards + [0] * (n_exps - len(rewards))
        reward_means.append(np.mean(rewards))
        reward_stds.append(np.std(rewards))

        success_rate = success / n_exps
        succes_rates.append(success_rate)

        replans_counters.append(replans_counter)
                
    print(f"Success rates for {env_name}:\n", succes_rates)
    result_dict[env_name] = {
        "success_rates": succes_rates,
        "reward_means": reward_means,
        "reward_stds": reward_stds,
        "replans_counts": replans_counters
    }
    with open(f"{result_root}/result_dict.json", "w") as f:
        json.dump(result_dict, f, indent=4)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--env_name", type=str, default="door-open-v2-goal-observable")
    parser.add_argument("--n_exps", type=int, default=25)
    parser.add_argument("--ckpt_dir", type=str, default="../ckpts/metaworld")
    parser.add_argument("--milestone", type=int, default=24)
    parser.add_argument("--result_root", type=str, default="../results/results_AVDC_full")
    args = parser.parse_args()

    try:
        with open(f"{args.result_root}/result_dict.json", "r") as f:
            result_dict = json.load(f)
    except:
        result_dict = {}

    assert args.env_name in name2maskid.keys()
    if args.env_name in result_dict.keys():
        print("already done")
    else:
        run(args)
        