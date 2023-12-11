from thor_exp.utils import get_cmat
from ai2thor.controller import Controller
from myutils import get_transforms_nav, transforms2actions
from myutils import get_flow_model, pred_flow_frame
from flowdiffusion.inference_utils import get_video_model_thor, pred_video_thor
import random
import torch
import imageio
import numpy as np
import os
from argparse import ArgumentParser
import json
from tqdm import tqdm
import cv2

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

class ThorEnv():
    def __init__(self, scene, target, seed=np.random.randint(1e6), resolution=(64, 64), max_eplen=50):
        self.controller = Controller(
            scene=scene, 
            rotateStepDegrees=45, 
            visibilityDistance=1.5,
            width=resolution[0],
            height=resolution[1],
            renderDepthImage=True,
        )
        self.max_eplen = max_eplen
        self.target = target
        self.seed(seed)

    def seed(self, seed):
        self.rng = np.random.RandomState(seed)
        
    def reset(self):
        self.eplen = 0

        self._randomize_agent_pose()
        for i, obj in enumerate(self.controller.last_event.metadata["objects"]):
            if self.target in obj["name"]:
                objidx = i
                break
        self.objidx = objidx

        return self._get_obs()

    def step(self, action):
        assert action in ["MoveAhead", "RotateRight", "RotateLeft", "Done"]
        self.eplen += 1
        self.controller.step(action)
        done = self.eplen >= self.max_eplen or action == "Done"
        success = self._success()
        obs = self._get_obs()
        return obs, success, done

    def _get_obs(self):
        frame = self.controller.last_event.frame
        depth = self.controller.last_event.depth_frame
        return (frame, depth)

    def _randomize_agent_pose(self):
        self.controller.reset()
        positions = self.controller.step(
            action="GetReachablePositions"
        ).metadata["actionReturn"]
        print(len(positions))
        position = self.rng.choice(positions)
        # rotation = random.choice([0, 90, 180, 270])
        self.controller.step(
            "Teleport",
            position=position,
            # rotation=dict(x=0, y=rotation, z=0),
        )

    def _success(self):
        objstate = self.controller.last_event.metadata["objects"][self.objidx]
        return objstate["visible"] 

def draw_arrow(frames, actions):
    start_point = (31, 63)
    images = []
    for i in range(len(actions)):
        action = actions[i]
        if action == "RotateLeft":
            end_point = (0, 47)
        elif action == "RotateRight":
            end_point = (47, 47)
        elif action == "MoveAhead":
            end_point = (31, 47)
        else:
            end_point = (31, 63)
        image = (frames[i].transpose(1, 2, 0)).copy()
        image = cv2.arrowedLine(image, start_point, end_point, (0, 0, 255), 2)
        images.append(image)

    return images

def log_func(scene, target, seed, plans, flow_colors, tfms, actions, frames, depths, logdir="./thor_vis"):
    savedir = os.path.join(logdir, scene, target, str(seed).zfill(2))
    os.makedirs(savedir, exist_ok=True)
    assert len(plans) == len(flow_colors) == len(tfms) == len(actions)
    for segid in range(len(plans)):
        prefix = os.path.join(savedir, f"plan_segments", str(segid).zfill(2))
        os.makedirs(os.path.join(prefix, "first_frame"), exist_ok=True)
        os.makedirs(os.path.join(prefix, "plan"), exist_ok=True)
        for i, frame in enumerate(plans[segid]):
            if i == 0:
                imageio.imwrite(os.path.join(prefix, "first_frame", f"first_frame.png"), frame.transpose(1, 2, 0))
            imageio.imwrite(os.path.join(prefix, "plan", f"{str(i).zfill(2)}.png"), frame.transpose(1, 2, 0))
        os.makedirs(os.path.join(prefix, "flows"), exist_ok=True)
        for i, flow_color in enumerate(flow_colors[segid]):
            imageio.imwrite(os.path.join(prefix, "flows", f"{str(i).zfill(2)}.png"), flow_color)

        actions_vis = draw_arrow(plans[segid], actions[segid])
        os.makedirs(os.path.join(prefix, "actions_vis"), exist_ok=True)
        for i, action_vis in enumerate(actions_vis):
            imageio.imwrite(os.path.join(prefix, "actions_vis", f"{str(i).zfill(2)}.png"), action_vis)

        with open(os.path.join(prefix, "infos.json"), "w") as f:
            json.dump({
                "text": target,
                "transform_params": tfms[segid].tolist(),
                "actions": actions[segid],
            }, f, indent=4)

    os.makedirs(os.path.join(savedir, "execution_results", "raw_frames"), exist_ok=True)
    for i, frame in enumerate(frames):
        imageio.imwrite(os.path.join(savedir, "execution_results", "raw_frames", f"{str(i).zfill(2)}.png"), frame)
    imageio.mimsave(os.path.join(savedir, "execution_results", "video.mp4"), frames, fps=5)

scene2targets = {
    "FloorPlan1":  ["Toaster", "Spatula", "Bread"],
    "FloorPlan201": ["Painting", "Laptop", "Television"],
    "FloorPlan301": ["Blinds", "DeskLamp", "Pillow"],
    "FloorPlan401": ["Mirror", "ToiletPaper", "SoapBar"],
}

def eval(scene, target, n_seeds=20, log=True):
    video_model = get_video_model_thor(ckpts_dir='../ckpts/ithor', milestone=16)
    flow_model = get_flow_model()

    env = ThorEnv(scene, target)
    if log:
        render_env = ThorEnv(scene, target, resolution=(512, 512))
    successes = 0
    framess = []
    for seed in tqdm(range(n_seeds)):
        # try:
            env.seed(seed)
            if log:
                render_env.seed(seed)
            cmat = get_cmat()[:3]
            frame, depth = env.reset()
            if log:
                render_frame, _ = render_env.reset()
                frames = [render_frame]
            else:
                frames = [frame]
            plans = []
            flows = []
            actionss = []
            transformss = []
            depths = [depth]
            while True:
                vidplan = pred_video_thor(video_model, frame, target)
                image1, image2, color, flow, flow_b = pred_flow_frame(flow_model, vidplan)

                transforms = get_transforms_nav(depth, cmat, flow, rgd_tfm_tries=8)
                actions = transforms2actions(transforms, verbose=False)
                actionss.append(actions)
                plans.append(vidplan[:len(actions)+1])
                flows.append(color[:len(actions)])
                transformss.append(transforms[:len(actions)])
                for action in actions:
                    (frame, depth), success, done = env.step(action)
                    depths.append(depth)
                    if log:
                        (render_frame, _), _, _ = render_env.step(action)
                        frames.append(render_frame)
                    else:
                        frames.append(frame)
                    if done:
                        break
                if done:
                    break
            successes += success

            if success and log:
                log_func(scene, target, seed, plans, flows, transformss, actionss, frames, depths)
            
            framess.append(frames)

    success_rate = successes / n_seeds
    return success_rate, framess

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--scene", type=str, default="FloorPlan201")
    parser.add_argument("--target", type=str, default="Television")
    parser.add_argument("--n_seeds", type=int, default=20)
    parser.add_argument("--log", action="store_true")
    args = parser.parse_args()
    scene = args.scene
    target = args.target
    n_seeds = args.n_seeds
    log = args.log
    savedir = f"../results/results_AVDC_thor/{scene}/{target}"
    if os.path.exists(savedir):
        print(f"Skipping {scene} {target}")
    else: 
        success_rate, framess = eval(scene, target, n_seeds=n_seeds, log=log)
        print(f"{scene} {target}: {success_rate}")
        os.makedirs(os.path.join(savedir, "records"), exist_ok=True)
        for i, frames in enumerate(framess):
            imageio.mimsave(os.path.join(savedir, "records", f"{i}.mp4"), frames, fps=5)
        with open(f"{savedir}/result.json", "w") as f:
            json.dump({"success_rate": success_rate}, f)
    

