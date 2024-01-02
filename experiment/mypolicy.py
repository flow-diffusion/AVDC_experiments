from metaworld.policies.policy import Policy, assert_fully_parsed, move
from metaworld.policies.action import Action
import numpy as np
from metaworld_exp.utils import get_seg, get_cmat
import json
import cv2
from flowdiffusion.inference_utils import pred_video
from myutils import pred_flow_frame, get_transforms, get_transformation_matrix
import torch
from PIL import Image
from torchvision import transforms as T
import torch
import time
import pickle
import random

random.seed(1)
np.random.seed(1)
torch.manual_seed(1)

with open("name2maskid.json", "r") as f:
    name2maskid = json.load(f)

def log_time(time_vid, time_flow, time_action, n_replan, log_dir="logs"):
    with open(f"{log_dir}/time_vid_{n_replan}.txt", "a") as f:
        f.write(f"{time_vid}\n")
    with open(f"{log_dir}/time_flow_{n_replan}.txt", "a") as f:
        f.write(f"{time_flow}\n")
    with open(f"{log_dir}/time_action_{n_replan}.txt", "a") as f:
        f.write(f"{time_action}\n")

def log_time_execution(time_execution, n_replan, log_dir="logs"):
    with open(f"{log_dir}/time_execution_{n_replan}.txt", "a") as f:
        f.write(f"{time_execution}\n")

class ProxyPolicy(Policy):
    def __init__(self, env, proxy_model, camera, task, resolution=(320, 240)):
        self.env = env
        self.proxy_model = proxy_model
        self.camera = camera
        self.task = task
        self.last_pos = np.array([0, 0, 0])
        self.grasped = False
        with open("../text_embeds.pkl", "rb") as f:
            self.task2embed = pickle.load(f)
        with open("name2mode.json", "r") as f:
            name2mode = json.load(f)
        self.mode = name2mode[task]
        self.resolution = resolution
        self.transform = T.Compose([
            T.CenterCrop((128, 128)),
            T.ToTensor(),
        ])
        self.seg_ids = name2maskid[task]

        grasp, transforms = self.calculate_next_plan()

        subgoals = self.calc_subgoals(grasp, transforms)

        subgoals_np = np.array(subgoals)
        if self.mode == "push":
            subgoals = [s - np.array([0, 0, 0.03]) for s in subgoals]
        self.grasp = grasp
        self.subgoals = subgoals
        self.init_grasp()

    def calc_subgoals(self, grasp, transforms):
        print("Calculating subgoals...")
        subgoals = [grasp]
        for transforms in transforms:
            grasp_ext = np.concatenate([subgoals[-1], [1]])
            next_subgoal = (transforms @ grasp_ext)[:3]
            subgoals.append(next_subgoal)
        return subgoals

    def calculate_next_plan(self):
        image, depth = self.env.render(resolution=self.resolution, depth=True, camera_name=self.camera)
        cmat = get_cmat(self.env, self.camera, resolution=self.resolution)
        seg = get_seg(self.env, resolution=self.resolution, camera=self.camera, seg_ids=self.seg_ids)

        x = self.transform(Image.fromarray(image*np.expand_dims(seg, axis=2))).unsqueeze(0)
        # substract "-v2-goal-observable" from task string without rstip
        
        
        task_embed = torch.tensor(self.task2embed[self.task.split("-v2-goal-observable")[0]]).unsqueeze(0)
        flow = self.proxy_model(x, task_embed).squeeze(0).cpu().numpy()

        # make flow back to (320, 240), paste the (128, 128) flow to the center
        blank = np.zeros((2, 240, 320))
        blank[:, 56:184, 96:224] = flow
        flow = blank * 133.5560760498047 ## flow_abs_max=133.5560760498047
        flow = [flow.transpose(1, 2, 0)]
        

        # measure time for action planning
        grasp, transforms, center_2ds, sampless = get_transforms(seg, depth, cmat, flow)
        transform_mats = [get_transformation_matrix(*transform) for transform in transforms]

        return grasp[0], transform_mats

    @staticmethod
    @assert_fully_parsed
    def _parse_obs(obs):
        return {
            'hand_pos': obs[:3],
            'unused_info': obs[3:],
        }
        
    def init_grasp(self):
        self.grasped = False
        if self.mode == "push":
            for subgoal in self.subgoals:
                norm = np.linalg.norm(subgoal[:2] - self.grasp[:2])
                direction = subgoal[:2] - self.grasp[:2]
                direction = direction / norm
                if norm > 0.1:
                    break
            self.grasp[:2] = self.grasp[:2] - direction * 0.08

    def get_action(self, obs):
        o_d = self._parse_obs(obs)
        # if stucked (not moving), step the countdown
        self.last_pos = o_d['hand_pos']

        action = Action({
            'delta_pos': np.arange(3),
            'grab_effort': 3
        })

        action['delta_pos'] = move(o_d['hand_pos'], to_xyz=self._desired_pos(o_d), p=20.)
        action['grab_effort'] = self._grab_effort(o_d)

        return action.array

    def _desired_pos(self, o_d):
        pos_curr = o_d['hand_pos']
        move_precision = 0.12 if self.mode == "push" else 0.04

        # place end effector above object
        if not self.grasped and np.linalg.norm(pos_curr[:2] - self.grasp[:2]) > 0.02:
            return self.grasp + np.array([0., 0., 0.2])
        # drop end effector down on top of object
        elif not self.grasped and np.linalg.norm(pos_curr[2] - self.grasp[2]) > 0.04:
            return self.grasp
        # grab object (if in grasp mode)
        elif not self.grasped and np.linalg.norm(pos_curr[2] - self.grasp[2]) <= 0.04:
            self.grasped = True
            return self.grasp
        # move end effector to the current subgoal
        elif np.linalg.norm(pos_curr - self.subgoals[0]) > move_precision:
            return self.subgoals[0]
        # if close enough to the current subgoal, move to the next subgoal
        elif len(self.subgoals) > 1:
            self.subgoals.pop(0)
            return self.subgoals[0]
        # replan
        else:
            grasp, transforms = self.calculate_next_plan()
            self.grasp = grasp
            self.subgoals = self.calc_subgoals(grasp, transforms)
            if self.mode == "push": self.init_grasp()
            return self.subgoals[0]
               
    def _grab_effort(self, o_d):
        pos_curr = o_d['hand_pos']

        if self.grasped or self.mode == "push" or not self.grasped and np.linalg.norm(pos_curr[2] - self.grasp[2]) < 0.08:
            return 0.8
        else:
            return -0.8

class DiffusionPolicy(Policy):
    def __init__(self, env, policy_model, camera, task, resolution=(320, 240), obs_cache_size=2, min_action_cache_size=8):
        self.env = env
        self.policy_model = policy_model
        self.camera = camera
        self.task = task
        self.resolution = resolution
        self.obs_cache_size = obs_cache_size # To
        self.min_action_cache_size = min_action_cache_size # Tp - Ta
        assert self.obs_cache_size > 0 and self.min_action_cache_size >= 0

        self.obs_cache = []
        self.action_cache = []

    def reset(self):
        self.obs_cache = []
        self.action_cache = []

    @staticmethod
    @assert_fully_parsed
    def _parse_obs(obs):
        return {
            'hand_pos': obs[:3],
            'unused_info': obs[3:],
        }

    def get_stack_obs(self):
        return np.stack(self.obs_cache, axis=0)
    
    def update_obs_cache(self, obs):
        while len(self.obs_cache) < self.obs_cache_size:
            self.obs_cache.append(obs)
        self.obs_cache.append(obs)
        self.obs_cache.pop(0)
        assert len(self.obs_cache) == self.obs_cache_size
    
    def replan(self):
        stack_obs = self.get_stack_obs()
        self.action_cache = [a for a in self.policy_model(stack_obs, self.task)]
    
    def get_action(self, obs):
        obs, _ = self.env.render(resolution=self.resolution, depth=True, camera_name=self.camera)
        self.update_obs_cache(obs)
        
        if len(self.action_cache) <= self.min_action_cache_size:
            self.replan()
        
        return self.action_cache.pop(0)

class IDPolicy(Policy):
    def __init__(self, env, ID_model, video_model, camera, task, resolution=(320, 240), max_replans=5):
        self.env = env
        self.remain_replans = max_replans + 1
        self.vid_plan = []
        self.ID_model = ID_model
        self.ID_model.cuda()
        self.subgoal_idx = 0
        self.video_model = video_model
        self.resolution = resolution
        self.task = task
        with open("ID_exp/all_cams.json", "r") as f:
            all_cams = json.load(f)
        cam2vec = {cam: torch.eye(len(all_cams))[i] for i, cam in enumerate(all_cams)}
        self.camera = camera
        self.cam_vec = cam2vec[camera]

        self.transform = T.Compose([
            T.CenterCrop((128, 128)),
            T.Resize((224, 224)),
            T.ToTensor(),
        ])
        self.replan()

    def replan(self):
        image, _ = self.env.render(resolution=self.resolution, depth=True, camera_name=self.camera)
        self.subgoal_idx = 0
        self.vid_plan = []
        self.vid_plan = pred_video(self.video_model, image, self.task)
        self.remain_replans -= 1

    @staticmethod
    @assert_fully_parsed
    def _parse_obs(obs):
        return {
            'hand_pos': obs[:3],
            'unused_info': obs[3:],
        }

    def get_action(self, obs):
        obs, _ = self.env.render(resolution=self.resolution, depth=True, camera_name=self.camera)
        obs = self.transform(Image.fromarray(obs))
        subgoal = self.vid_plan[self.subgoal_idx].transpose(1, 2, 0)
        subgoal = self.transform(Image.fromarray(subgoal))
        cam = self.cam_vec

        with torch.no_grad():
            action, is_last = self.ID_model(obs.unsqueeze(0).cuda(), subgoal.unsqueeze(0).cuda(), cam.unsqueeze(0).cuda())
            action = action.squeeze().cpu().numpy()
            is_last = is_last.squeeze().cpu().numpy() > 0
        
        if is_last:
            if self.subgoal_idx < len(self.vid_plan) - 1:
                self.subgoal_idx += 1
            elif self.remain_replans > 0:
                self.replan()
        
        return action

class MyPolicy(Policy):
    def __init__(self, grasp, transforms):
        subgoals = []
        grasp = grasp[0]
        subgoals.append(grasp)
        for transforms in transforms:
            grasp_ext = np.concatenate([subgoals[-1], [1]])
            next_subgoal = (transforms @ grasp_ext)[:3]
            subgoals.append(next_subgoal)
        ### for stablity, extrapolate the last subgoal
        # next_subgoal = subgoals[-1] + (subgoals[-1] - subgoals[-2])
        # subgoals.append(next_subgoal)
        subgoals_np = np.array(subgoals)
        # print(subgoals_np)
        max_deltaz = abs(subgoals_np[1:-2, 2] - subgoals_np[2:-1, 2]).max()
        if max_deltaz > 0.1:
            self.mode = "grasp"
            subgoals = [s + np.array([0, 0, 0.03]) for s in subgoals[:-1]] + [subgoals[-1]]
        else:
            self.mode = "push"
            subgoals = [s - np.array([0, 0, 0.03]) for s in subgoals]
        
        self.grasp = grasp
        self.subgoals = subgoals
        self.init_grasp()  
        
    @staticmethod
    @assert_fully_parsed
    def _parse_obs(obs):
        return {
            'hand_pos': obs[:3],
            'unused_info': obs[3:],
        }
        
    def init_grasp(self):
        self.grasped = False
        if self.mode == "push":
            for subgoal in self.subgoals:
                norm = np.linalg.norm(subgoal[:2] - self.grasp[:2])
                direction = subgoal[:2] - self.grasp[:2]
                direction = direction / norm
                if norm > 0.1:
                    break
            self.grasp[:2] = self.grasp[:2] - direction * 0.08

    def get_action(self, obs):
        o_d = self._parse_obs(obs)

        action = Action({
            'delta_pos': np.arange(3),
            'grab_effort': 3
        })

        action['delta_pos'] = move(o_d['hand_pos'], to_xyz=self._desired_pos(o_d), p=20.)
        action['grab_effort'] = self._grab_effort(o_d)

        return action.array

    def _desired_pos(self, o_d):
        pos_curr = o_d['hand_pos']
        move_precision = 0.12 if self.mode == "push" else 0.04

        # place end effector above object
        if not self.grasped and np.linalg.norm(pos_curr[:2] - self.grasp[:2]) > 0.02:
            return self.grasp + np.array([0., 0., 0.2])
        # drop end effector down on top of object
        elif not self.grasped and np.linalg.norm(pos_curr[2] - self.grasp[2]) > 0.04:
            return self.grasp + np.array([0., 0., 0.03])
        # grab object
        elif not self.grasped and np.linalg.norm(pos_curr[2] - self.grasp[2]) <= 0.04:
            self.grasped = True
            return self.grasp
        # move end effector to the next subgoal
        elif np.linalg.norm(pos_curr - self.subgoals[0]) > move_precision:
            return self.subgoals[0]
        elif len(self.subgoals) > 1:
            self.subgoals.pop(0)
            return self.subgoals[0]
        elif np.linalg.norm(pos_curr - self.subgoals[0]) > 0.02:
            return self.subgoals[0]
        else:
            self.grasped=False
            return self.subgoals[0]
        
    def _grab_effort(self, o_d):
        pos_curr = o_d['hand_pos']

        if self.grasped or self.mode == "push" or not self.grasped and np.linalg.norm(pos_curr[2] - self.grasp[2]) < 0.08:
            return 0.8
        else:
            return -0.8

class MyPolicy_CL(Policy):
    def __init__(self, env, task, camera, video_model, flow_model, resolution=(320, 240), plan_timeout=15, max_replans=0, log=False):
        self.env = env
        self.seg_ids = name2maskid[task]
        self.task = " ".join(task.split('-')[:-3])
        self.camera = camera
        self.video_model = video_model
        self.flow_model = flow_model
        self.resolution = resolution
        self.plan_timeout = plan_timeout
        self.last_pos = np.array([0, 0, 0])
        self.max_replans = max_replans
        self.replans = max_replans + 1
        self.time_from_last_plan = 0
        self.log = log

        grasp, transforms = self.calculate_next_plan()
        grasp = grasp[0]

        subgoals = self.calc_subgoals(grasp, transforms)
        ### for stablity, extrapolate the last subgoal
        # next_subgoal = subgoals[-1] + (subgoals[-1] - subgoals[-2])
        # subgoals.append(next_subgoal)
        subgoals_np = np.array(subgoals)
        # print(subgoals_np)
        max_deltaz = abs(subgoals_np[1:-2, 2] - subgoals_np[2:-1, 2]).max()
        if max_deltaz > 0.1:
            self.mode = "grasp"
        else:
            self.mode = "push"
            # move the gripper down a bit for robustness
            # (Attempt to use the gripper wrist for pushing, not fingers)
            subgoals = [s - np.array([0, 0, 0.03]) for s in subgoals]
        
        self.grasp = grasp
        self.subgoals = subgoals
        self.init_grasp()  

    def calc_subgoals(self, grasp, transforms):
        subgoals = [grasp]
        for transforms in transforms:
            grasp_ext = np.concatenate([subgoals[-1], [1]])
            next_subgoal = (transforms @ grasp_ext)[:3]
            subgoals.append(next_subgoal)
        return subgoals

    def calculate_next_plan(self):
        image, depth = self.env.render(resolution=self.resolution, depth=True, camera_name=self.camera)
        cmat = get_cmat(self.env, self.camera, resolution=self.resolution)
        seg = get_seg(self.env, resolution=self.resolution, camera=self.camera, seg_ids=self.seg_ids)

        # measure time for vidgen
        start = time.time()
        images = pred_video(self.video_model, image, self.task)
        time_vid = time.time() - start

        # measure time for flow
        start = time.time()
        image1, image2, color, flow, flow_b = pred_flow_frame(self.flow_model, images)
        time_flow = time.time() - start

        # measure time for action planning
        grasp, transforms, center_2ds, sampless = get_transforms(seg, depth, cmat, flow)
        transform_mats = [get_transformation_matrix(*transform) for transform in transforms]
        time_action = time.time() - start

        t = len(transform_mats)
        if self.log: log_time(time_vid/t, time_flow/t, time_action/t, self.max_replans-self.replans+1)
        if self.log and (self.time_from_last_plan!=0): log_time_execution(self.time_from_last_plan*0.1/t, self.max_replans-self.replans)

        self.replans -= 1
        self.replan_countdown = self.plan_timeout
        self.time_from_last_plan = 0

        return grasp, transform_mats

    @staticmethod
    @assert_fully_parsed
    def _parse_obs(obs):
        return {
            'hand_pos': obs[:3],
            'unused_info': obs[3:],
        }
        
    def init_grasp(self):
        self.grasped = False
        if self.mode == "push":
            for subgoal in self.subgoals:
                norm = np.linalg.norm(subgoal[:2] - self.grasp[:2])
                direction = subgoal[:2] - self.grasp[:2]
                direction = direction / norm
                if norm > 0.1:
                    break
            self.grasp[:2] = self.grasp[:2] - direction * 0.08

    def get_action(self, obs):
        o_d = self._parse_obs(obs)
        # if stucked (not moving), step the countdown
        if np.linalg.norm(o_d['hand_pos'] - self.last_pos) < 0.001:
            self.replan_countdown -= 1
        self.last_pos = o_d['hand_pos']

        self.time_from_last_plan += 1

        action = Action({
            'delta_pos': np.arange(3),
            'grab_effort': 3
        })

        action['delta_pos'] = move(o_d['hand_pos'], to_xyz=self._desired_pos(o_d), p=20.)
        action['grab_effort'] = self._grab_effort(o_d)

        return action.array

    def _desired_pos(self, o_d):
        pos_curr = o_d['hand_pos']
        move_precision = 0.12 if self.mode == "push" else 0.04

        # if stucked/stopped(all subgoals reached), replan
        if self.replan_countdown <= 0 and self.replans > 0:
            grasp, transforms = self.calculate_next_plan()
            self.grasp = grasp[0]
            self.subgoals = self.calc_subgoals(grasp[0], transforms)
            if self.mode == "push": self.init_grasp()
            return self.subgoals[0]
        # place end effector above object
        elif not self.grasped and np.linalg.norm(pos_curr[:2] - self.grasp[:2]) > 0.02:
            return self.grasp + np.array([0., 0., 0.2])
        # drop end effector down on top of object
        elif not self.grasped and np.linalg.norm(pos_curr[2] - self.grasp[2]) > 0.04:
            return self.grasp
        # grab object (if in grasp mode)
        elif not self.grasped and np.linalg.norm(pos_curr[2] - self.grasp[2]) <= 0.04:
            self.grasped = True
            return self.grasp
        # move end effector to the current subgoal
        elif np.linalg.norm(pos_curr - self.subgoals[0]) > move_precision:
            return self.subgoals[0]
        # if close enough to the current subgoal, move to the next subgoal
        elif len(self.subgoals) > 1:
            self.subgoals.pop(0)
            return self.subgoals[0]
        # move to the last subgoal
        # ideally the gripper will stop at the last subgoal and the countdown will run out quickly
        # and then the next plan (next set of subgoals) will be calculated
        else:
            return self.subgoals[0]
               
    def _grab_effort(self, o_d):
        pos_curr = o_d['hand_pos']

        if self.grasped or self.mode == "push" or not self.grasped and np.linalg.norm(pos_curr[2] - self.grasp[2]) < 0.08:
            return 0.8
        else:
            return -0.8


class MyPolicy_CL_seg(Policy):
    def __init__(self, env, task, camera, video_model, flow_model, seg_model, resolution=(320, 240), plan_timeout=15, max_replans=0, log=False):
        self.env = env
        self.seg_ids = name2maskid[task]
        self.task = " ".join(task.split('-')[:-3])
        self.camera = camera
        self.video_model = video_model
        self.flow_model = flow_model
        self.seg_model = seg_model
        self.resolution = resolution
        self.plan_timeout = plan_timeout
        self.last_pos = np.array([0, 0, 0])
        self.max_replans = max_replans
        self.replans = max_replans + 1
        self.time_from_last_plan = 0
        self.log = log

        grasp, transforms = self.calculate_next_plan()
        grasp = grasp[0]

        subgoals = self.calc_subgoals(grasp, transforms)
        ### for stablity, extrapolate the last subgoal
        # next_subgoal = subgoals[-1] + (subgoals[-1] - subgoals[-2])
        # subgoals.append(next_subgoal)
        subgoals_np = np.array(subgoals)
        # print(subgoals_np)
        max_deltaz = abs(subgoals_np[1:-2, 2] - subgoals_np[2:-1, 2]).max()
        if max_deltaz > 0.1:
            self.mode = "grasp"
        else:
            self.mode = "push"
            # move the gripper down a bit for robustness
            # (Attempt to use the gripper wrist for pushing, not fingers)
            subgoals = [s - np.array([0, 0, 0.03]) for s in subgoals]
        
        self.grasp = grasp
        self.subgoals = subgoals
        self.init_grasp()  

    def get_seg(self, resolution):
        image, _ = self.env.render(depth=True, camera_name=self.camera)
        image = Image.fromarray(image)
        with open("seg_text.json", "r") as f:
            seg_text = json.load(f)
            text_prompt = seg_text[self.task]

        with torch.no_grad():
            masks, *_ = self.seg_model.predict(image, text_prompt)
            mask = masks[0].cpu().numpy()
        # resize to resolution
        mask = cv2.resize(mask.astype('uint8') * 255, resolution)
        # convert to binary
        mask = (mask > 0)
        return mask


    def calc_subgoals(self, grasp, transforms):
        subgoals = [grasp]
        for transforms in transforms:
            grasp_ext = np.concatenate([subgoals[-1], [1]])
            next_subgoal = (transforms @ grasp_ext)[:3]
            subgoals.append(next_subgoal)
        return subgoals

    def calculate_next_plan(self):
        image, depth = self.env.render(resolution=self.resolution, depth=True, camera_name=self.camera)
        cmat = get_cmat(self.env, self.camera, resolution=self.resolution)
        # seg = get_seg(self.env, resolution=self.resolution, camera=self.camera, seg_ids=self.seg_ids)
        seg = self.get_seg(self.resolution)

        # measure time for vidgen
        start = time.time()
        images = pred_video(self.video_model, image, self.task)
        time_vid = time.time() - start

        # measure time for flow
        start = time.time()
        image1, image2, color, flow, flow_b = pred_flow_frame(self.flow_model, images, device="cuda:0")
        time_flow = time.time() - start

        # measure time for action planning
        grasp, transforms, center_2ds, sampless = get_transforms(seg, depth, cmat, flow)
        transform_mats = [get_transformation_matrix(*transform) for transform in transforms]
        time_action = time.time() - start

        t = len(transform_mats)
        if self.log: log_time(time_vid/t, time_flow/t, time_action/t, self.max_replans-self.replans+1)
        if self.log and (self.time_from_last_plan!=0): log_time_execution(self.time_from_last_plan*0.1/t, self.max_replans-self.replans)

        self.replans -= 1
        self.replan_countdown = self.plan_timeout
        self.time_from_last_plan = 0

        return grasp, transform_mats

    @staticmethod
    @assert_fully_parsed
    def _parse_obs(obs):
        return {
            'hand_pos': obs[:3],
            'unused_info': obs[3:],
        }
        
    def init_grasp(self):
        self.grasped = False
        if self.mode == "push":
            for subgoal in self.subgoals:
                norm = np.linalg.norm(subgoal[:2] - self.grasp[:2])
                direction = subgoal[:2] - self.grasp[:2]
                direction = direction / norm
                if norm > 0.1:
                    break
            self.grasp[:2] = self.grasp[:2] - direction * 0.08

    def get_action(self, obs):
        o_d = self._parse_obs(obs)
        # if stucked (not moving), step the countdown
        if np.linalg.norm(o_d['hand_pos'] - self.last_pos) < 0.001:
            self.replan_countdown -= 1
        self.last_pos = o_d['hand_pos']

        self.time_from_last_plan += 1

        action = Action({
            'delta_pos': np.arange(3),
            'grab_effort': 3
        })

        action['delta_pos'] = move(o_d['hand_pos'], to_xyz=self._desired_pos(o_d), p=20.)
        action['grab_effort'] = self._grab_effort(o_d)

        return action.array

    def _desired_pos(self, o_d):
        pos_curr = o_d['hand_pos']
        move_precision = 0.12 if self.mode == "push" else 0.04

        # if stucked/stopped(all subgoals reached), replan
        if self.replan_countdown <= 0 and self.replans > 0:
            grasp, transforms = self.calculate_next_plan()
            self.grasp = grasp[0]
            self.subgoals = self.calc_subgoals(grasp[0], transforms)
            if self.mode == "push": self.init_grasp()
            return self.subgoals[0]
        # place end effector above object
        elif not self.grasped and np.linalg.norm(pos_curr[:2] - self.grasp[:2]) > 0.02:
            return self.grasp + np.array([0., 0., 0.2])
        # drop end effector down on top of object
        elif not self.grasped and np.linalg.norm(pos_curr[2] - self.grasp[2]) > 0.04:
            return self.grasp
        # grab object (if in grasp mode)
        elif not self.grasped and np.linalg.norm(pos_curr[2] - self.grasp[2]) <= 0.04:
            self.grasped = True
            return self.grasp
        # move end effector to the current subgoal
        elif np.linalg.norm(pos_curr - self.subgoals[0]) > move_precision:
            return self.subgoals[0]
        # if close enough to the current subgoal, move to the next subgoal
        elif len(self.subgoals) > 1:
            self.subgoals.pop(0)
            return self.subgoals[0]
        # move to the last subgoal
        # ideally the gripper will stop at the last subgoal and the countdown will run out quickly
        # and then the next plan (next set of subgoals) will be calculated
        else:
            return self.subgoals[0]
               
    def _grab_effort(self, o_d):
        pos_curr = o_d['hand_pos']

        if self.grasped or self.mode == "push" or not self.grasped and np.linalg.norm(pos_curr[2] - self.grasp[2]) < 0.08:
            return 0.8
        else:
            return -0.8



class MyPolicy_Flow(Policy):
    def __init__(self, env, task, camera, video_flow_model, resolution=(320, 240), plan_timeout=15, max_replans=0):
        self.env = env
        self.seg_ids = name2maskid[task]
        self.task = " ".join(task.split('-')[:-3])
        self.camera = camera
        self.video_flow_model = video_flow_model
        self.resolution = resolution
        self.plan_timeout = plan_timeout
        self.last_pos = np.array([0, 0, 0])
        self.replans = max_replans + 1

        grasp, transforms = self.calculate_next_plan()
        grasp = grasp[0]

        subgoals = self.calc_subgoals(grasp, transforms)
        ### for stablity, extrapolate the last subgoal
        # next_subgoal = subgoals[-1] + (subgoals[-1] - subgoals[-2])
        # subgoals.append(next_subgoal)
        subgoals_np = np.array(subgoals)
        # print(subgoals_np)
        max_deltaz = abs(subgoals_np[1:-2, 2] - subgoals_np[2:-1, 2]).max()
        if max_deltaz > 0.1:
            self.mode = "grasp"
        else:
            self.mode = "push"
            # move the gripper down a bit for robustness
            # (Attempt to use the gripper wrist for pushing, not fingers)
            subgoals = [s - np.array([0, 0, 0.03]) for s in subgoals]
        
        self.grasp = grasp
        self.subgoals = subgoals
        self.init_grasp()  

    def calc_subgoals(self, grasp, transforms):
        subgoals = [grasp]
        for transforms in transforms:
            grasp_ext = np.concatenate([subgoals[-1], [1]])
            next_subgoal = (transforms @ grasp_ext)[:3]
            subgoals.append(next_subgoal)
        return subgoals

    def calculate_next_plan(self):
        self.replans -= 1
        self.replan_countdown = self.plan_timeout
        self.time_from_last_plan = 0
        image, depth = self.env.render(resolution=self.resolution, depth=True, camera_name=self.camera)
        cmat = get_cmat(self.env, self.camera, resolution=self.resolution)
        seg = get_seg(self.env, resolution=self.resolution, camera=self.camera, seg_ids=self.seg_ids)

        flows = pred_video(self.video_flow_model, image, self.task, flow=True)

        grasp, transforms, center_2ds, sampless = get_transforms(seg, depth, cmat, flows)

        transform_mats = [get_transformation_matrix(*transform) for transform in transforms]

        return grasp, transform_mats

    @staticmethod
    @assert_fully_parsed
    def _parse_obs(obs):
        return {
            'hand_pos': obs[:3],
            'unused_info': obs[3:],
        }
        
    def init_grasp(self):
        self.grasped = False
        if self.mode == "push":
            for subgoal in self.subgoals:
                norm = np.linalg.norm(subgoal[:2] - self.grasp[:2])
                direction = subgoal[:2] - self.grasp[:2]
                direction = direction / norm
                if norm > 0.1:
                    break
            self.grasp[:2] = self.grasp[:2] - direction * 0.08

    def get_action(self, obs):
        o_d = self._parse_obs(obs)
        # if stucked (not moving), step the countdown
        if np.linalg.norm(o_d['hand_pos'] - self.last_pos) < 0.001:
            self.replan_countdown -= 1
        self.last_pos = o_d['hand_pos']

        action = Action({
            'delta_pos': np.arange(3),
            'grab_effort': 3
        })

        action['delta_pos'] = move(o_d['hand_pos'], to_xyz=self._desired_pos(o_d), p=20.)
        action['grab_effort'] = self._grab_effort(o_d)

        return action.array

    def _desired_pos(self, o_d):
        pos_curr = o_d['hand_pos']
        move_precision = 0.12 if self.mode == "push" else 0.04

        # if stucked/stopped(all subgoals reached), replan
        if self.replan_countdown <= 0 and self.replans > 0:
            grasp, transforms = self.calculate_next_plan()
            self.grasp = grasp[0]
            self.subgoals = self.calc_subgoals(grasp[0], transforms)
            if self.mode == "push": self.init_grasp()
            return self.subgoals[0]
        # place end effector above object
        elif not self.grasped and np.linalg.norm(pos_curr[:2] - self.grasp[:2]) > 0.02:
            return self.grasp + np.array([0., 0., 0.2])
        # drop end effector down on top of object
        elif not self.grasped and np.linalg.norm(pos_curr[2] - self.grasp[2]) > 0.04:
            return self.grasp
        # grab object (if in grasp mode)
        elif not self.grasped and np.linalg.norm(pos_curr[2] - self.grasp[2]) <= 0.04:
            self.grasped = True
            return self.grasp
        # move end effector to the current subgoal
        elif np.linalg.norm(pos_curr - self.subgoals[0]) > move_precision:
            return self.subgoals[0]
        # if close enough to the current subgoal, move to the next subgoal
        elif len(self.subgoals) > 1:
            self.subgoals.pop(0)
            return self.subgoals[0]
        # move to the last subgoal
        # ideally the gripper will stop at the last subgoal and the countdown will run out quickly
        # and then the next plan (next set of subgoals) will be calculated
        else:
            return self.subgoals[0]
            
        
    def _grab_effort(self, o_d):
        pos_curr = o_d['hand_pos']

        if self.grasped or self.mode == "push" or not self.grasped and np.linalg.norm(pos_curr[2] - self.grasp[2]) < 0.08:
            return 0.8
        else:
            return -0.8