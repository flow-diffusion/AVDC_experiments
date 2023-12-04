from mujoco_py.generated import const
import numpy as np
import cv2

def get_robot_seg(env):
    seg = env.render(segmentation=True)
    img = np.zeros(seg.shape[:2], dtype=bool)
    types = seg[:, :, 0]
    ids = seg[:, :, 1]
    geoms = types == const.OBJ_GEOM
    geoms_ids = np.unique(ids[geoms])

    for i in geoms_ids:
        if 2 <= i <= 33:
            img[ids == i] = True
    return img

def get_seg(env, camera, resolution, seg_ids):
    seg = env.render(segmentation=True, resolution=resolution, camera_name=camera)
    img = np.zeros(seg.shape[:2], dtype=bool)
    types = seg[:, :, 0]
    ids = seg[:, :, 1]
    geoms = types == const.OBJ_GEOM
    geoms_ids = np.unique(ids[geoms])
    # print(geoms_ids)

    for i in geoms_ids:
        if i in seg_ids:
            img[ids == i] = True
    img = img.astype('uint8') * 255
    return cv2.medianBlur(img, 3)

def get_cmat(env, cam_name, resolution):
    id = env.sim.model.camera_name2id(cam_name)
    fov = env.sim.model.cam_fovy[id]
    pos = env.sim.data.cam_xpos[id]
    rot = env.sim.data.cam_xmat[id].reshape(3, 3).T
    width, height = resolution
    # Translation matrix (4x4).
    translation = np.eye(4)
    translation[0:3, 3] = -pos
    # Rotation matrix (4x4).
    rotation = np.eye(4)
    rotation[0:3, 0:3] = rot
    # Focal transformation matrix (3x4).
    focal_scaling = (1./np.tan(np.deg2rad(fov)/2)) * height / 2.0 # focal length
    focal = np.diag([-focal_scaling, -focal_scaling, 1.0, 0])[0:3, :]
    # Image matrix (3x3).
    image = np.eye(3)
    image[0, 2] = (width - 1) / 2.0
    image[1, 2] = (height - 1) / 2.0
    
    return image @ focal @ rotation @ translation

def collect_video(init_obs, env, policy, camera_name='corner3', resolution=(640, 480)):
    images = []
    depths = []
    episode_return = 0
    done = False
    obs = init_obs
    if camera_name is None:
        cameras = ["corner3", "corner", "corner2"]
        camera_name = np.random.choice(cameras)

    image, depth = env.render(depth=True, offscreen=True, camera_name=camera_name, resolution=resolution)
    images += [image]
    depths += [depth]
    
    dd = 10 ### collect a few more steps after done
    while dd:
        action = policy.get_action(obs)
        try:
            obs, reward, done, info = env.step(action)
            done = info['success']
            dd -= done
            episode_return += reward
        except Exception as e:
            print(e)
            break
        if dd != 10 and not done:
            break
        image, depth = env.render(depth=True, offscreen=True, camera_name=camera_name, resolution=resolution)
        images += [image]
        depths += [depth]
        
        
        
    return images, depths, episode_return

def sample_n_frames(frames, n):
    new_vid_ind = [int(i*len(frames)/(n-1)) for i in range(n-1)] + [len(frames)-1]
    return np.array([frames[i] for i in new_vid_ind])



    
    