from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE as env_dict
import metaworld
import imageio

resolution = (640, 480)
print(env_dict.keys())

env = env_dict['push-v2-goal-observable']()

env.reset()
image, _ = env.render(resolution=resolution, depth=True, camera_name="corner3")

print(image.shape)
# save img
imageio.imwrite('test.png', image)

