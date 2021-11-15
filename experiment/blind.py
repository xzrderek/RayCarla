from oatomobile.envs import CARLAEnv

# Initializes a CARLA environment.
environment = CARLAEnv(town="Town01")

# Makes an initial observation.
observation = environment.reset()
done = False

while not done:
  # Selects a random action.
  action = environment.action_space.sample()
  observation, reward, done, info = environment.step(action)

  # Renders interactive display.
  environment.render(mode="human")

# Book-keeping: closes
environment.close()
