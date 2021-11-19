# Rule-based agents.

import oatomobile
from oatomobile.envs import CARLAEnv
from oatomobile.baselines.rulebased import AutopilotAgent

town = "Town01"

# # Initializes a CARLA environment.
# environment = CARLAEnv(town="Town01")
# observation = environment.reset()

agent = AutopilotAgent(environment)
action = agent.act(observation)

environment.close()

def main():
  try:
    # Setups the environment.
    env = CARLAEnv(
        town=town,
        fps=20,
        sensors=sensors,
    )
    if max_episode_steps is not None:
      env = oatomobile.FiniteHorizonWrapper(
          env,
          max_episode_steps=max_episode_steps,
      )
    if output_dir is not None:
      env = oatomobile.SaveToDiskWrapper(env, output_dir=output_dir)
    env = oatomobile.MonitorWrapper(env, output_fname="tmp/yoo.gif")

    # Runs the environment loop.
    oatomobile.EnvironmentLoop(
        agent_fn=AutopilotAgent,
        environment=env,
        render_mode="human" if render else "none",
    ).run()

  finally:
    # Garbage collector.
    try:
      env.close()
    except NameError:
      pass
