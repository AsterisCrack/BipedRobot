import numpy as np
import time
from envs import basic_env # Replace with your actual environment import

def sinusoidal_action_pattern(step, frequency=0.1, amplitude=1.0, action_dim=6):
    """
    Generates a sinusoidal action pattern for smoother movements.
    :param step: Current time step.
    :param frequency: Frequency of the sinusoidal wave.
    :param amplitude: Amplitude of the sinusoidal wave.
    :param action_dim: Number of actions (dimensionality of action space).
    :return: Array of sinusoidal actions.
    """
    return amplitude * np.sin(frequency * step + np.linspace(0, np.pi, action_dim))

def no_action_pattern(step, action_dim=6):
    """
    Generates a no action pattern.
    :param step: Current time step.
    :param action_dim: Number of actions (dimensionality of action space).
    :return: Array of zeros.
    """
    return np.zeros(action_dim)

def test_env():
    env = basic_env.BasicEnv(render_mode="human")
    obs, info = env.reset()
    action_dim = env.action_space.shape[0]

    print("Testing Basic Environment. Press Ctrl+C to stop.")
    try:
        for step in range(1000):  # Run for a fixed number of steps or until manually interrupted
            # action = sinusoidal_action_pattern(step, frequency=0.05, amplitude=0.5, action_dim=action_dim)
            action = no_action_pattern(step, action_dim=action_dim)
            obs, reward, terminated, truncated, info = env.step(action)
            print("Observation:", obs)
            env.render()

            # Debugging information
            # print(f"Step: {step}, Action: {action}, Observation: {obs}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")

            if terminated or truncated:
                obs, info = env.reset()
                
            time.sleep(1 / env.metadata["render_fps"])  # Add delay to match rendering FPS
    except KeyboardInterrupt:
        print("\nTest interrupted by user.")
    finally:
        env.close()

if __name__ == "__main__":
    test_env()
