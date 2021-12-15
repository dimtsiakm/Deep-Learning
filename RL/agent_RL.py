from tiago_RL import Env
from stable_baselines3 import PPO
import random
from stable_baselines3.common.env_checker import check_env
import numpy as np
import os
from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback


class SaveOnBestTrainingRewardCallback(BaseCallback):
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), 'timesteps')
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])
                if self.verbose > 0:
                    print("Num timesteps: {}".format(self.num_timesteps))
                    print(
                        "Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward,
                                                                                                 mean_reward))
                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    if self.verbose > 0:
                        print("Saving new best model to {}".format(self.save_path))
                    self.model.save(self.save_path)
        return True


if __name__ == '__main__':
    log_dir = "tmp/"
    os.makedirs(log_dir, exist_ok=True)

    print('brake dance')

    env = Env()
    env = Monitor(env, log_dir)
    env.render = True
    # model = PPO.load('./results/zoom_and_distance.zip')
    # model.set_env(env)

    model = PPO('CnnPolicy', env, verbose=1, _init_setup_model=True, use_sde=False,
                tensorboard_log='./results/tensorboard/', learning_rate=2e-3, n_epochs=5, n_steps=500)

    callback = SaveOnBestTrainingRewardCallback(check_freq=10000, log_dir=log_dir)
    model.learn(total_timesteps=200000, callback=callback)


    model.save('./results/zoom_and_distance')
