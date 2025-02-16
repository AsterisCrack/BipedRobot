'''Builders for distributed training.'''

import multiprocessing

import numpy as np


class Sequential:
    '''A group of environments used in sequence.'''

    def __init__(self, environment_builder, max_episode_steps, workers):
        self.environments = [environment_builder() for _ in range(workers)]
        self.max_episode_steps = max_episode_steps
        self.observation_space = self.environments[0].observation_space
        self.action_space = self.environments[0].action_space
        self.name = self.environments[0].name

    def start(self):
        '''Used once to get the initial observations.'''
        observations = [env.reset() for env in self.environments]
        self.lengths = np.zeros(len(self.environments), int)
        return np.array(observations, np.float32)

    def step(self, actions):
        next_observations = []  # Observations for the transitions.
        rewards = []
        resets = []
        terminations = []
        observations = []  # Observations for the actions selection.

        for i in range(len(self.environments)):
            ob, rew, term, _ = self.environments[i].step(actions[i])

            self.lengths[i] += 1
            # Timeouts trigger resets but are not true terminations.
            reset = term or self.lengths[i] == self.max_episode_steps
            next_observations.append(ob)
            rewards.append(rew)
            resets.append(reset)
            terminations.append(term)

            if reset:
                ob = self.environments[i].reset()
                self.lengths[i] = 0

            observations.append(ob)

        observations = np.array(observations, np.float32)
        infos = dict(
            observations=np.array(next_observations, np.float32),
            rewards=np.array(rewards, np.float32),
            resets=np.array(resets, bool),
            terminations=np.array(terminations, bool))
        return observations, infos

    def render(self, mode='human', *args, **kwargs):
        outs = []
        for env in self.environments:
            out = env.render(mode=mode, *args, **kwargs)
            outs.append(out)
        if mode != 'human':
            return np.array(outs)


def distribute(environment_builder, workers_per_group=1):
    '''Distributes workers over parallel and sequential groups.'''
    dummy_environment = environment_builder()
    max_episode_steps = dummy_environment.max_episode_steps
    del dummy_environment
    return Sequential(
        environment_builder, max_episode_steps=max_episode_steps,
        workers=workers_per_group)
