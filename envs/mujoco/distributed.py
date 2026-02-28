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
        results = [env.reset() for env in self.environments]
        observations = [r[0] for r in results]
        self.lengths = np.zeros(len(self.environments), int)
        
        if isinstance(observations[0], dict):
            keys = observations[0].keys()
            return {k: np.array([o[k] for o in observations], np.float32) for k in keys}
        return np.array(observations, np.float32)

    def step(self, actions):
        next_observations = []  # Observations for the transitions.
        rewards = []
        resets = []
        terminations = []
        observations = []  # Observations for the actions selection.

        for i in range(len(self.environments)):
            ob, rew, term, trunc, _ = self.environments[i].step(actions[i])
            
            self.lengths[i] += 1
            # Timeouts trigger resets but are not true terminations.
            reset = term or trunc or self.lengths[i] == self.max_episode_steps
            next_observations.append(ob)
            rewards.append(rew)
            resets.append(reset)
            terminations.append(term)

            if reset:
                ob, _ = self.environments[i].reset()
                self.lengths[i] = 0

            observations.append(ob)

        if isinstance(observations[0], dict):
            keys = observations[0].keys()
            observations = {k: np.array([o[k] for o in observations], np.float32) for k in keys}
            next_observations_stacked = {k: np.array([o[k] for o in next_observations], np.float32) for k in keys}
            infos = dict(
                observations=next_observations_stacked,
                rewards=np.array(rewards, np.float32),
                resets=np.array(resets, bool),
                terminations=np.array(terminations, bool))
            return observations, infos
        else:
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

def proc(action_pipe, index, environment_builder, max_episode_steps, workers_per_group, output_queue):
    '''Process holding a sequential group of environments.'''
    envs = Sequential(environment_builder, max_episode_steps, workers_per_group)
    observations = envs.start()
    output_queue.put((index, observations))

    while True:
        try:
            actions = action_pipe.recv()
            out = envs.step(actions)
            output_queue.put((index, out))
        except (EOFError, BrokenPipeError):
            break
        
class Parallel:
    '''A group of sequential environments used in parallel.'''

    def __init__(
        self, environment_builder, worker_groups, workers_per_group,
        max_episode_steps
    ):
        self.environment_builder = environment_builder
        self.worker_groups = worker_groups
        self.workers_per_group = workers_per_group
        self.max_episode_steps = max_episode_steps

        dummy_environment = self.environment_builder()
        self.observation_space = dummy_environment.observation_space
        self.action_space = dummy_environment.action_space
        del dummy_environment
        self.started = False

        self.output_queue = multiprocessing.Queue()
        self.action_pipes = []

        for i in range(self.worker_groups):
            pipe, worker_end = multiprocessing.Pipe()
            self.action_pipes.append(pipe)
            process = multiprocessing.Process(
                target=proc, args=(worker_end, i, self.environment_builder,
                                   self.max_episode_steps, self.workers_per_group, self.output_queue))
            process.daemon = True
            process.start()

    def start(self):
        '''Used once to get the initial observations.'''
        assert not self.started
        self.started = True
        observations_list = [None for _ in range(self.worker_groups)]

        for _ in range(self.worker_groups):
            index, observations = self.output_queue.get()
            observations_list[index] = observations

        self.observations_list = observations_list # Keep as list
        
        self.rewards_list = np.zeros(
            (self.worker_groups, self.workers_per_group), np.float32)
        self.resets_list = np.zeros(
            (self.worker_groups, self.workers_per_group), bool)
        self.terminations_list = np.zeros(
            (self.worker_groups, self.workers_per_group), bool)
            
        # Check if observations are dicts
        if isinstance(observations_list[0], dict):
            keys = observations_list[0].keys()
            self.next_observations_list = [{k: np.zeros_like(observations_list[i][k]) for k in keys} for i in range(self.worker_groups)]
            return {k: np.concatenate([o[k] for o in observations_list]) for k in keys}
        else:
            self.observations_list = np.array(observations_list)
            self.next_observations_list = np.zeros_like(self.observations_list)
            return np.concatenate(self.observations_list)

    def step(self, actions):
        actions_list = np.split(actions, self.worker_groups)
        for actions, pipe in zip(actions_list, self.action_pipes):
            pipe.send(actions)

        for _ in range(self.worker_groups):
            index, (observations, infos) = self.output_queue.get()
            self.observations_list[index] = observations
            self.next_observations_list[index] = infos['observations']
            self.rewards_list[index] = infos['rewards']
            self.resets_list[index] = infos['resets']
            self.terminations_list[index] = infos['terminations']

        if isinstance(self.observations_list[0], dict):
            keys = self.observations_list[0].keys()
            observations = {k: np.concatenate([o[k] for o in self.observations_list]) for k in keys}
            next_observations = {k: np.concatenate([o[k] for o in self.next_observations_list]) for k in keys}
        else:
            observations = np.concatenate(self.observations_list)
            next_observations = np.concatenate(self.next_observations_list)
            
        infos = dict(
            observations=next_observations,
            rewards=np.concatenate(self.rewards_list),
            resets=np.concatenate(self.resets_list),
            terminations=np.concatenate(self.terminations_list))
        return observations, infos


def distribute(environment_builder, worker_groups=1, workers_per_group=1, max_episode_steps=2000):
    '''Distributes workers over parallel and sequential groups.'''

    if worker_groups < 2:
        return Sequential(
            environment_builder, max_episode_steps=max_episode_steps,
            workers=workers_per_group)

    return Parallel(
        environment_builder, worker_groups=worker_groups,
        workers_per_group=workers_per_group,
        max_episode_steps=max_episode_steps)