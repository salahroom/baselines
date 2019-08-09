import numpy as np
from baselines.a2c.utils import discount_with_dones
from baselines.common.runners import AbstractEnvRunner

class Runner(AbstractEnvRunner):
    """
    We use this class to generate batches of experiences

    __init__:
    - Initialize the runner

    run():
    - Make a mini batch of experiences
    """
    def __init__(self, env, model, nsteps=5, gamma=0.99):
        super().__init__(env=env, model=model, nsteps=nsteps)
        self.gamma = gamma
        self.batch_action_shape = [x if x is not None else -1 for x in model.train_model.action.shape.as_list()]
        self.ob_dtype = model.train_model.X.dtype.as_numpy_dtype

    def run(self):
        # We initialize the lists that will contain the mb of experiences
        #print("stop1")
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones = [],[],[],[],[]
        #print("stop2")
        mb_states = self.states
        #print("stop3")
        epinfos = []
        #print("stop4")
        for n in range(self.nsteps):
            # Given observations, take action and value (V(s))
            # We already have self.obs because Runner superclass run self.obs[:] = env.reset() on init
            #print("steooooop1")
            #print(self.obs, self.states, self.dones)
            #print(self.model)
            #print(self.model.step)
            actions, values, states, _ = self.model.step(self.obs, S=self.states, M=self.dones)
            #print("steooooop2")

            # Append the experiences
            mb_obs.append(np.copy(self.obs))
            #print("steooooop3")
            mb_actions.append(actions)
            #print("steooooop4")
            mb_values.append(values)
            #print("steooooop5")
            mb_dones.append(self.dones)

            #print("steooooop6")
            # Take actions in env and look the results
            obs, rewards, dones, infos = self.env.step(actions)
            #print("steooooop7")
            for info in infos:
                maybeepinfo = info.get('episode')
                if maybeepinfo: epinfos.append(maybeepinfo)
            #print("steooooop8")
            self.states = states
            #print("steooooop9")
            self.dones = dones
            #print("steooooop10")
            self.obs = obs
            #print("steooooop11")
            mb_rewards.append(rewards)
            #print("steooooop12")
        mb_dones.append(self.dones)
        #print("steooooop13")
        # Batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=self.ob_dtype).swapaxes(1, 0).reshape(self.batch_ob_shape)
        #print("steooooop14")
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32).swapaxes(1, 0)
        #print("steooooop15")
        mb_actions = np.asarray(mb_actions, dtype=self.model.train_model.action.dtype.name).swapaxes(1, 0)
        #print("steooooop16")
        mb_values = np.asarray(mb_values, dtype=np.float32).swapaxes(1, 0)
        #print("steooooop17")
        mb_dones = np.asarray(mb_dones, dtype=np.bool).swapaxes(1, 0)
        #print("steooooop18")
        mb_masks = mb_dones[:, :-1]
        #print("steooooop19")
        mb_dones = mb_dones[:, 1:]
        #print("steooooop20")


        if self.gamma > 0.0:
            # Discount/bootstrap off value fn
            last_values = self.model.value(self.obs, S=self.states, M=self.dones).tolist()
            for n, (rewards, dones, value) in enumerate(zip(mb_rewards, mb_dones, last_values)):
                rewards = rewards.tolist()
                dones = dones.tolist()
                if dones[-1] == 0:
                    rewards = discount_with_dones(rewards+[value], dones+[0], self.gamma)[:-1]
                else:
                    rewards = discount_with_dones(rewards, dones, self.gamma)

                mb_rewards[n] = rewards

        mb_actions = mb_actions.reshape(self.batch_action_shape)

        mb_rewards = mb_rewards.flatten()
        mb_values = mb_values.flatten()
        mb_masks = mb_masks.flatten()
        return mb_obs, mb_states, mb_rewards, mb_masks, mb_actions, mb_values, epinfos
