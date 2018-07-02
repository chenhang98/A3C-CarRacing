from matplotlib import pyplot as plt
from torch.autograd import Variable
from collections import deque
import torch.nn.functional as F
import numpy as np
import torch
import time

from ACmodel import ActorCritic
from envs import create_env
import constants as c


if __name__ == "__main__":
    with torch.no_grad():
        # load model and simulator
        env = create_env(c.env_name)
        model = ActorCritic(env.observation_space.shape[0], c.output_size)
        model.load_state_dict(torch.load("bak/weights.pkl"))

        # shift model to eval mode
        model.eval()

        # init values
        done = False
        rewards = []
        episode_length = 0
        state = env.reset()
        recent_rewards = deque(maxlen = 500)
        
        hx = Variable(torch.zeros(1, 256))
        cx = Variable(torch.zeros(1, 256))

        # play
        while not done:
            episode_length += 1
            state = torch.from_numpy(state)
        
            # get predict action
            value, logit, (hx, cx) = model((Variable(state.unsqueeze(0)), (hx, cx)))
            prob = F.softmax(logit, dim = 1).squeeze()
            # action = prob.numpy().argmax()
            action = prob.multinomial(num_samples = 1).data

            # show actions
            if c.show_action:
                print("value %.2f  \tpolicy: %s\n%s" 
                    %(value, c.action_name[action], prob.numpy().reshape(3,3)))

            # step the simulator
            env.render()
            state, reward, done, _ = env.step(action)
            rewards.append(reward)

            # stop if no hope
            recent_rewards.append(reward)
            if max(recent_rewards) <= 0:
                done = True

        # close simulator
        env.close()

        # print logs and show rewards
        print("=" * 50)
        print("total episode length %d, total rewards %.2f, mean rewards %.2f"
            %( episode_length, np.sum(rewards), np.mean(rewards)))

        if(c.show_rewards_curve):
            plt.xlabel("episode")
            plt.ylabel("reward")
            plt.plot(rewards, "x")
            plt.show()
