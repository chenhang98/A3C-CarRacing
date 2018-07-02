from torch.autograd import Variable
from collections import deque
import torch.nn.functional as F
import torch.optim as optim
import numpy as np 
import torch

from ACmodel import ActorCritic
from envs import create_env
import constants as c


def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(), shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad


def train(rank, shared_model, counter, lock, optimizer=None):
    torch.manual_seed(c.seed + rank)
    
    # create game simulator
    env = create_env(c.env_name)
    env.seed(c.seed + rank)

    # env.observation_space.shape is channel first here
    model = ActorCritic(env.observation_space.shape[0], c.output_size)

    if optimizer is None:
        optimizer = optim.Adam(shared_model.parameters(), lr=c.lr)

    # shift model to train mode
    model.train()

    # prepare game simulator
    state = env.reset()
    state = torch.from_numpy(state)
    done = True
    episode_length = 0

    # train
    while True:
        # Synchronize this thread's model with the shared model (method state_dict return the whole state of a model)
        model.load_state_dict(shared_model.state_dict())

        # reset cx, hx according to whether a round of game is done
        if done:
            cx = Variable(torch.zeros(1, 256))
            hx = Variable(torch.zeros(1, 256))
        else:
            cx = Variable(cx.data)
            hx = Variable(hx.data)

        # prepare values, log_pi, rewards, entropies loggers
        values = []
        log_probs = []
        rewards = []
        entropies = []

        # play and generate (s,r,a) pairs
        for step in range(c.num_steps):
            episode_length += 1

            # state.unsqueeze(0) to incrase dim, i.e [3,42,42] to [1,3,42,42]
            # logit is a linear output, i.e logit \in R 
            value, logit, (hx, cx) = model(( Variable(state.unsqueeze(0)), (hx, cx) ))	
            
            # prepare proility of pi and entropy (H)
            prob = F.softmax(logit, dim = 1)
            log_prob = F.log_softmax(logit, dim = 1)

            entropy = - (log_prob * prob).sum(1, keepdim = True)
            entropies.append(entropy)

            # sample an action by distribution prob
            action = prob.multinomial(num_samples = 1).data     # a tensor
            log_prob = log_prob.gather(1, Variable(action))     # select the log_prob[action]

            # step the simulator
            env.render()            # if not render, state may get a wrong frame (may be it's a bug of CarRacing)
            state, reward, done, _ = env.step(action.numpy())    # method item to transfer tensor(rank=1) to a float

            # end the game if waste too long time
            done = done or episode_length >= c.max_episode_length
            # normalize reward
            reward = max(min(reward, 1), -1)

            # thread lock
            with lock:
                counter.value += 1

            # reset the simulator if game was done
            if done:
                episode_length = 0
                state = env.reset()

            # replace state with new state, logging values, log_pi, r
            state = torch.from_numpy(state)
            values.append(value)
            log_probs.append(log_prob)
            rewards.append(reward)

            if done:
                break

        # R_t (the last) is zeros if termination else use model's predict
        R = torch.zeros(1, 1)
        if not done:
            value, _, _ = model((Variable(state.unsqueeze(0)), (hx, cx)))
            R = value.data

        values.append(Variable(R))  # V_t+1, used in GAE
        policy_loss = 0
        value_loss = 0

        R = Variable(R)
        gae = torch.zeros(1, 1)

        for i in reversed(range(len(rewards))):
            # estimate R_t by gamma * R_t+1 + r_t
            R = c.gamma * R + rewards[i]
            # advantage A_t = R_t - V_t
            advantage = R - values[i]
            
            # Loss_V = (0.5 *) \Sigma_t (R_t - V_t)^2
            value_loss = value_loss + 0.5 * advantage.pow(2)

            # Generalized Advantage Estimataion
            delta_t = rewards[i] + c.gamma * values[i + 1].data - values[i].data
            gae = gae * c.gamma * c.tau + delta_t

            # Loss_pi, minus for gradient descent
            policy_loss = policy_loss - log_probs[i] * Variable(gae) - c.entropy_coef * entropies[i]

        
        # back propagation:
        optimizer.zero_grad()
        (policy_loss + c.value_loss_coef * value_loss).backward()
        # clip gradient to avoid gradient explosion
        torch.nn.utils.clip_grad_norm_(model.parameters(), c.max_grad_norm)

        ensure_shared_grads(model, shared_model)
        optimizer.step()

