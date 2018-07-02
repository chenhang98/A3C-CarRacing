from torch.autograd import Variable
from collections import deque
import torch.nn.functional as F
import torch
import time
import os

from ACmodel import ActorCritic
from envs import create_env
import constants as c


class Logger():
    def __init__(self, path, mode = 'w'):
        self.path = path
        self.mode = mode
        self.item = None    # save a piece of log temporarily
        self.titl = "Time, num steps, FPS, episode reward, episode length"
        self.form = ''

    def title(self, titl):
        # if you need to call title, surely you want to rewrite the log
        self.mode = 'w'
        self.titl = titl
        with open(self.path, self.mode) as f:
            f.write(titl + '\n')

    def format(self, form):
        for (t, f) in zip(self.titl.split(", "), form.split(", ")):
            self.form += t + " " + f + ", "
        # remove the last funny ","
        self.form = self.form[:-2]

    def show(self):
        print(self.item)

    def add(self, *args):
        # generate a piece of log
        self.item = self.form %args
        with open(self.path, 'a') as f:
            f.write(str(args)[1:-1] + '\n')

    def clean(self):
        os.environment("rm %s" %path)



def test(rank, shared_model, counter):
    # create logger
    logger = Logger("log.csv")
    if not c.load_model:
        logger.title("Time, num steps, FPS, episode reward, episode length" )
    logger.format("%s, %i, %.0f, %.2f, %i")

    # Disabling gradient calculation, to reduce memory cost
    with torch.no_grad():

        # create simulator and build model
        torch.manual_seed(c.seed + rank)

        env = create_env(c.env_name)
        env.seed(c.seed + rank)

        model = ActorCritic(env.observation_space.shape[0], c.output_size)

        # shift model to eval mode
        model.eval()

        state = env.reset()
        state = torch.from_numpy(state)
        reward_sum = 0
        done = True

        start_time = time.time()

        # a quick hack to prevent stucking (if rewards is always negative in 200 frames, break)
        recent_rewards = deque(maxlen = 200)
        recent_rewards.append(1)
        episode_length = 0

        while True:
            episode_length += 1

            # copy the newest weights from shared model
            if done:
                model.load_state_dict(shared_model.state_dict())
                cx = Variable(torch.zeros(1, 256))
                hx = Variable(torch.zeros(1, 256))
            else:
                cx = Variable(cx.data)
                hx = Variable(hx.data)

            # get predict action from the model
            value, logit, (hx, cx) = model((Variable(state.unsqueeze(0)), (hx, cx)))
            prob = F.softmax(logit, dim = 1)
            # action = prob.argmax().item()
            action = prob.multinomial(num_samples = 1).data

            # step the simulator and render
            env.render()
            state, reward, done, _ = env.step(action[0, 0])
            done = done or episode_length >= c.max_episode_length
            reward_sum += reward

            # a quick hack to prevent stucking
            recent_rewards.append(reward)
            if max(recent_rewards) <= 0:
                done = True
                print("test stucking")

            # print log and reset state
            if done:
                # print log
                logger.add( time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start_time)),
                            counter.value,
                            counter.value / (time.time() - start_time),
                            reward_sum, 
                            episode_length )
                print("-" * 50)
                logger.show()

                # save weights
                torch.save(model.state_dict(), 'weights.pkl')
                print("weights saved successfully\n")

                # init
                reward_sum = 0
                episode_length = 0
                state = env.reset()

                # sleep test_interval seconds before next eval (notice that test will be a independent thread)
                time.sleep(c.test_interval)

            # tranfer state from array to tensor
            state = torch.from_numpy(state)
