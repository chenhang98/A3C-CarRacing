from pyvirtualdisplay import Display
import torch.multiprocessing as mp
import torch
import os

from ACmodel import ActorCritic
from envs import create_env
from train import train
from test import test
import constants as c
import ACoptim

def part_pretrain(model, filename = "weights.pkl"):
    # load state dict
    pretrained_dict = torch.load(filename) 
    model_dict = model.state_dict()
    # filter useful parameters
    pretrained_dict['actor_linear.weight'] = model_dict['actor_linear.weight']
    pretrained_dict['actor_linear.bias'] = model_dict['actor_linear.bias']
    # load state_dict
    model.load_state_dict(pretrained_dict) 


if __name__ == '__main__':
    # open in viture screen
    display = Display(visible = 0, size = (1366, 768))
    display.start()

    # Specify thread number and gpu
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = ""

    # create simulator and shared model
    torch.manual_seed(c.seed)
    env = create_env(c.env_name)
    shared_model = ActorCritic(env.observation_space.shape[0], c.output_size)
    shared_model.share_memory()

    part_pretrain(shared_model)

    # # load model if required
    if c.load_model:
        shared_model.load_state_dict(torch.load("weights.pkl"))
        print("weights loaded sucessfully")

    # create optimizer for shared model 
    optimizer = ACoptim.SharedAdam(shared_model.parameters(), lr = c.lr)
    optimizer.share_memory()

    processes = []
    mp.set_start_method("spawn")
    counter = mp.Value('i', 0)  # 'i' means int
    lock = mp.Lock()

    # put test function to thread num_processes
    p = mp.Process(target = test, args = (c.num_processes, shared_model, counter))
    p.start()
    processes.append(p)

    # put train function to thread 0 ~ num_processes
    for rank in range(0, c.num_processes):
        p = mp.Process(target = train, args = (rank, shared_model, counter, lock, optimizer))
        p.start()
        processes.append(p)

    # wait util all train thread done 
    for p in processes:
        p.join()

    # close viture screen
    display.stop()
