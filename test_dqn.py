from rl.dqn import DQN
import torch as th
from rl.gymrunner import GymRunner
from rl.models import LinearQNet
import gym

device = th.device('cuda' if th.cuda.is_available() else 'cpu')
env = gym.make('LunarLander-v2') 
model = LinearQNet(env.observation_space.shape[0], env.action_space.n)
agent = DQN(device, model, 'dqn_lunar', batch_size=256, train_ratio=4, target_update=5000)

runner = GymRunner(device, agent, 'LunarLander-v2')
for epoch in range(100000):
    datas = runner.run_steps_dqn()
    agent.train_batches(datas)
    if epoch%100 == 0:
        print(epoch, runner.get_avg_high())