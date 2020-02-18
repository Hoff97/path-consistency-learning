import torch
from torch.utils.tensorboard import SummaryWriter


class RLBoard:
    def __init__(self):
        self.writer = SummaryWriter()

    def log(self, iteration, rewards, loss, lens, exploration_rate):
        avg_len = torch.mean(lens.float()).item()
        avg_reward = torch.mean(torch.sum(rewards, dim=1)).item()

        self.writer.add_scalar('reward', avg_reward, global_step=iteration)
        self.writer.add_scalar('loss', loss.item(), global_step=iteration)
        self.writer.add_scalar('len', avg_len, global_step=iteration)
        self.writer.add_scalar('exploration_rate', exploration_rate, global_step=iteration)

        print(f'[{iteration+1}] Loss: {loss.item():.2f}, Rewards: {avg_reward:.2f}, Len: {avg_len:.0f}, Exp: {exploration_rate:.2f}')
