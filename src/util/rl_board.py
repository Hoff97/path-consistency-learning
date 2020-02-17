from torch.utils.tensorboard import SummaryWriter
import torch

class RLBoard:
    def __init__(self):
        self.writer = SummaryWriter()

    def log(self, iteration, rewards, loss, lens):
        avg_len = torch.mean(lens.float()).item()
        avg_reward = torch.mean(torch.sum(rewards, dim=1)).item()

        self.writer.add_scalar('reward', avg_reward, global_step=iteration)
        self.writer.add_scalar('loss', loss.item(), global_step=iteration)
        self.writer.add_scalar('len', avg_len, global_step=iteration)

        print(f'[{iteration+1}] Loss: {loss.item()}, Rewards: {avg_reward}, Len: {avg_len}')