import d4rl
import numpy as np
import copy

class d4rl_dataset():
    def __init__(self,env):
        self.max_epi_len = 1000
        # dataset = d4rl.qlearning_dataset(env)
        self.dataset = env.get_dataset()

        self.len = self.dataset['observations'].shape[0]
        print("[Data length] : ",self.len)
        self.gamma = 0.99
        self.add_Return()


    def add_Return(self):
        Return_buffer = []
        tmp_return = 0
        for i in reversed(range(self.len)):
            if self.dataset['terminals'][i] or self.dataset['timeouts'][i] or i == (self.len-1):
                tmp_return = self.dataset["rewards"][i]
                Return_buffer.append(tmp_return)
            else:
                tmp_return = self.dataset["rewards"][i] + self.gamma*tmp_return
                Return_buffer.append(tmp_return)

        Return_buffer = copy.deepcopy(np.flip(np.array(Return_buffer,dtype=np.float32)))
        self.dataset["Return"] = Return_buffer


    def get_data(self,batch_size=256):
        idx = np.random.choice(self.len, batch_size)
        return self.dataset['observations'][idx], self.dataset['actions'][idx], self.dataset['rewards'][idx], self.dataset['next_observations'][idx], self.dataset['terminals'][idx], self.dataset['Return'][idx]



def Eval(env,agent,epoch_count,args):
    epi_length = env.spec.max_episode_steps
    action_max = env.action_space.high[0]
    epi_return = []  # 나중에 success까지 포함해야할듯
    agent.pi.eval()
    for eval_epi in range(args.eval_num):
      state = env.reset()
      total_reward = 0
      for step in range(epi_length):
        # if eval_epi == (eval_num-1):
        #     env.render()
        action = agent.select_action(state.reshape([1,-1]),eval=True)
        state, rwd, done, _ = env.step(action * action_max)
        total_reward += rwd
        if done:
          break
      epi_return.append(total_reward)
    print("==================[Eval]====================")
    print("Epoch        : ", epoch_count)
    print("Mean return  : ", np.mean(epi_return), "Min return", np.min(epi_return), "Max return", np.max(epi_return))
    return np.mean(epi_return), np.min(epi_return), np.max(epi_return)

class logger():
    def __init__(self,task_name,path,iter=0):
        self.path = path + '/' + task_name + '-' + str(iter)
        f = open(self.path + ".txt", 'w')
        f.close()

    def write_sep(self,stage):
        f = open(self.path + ".txt", 'a')
        f.write("====="+stage+"======")
        f.write("\n")
        f.close()

    def write_eval(self, epoch, mean, min, max):
        f = open(self.path + ".txt", 'a')
        f.write("Epoch")
        f.write(" ")
        f.write(str(epoch))
        f.write("     ")
        f.write(str(round(mean, 1)))
        f.write(" ")
        f.write(str(round(min, 1)))
        f.write(" ")
        f.write(str(round(max, 1)))
        f.write(" ")
        f.write("\n")
        f.close()

    def write_q(self, epoch, q1, q2):
        f = open(self.path + ".txt", 'a')
        f.write("Epoch")
        f.write(" ")
        f.write(str(epoch))
        f.write("     ")
        f.write(str(np.round(q1, 1)))
        f.write(" ")
        f.write(str(np.round(q2, 1)))
        f.write(" ")
        f.write("\n")
        f.close()




