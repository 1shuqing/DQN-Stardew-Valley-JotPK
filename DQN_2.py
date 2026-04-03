import cv2
import os
import  time
import pynput
import numpy as np
import random
import logging
from collections import deque
from ultralytics import YOLO
import torch
import torch.nn as nn
import pickle   #存储经验池

REPLAY_SIZE = 10000     #回放池
INITIAL_EPSILON = 0.95   #初始探索率
FINAL_EPSILON = 0.15    #最终探索率
EPSILON_DECAY_STEP = 3000  #探索率下降所需步数
BATCH = 128
GAMMA = 0.96 #下一动作Q值的权重
L_R = 0.0002
UPDATE_STEP = 800   #目标网络更新频率
EPOCH = 800    #训练轮数
ACTION_SPACE = 14


class NET(nn.Module):   #神经网络
    def __init__(self,height,width,action_dim):
        super(NET,self).__init__()
        self.state_w =width
        self.state_h = height
        self.action_dim = action_dim


        # 卷积层
        self.cnn = nn.Sequential(
            nn.Conv2d(1,32,kernel_size=[8,8],stride=4,padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Conv2d(32,64,kernel_size=[4,4],stride=2,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=[3,3], stride=1,padding=1),  # 14→12
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Flatten()
        )

        #全连接层
        self.fc1 = nn.Linear(4096,512)
        self.fc2 = nn.Linear(512,256)
        self.fc3 = nn.Linear(256,self.action_dim)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    #前向传播
    def forward(self,screen_x):
        x =self.cnn(screen_x)
        x = x.reshape(x.size(0),-1)

        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class DQN():
    def __init__(self,height,width,action,model,log,load_replay_buffer=True):
        self.model = model
        self.log = log
        #log用
        self.loss_history = []
        self.reward_history = []

        self.target_net = NET(height,width,action)
        self.target_net.to("cuda")
        self.eval_net = NET(height,width,action)
        self.eval_net.to("cuda")

        self.replay_buffer = deque(maxlen=REPLAY_SIZE)
        self.epsilon = INITIAL_EPSILON

        #优化器、loss
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(),lr=L_R)
        #每200轮学习率降为原来的0.9
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,step_size=200,gamma=0.9)
        self.loss = nn.MSELoss()

        self.action_dim = action

        #保存经验池
        self.buffer_path = "replay_buffer.pkl"
        if load_replay_buffer:
            self.load_buffer()
        else:
            print("不使用已有经验")


    def save_buffer(self):
        with open(self.buffer_path,'wb') as f:
            pickle.dump(self.replay_buffer,f)
        print(f"经验池已保存：{len(self.replay_buffer)}条")

    def load_buffer(self):
        try:
            with open(self.buffer_path,'rb') as f:
                self.replay_buffer = pickle.load(f)
            print(f"已加载经验：{len(self.replay_buffer)}条")
        except:
            print("未找到经验池，新建空池")
            self.replay_buffer = deque(maxlen=REPLAY_SIZE)

    def sample_batch(self, batch_size):
        """
        优先级经验采样：奖励绝对值越大的经验，被采样的概率越高
        :param batch_size: 批次大小（对应全局BATCH）
        :return: 按优先级采样后的批次数据
        """
        # 计算每条经验的优先级（取奖励的绝对值）
        priorities = [abs(exp[2]) for exp in self.replay_buffer]
        total_priority = sum(priorities)

        # 处理边界：若所有优先级为0（经验池全是无奖励经验），则退化为随机采样
        if total_priority == 0:
            return random.sample(self.replay_buffer, batch_size)

        # 计算每条经验的采样概率
        probs = [p / total_priority for p in priorities]

        # 按概率采样（replace=False表示不重复采样，和原random.sample一致）
        indices = np.random.choice(len(self.replay_buffer), batch_size, p=probs, replace=False)
        batch = [self.replay_buffer[i] for i in indices]
        return batch

    def choose_action(self,screen_state):
        #epsilon贪心算法随机探索
        self.epsilon = max(FINAL_EPSILON, self.epsilon - (INITIAL_EPSILON - FINAL_EPSILON) / EPSILON_DECAY_STEP)
        if random.random() <= self.epsilon:
            a = random.randint(0,self.action_dim-1)
            print(f"随机探索!动作{a}")
            return a
        else:
            screen_state = screen_state.to("cuda", dtype=torch.float).unsqueeze(0)

            # 根据剩余时间和yolo检测结果判断Q值最大的动作
            Q = self.eval_net(screen_state)    #eval_net返回一个存有各动作Q值的张量

            self.last_q = Q.cpu().detach().numpy().flatten()
            return torch.argmax(Q).item()  # 检索结果转换成python int

    def replay_data(self,screen_state,action,reward,n_screen_state,done):
        #动作转one-hot编码
        oh_action = np.zeros(self.action_dim)
        oh_action[action] = 1.0

        #存储经验
        self.replay_buffer.append(
            (screen_state.cpu(),oh_action,reward,n_screen_state.cpu(),done)
        )

    def train(self):
        if len(self.replay_buffer) < BATCH:     #经验数<batch，不训练
            return

        #从回放池中取训练用经验
        # batch = self.sample_batch(BATCH)
        batch = random.sample(self.replay_buffer, BATCH)
        screen_batch, oh_action_batch, reward_batch, n_screen_batch, done_batch = zip(*batch)
        #拆分batch数据
        screen_batch = torch.stack(screen_batch).to("cuda", dtype=torch.float)
        # print("screen_batch维度：", screen_batch.shape)
        oh_action_batch = torch.tensor(np.array(oh_action_batch),dtype=torch.float).to("cuda")
        reward_batch = torch.tensor(reward_batch, dtype=torch.float).to("cuda")
        n_screen_batch = torch.stack(n_screen_batch).to("cuda", dtype=torch.float)

        done_batch =  torch.tensor(done_batch, dtype=torch.bool).to("cuda")

        # target_net 计算下一状态Q值
        n_Q_batch = self.target_net(n_screen_batch).detach()
        Q_target=[]
        for i in range(BATCH):
            if done_batch[i]:
                Q_target.append(reward_batch[i])
            else:
                Q_target.append(reward_batch[i]+GAMMA*torch.max(n_Q_batch[i]).item())

        # eval_net 计算当前状态Q值
        Q_batch = self.eval_net(screen_batch,)
        Q_eval = torch.sum(Q_batch*oh_action_batch,dim=1)

        #loss
        Q_target = torch.tensor(Q_target).to(dtype=torch.float).to("cuda").detach()
        loss = self.loss(Q_eval,Q_target)
        #反向传播
        self.optimizer.zero_grad()
        loss.backward()
        #防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(self.eval_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        return loss.item()

    def save_model(self):
        torch.save(self.target_net.state_dict(),self.model)

    def update_target(self):
        self.target_net.load_state_dict(self.eval_net.state_dict())

    def load_model(self):
        self.target_net.load_state_dict(torch.load(self.model,weights_only=True))



from screenshot import catch_screen
from restart import restart
import c_action


HEIGHT = 4
WIDTH =96
# action = 9
model_path = r"model"
log_path = r"logs/train.log"
log_dir = os.path.dirname(log_path)
if not os.path.exists(log_dir):
    os.mkdir(log_dir)


# target_steps = 0    #用于判断更新target_net
paused = True     #暂停训练





# 时间历史缓存（用于滑动平均）
time_history = deque(maxlen=2)
def detect_time():
    gray = cv2.cvtColor(catch_screen(376,90,560,8),cv2.COLOR_RGB2GRAY)
    state = cv2.resize(gray,(WIDTH,HEIGHT))
    time_left = WIDTH-1
    for i in range(len(state[2])):
        if state[2][i] <= 50:
            time_left = i
    #         print(i,state[2][i])
            break
    #归一化
    time_left = time_left/(WIDTH-1)
    #防止时间帧抖动
    time_history.append(time_left)
    time_left = np.mean(time_history)
    #防溢出
    time_left = max(0.0,min(1.0,float(time_left)))
    return time_left



#加载yolo
yolo_model = YOLO(r"runs/detect/train5/weights/best.pt")
def detect(screen=None):
    """
    为了减掉原detect_yolo和check方法中都有截图与yolo.predict的冗余
    增设该方法，避免重复检测导致的画面信息不一致
    return: player_box,enemy_boxes,alive,enemy_count
    """
    if screen is None:
        screen = catch_screen(300, 100, 650, 600)
        # 一次YOLO推理，复用结果
    results = yolo_model.predict(screen, conf=0.4, verbose=False)

    player_box = None
    enemy_boxes = []
    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls)
            if cls == 0 and player_box is None:
                player_box = box.xyxy.cpu().numpy()[0].tolist()
            elif cls == 1 and len(enemy_boxes) < 20:
                enemy_boxes.append(box.xyxy.cpu().numpy()[0].tolist())
    # 派生check需要的状态
    alive = player_box is not None
    enemy_count = len(enemy_boxes)
    return player_box, enemy_boxes, alive, enemy_count






def detect_yolo(screenshot=None):
    player_box, enemy_boxes, _, _ = detect(screenshot)
    max_size = max(650, 650)
    player_feat = [0.0] * 4 if player_box is None else [x / max_size for x in player_box]
    enemy_feat = []
    for i in range(20):
        if i < len(enemy_boxes):
            enemy_feat.extend([x / max_size for x in enemy_boxes[i]])
        else:
            enemy_feat.extend([0.0] * 4)
    enemy_count = len(enemy_boxes) / 20
    # 拼接特征
    yolo_feat = player_feat + enemy_feat + [enemy_count]
    yolo_state = torch.tensor(yolo_feat, dtype=torch.float).to("cuda")
    yolo_state = yolo_state.unsqueeze(0)     #增加维度适配net输入格式
    return yolo_state
    # results = yolo_model(screenshot,conf=0.4,verbose=False)
    # player_box = None
    # enemy_boxes = []
    # for r in results:
    #     boxes = r.boxes
    #     for box in boxes:
    #         cls = int(box.cls)
    #         if cls == 0 and player_box is None:
    #             player_box = box.xyxy.cpu().numpy()[0].tolist()
    #         elif cls == 1 and len(enemy_boxes)<20:
    #             enemy_boxes.append(box.xyxy.cpu().numpy()[0].tolist())
    #
    # max_size = max(650,650)
    # player_feat = [0.0]*4 if player_box is None else [x/max_size for x in player_box]
    # enemy_feat = []
    # for i in range(20):
    #     if i<len(enemy_boxes):
    #         enemy_feat.extend([x/max_size for x in enemy_boxes[i]])
    #     else :
    #         enemy_feat.extend([0.0]*4)
    # #归一
    # enemy_count = len(enemy_boxes)/20
    #
    # #拼接特征
    # yolo_feat = player_feat + enemy_feat +[enemy_count]
    # yolo_state = torch.tensor(yolo_feat,dtype=torch.float).to("cuda")
    # yolo_state = yolo_state.unsqueeze(0)    #增加维度适配net输入格式
    # return yolo_state


def get_game_state():
    screen = catch_screen(350, 100, 600, 600)
    gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
    screen_r = cv2.resize(gray, (128, 128),interpolation=cv2.INTER_NEAREST)
    screen_nor = screen_r / 255.0
    screen_tensor = torch.from_numpy(screen_nor).to(dtype=torch.float).cuda()
    screen_tensor = screen_tensor.unsqueeze(0)

    # 传入已有的截图，避免detect_yolo重复截图
    # yolo_tensor = detect_yolo(screen)
    #
    # time_tensor = torch.tensor([detect_time()], dtype=torch.float).unsqueeze(0).cuda()

    return screen_tensor, screen





def move(action):
    if action == 0:
        c_action.up()
        time.sleep(0.1)
    elif action == 1:
        c_action.down()
        time.sleep(0.1)
    elif action == 2:
        c_action.left()
        time.sleep(0.1)
    elif action == 3:
        c_action.right()
        time.sleep(0.1)
    elif action == 4:
        c_action.shootup()
        time.sleep(0.1)
    elif action == 5:
        c_action.shootdown()
        time.sleep(0.1)
    elif action == 6:
        c_action.shootleft()
        time.sleep(0.1)
    elif action == 7:
        c_action.shootright()
        time.sleep(0.1)
    elif action == 8:
        c_action.tools()
        time.sleep(0.1)
    elif action == 9:
        time.sleep(0.1)

    elif action == 10:  #射左上
        c_action.shootup()
        c_action.shootleft()
        time.sleep(0.1)
    elif action == 11:  #射右上
        c_action.shootup()
        c_action.shootright()
        time.sleep(0.1)
    elif action == 12:  #射左下
        c_action.shootleft()
        c_action.shootdown()
        time.sleep(0.1)
    elif action == 13:  #射右下
        c_action.shootright()
        c_action.shootdown()
        time.sleep(0.1)


TIMELEFT = False
#检测游戏状态
def check(screen=None):
    player_box, enemy_boxes, alive, enemy_count = detect(screen)
    time_left = detect_time()
    # 通关条件：无敌人 + 时间耗尽
    global TIMELEFT
    if time_left<0.1:
        TIMELEFT = True    #防止时间刷新而finish状态检测错误
    finish = (enemy_count == 0) and TIMELEFT
    player_pos = None
    if player_box is not None:
        # 计算玩家框中心坐标并归一化
        player_x = (player_box[0] + player_box[2]) / 2
        player_y = (player_box[1] + player_box[3]) / 2
        player_pos = [player_x, player_y]
    return alive, finish, enemy_count, player_pos, enemy_boxes,time_left
    # screen = catch_screen(300, 100, 650, 600)
    # results = yolo_model.predict(screen,conf=0.3,verbose=False)
    # alive = False
    # enemy_count = 0
    # player_box = None
    # enemy_boxes = []
    # if player_feat is not None and len(player_feat) >= 4:
    #     x1, y1, x2, y2 = player_feat[:4]
    #     player_pos = [x1, y1, x2, y2]
    # for r in results:
    #     boxes = r.boxes
    #     for box in boxes:
    #         cls = int(box.cls)
    #         if cls == 0:
    #             alive = True
    #         elif cls == 1:
    #             enemy_count += 1
    # time_left = detect_time()
    # #通关
    # finish = (enemy_count == 0) and (time_left<0.1)
    # return alive ,finish ,enemy_count






def c_reward(p_enemy_count,enemy_count,alive,finish,action,player_box,enemy_boxes,time_left):
    reward = 0.0
    #消灭敌人
    if alive and enemy_count < p_enemy_count:
        reward += 1*(p_enemy_count - enemy_count)
        # print(f"消灭{p_enemy_count - enemy_count}个敌人+{1.0*(p_enemy_count - enemy_count):.1f}分")
    #死亡
    if not alive:
        reward -= 3
        # print(f"死亡-5分")
    #通关
    if finish:
        reward += 2
        print("通关+2分")
    #场上无敌人
    # if alive and enemy_count==0 and time_left<1 == 0:
    #     reward+= 0.3
        # print("清场+0.3分")
    # 原地不动惩罚
    if alive and action == 9 :
        reward -= 0.05
        # print("原地不动-0.05")
    # 7. 远离敌人奖励 靠近敌人惩罚
    if alive and player_box is not None and len(enemy_boxes) > 0:
        enemy_dist = c_distance(player_box, enemy_boxes)
        if enemy_dist < 0.2:
            reward -= 0.01
            # print(f"{enemy_dist}靠近敌人 -0.03分")
        elif enemy_dist > 0.55:
            reward += 0.01
            # print(f"{enemy_dist}远离敌人 +0.03分")
    #存活奖励
    # if alive and time_left<1:
    #     reward += 0.05
        # print("存活+0.05分")

    return reward


#计算玩家和最近敌人距离
def c_distance(player_box, enemy_boxes):
    if not enemy_boxes or not player_box:
        return 1.0
    min_dist = 1e9
    max_screen_dist = np.sqrt(650 ** 2 + 500 ** 2)
    player_x,player_y = player_box
    for box in enemy_boxes:
        enemy_x = (box[0] + box[2])/2
        enemy_y = (box[1] + box[3])/2
        dist = np.sqrt((player_x - enemy_x)**2 + (player_y - enemy_y)**2)
        min_dist = min(min_dist, dist)

    min_dist = min_dist/max_screen_dist
    return min_dist






#游戏窗口在屏幕左上角时的时间读条位置
x,y,width,height = 376,90,560,8

if __name__ == '__main__':



    print("----训练配置选择 ")
    # 1. 选择是否加载已有经验池
    load_replay_buffer_choice = input("是否加载已有经验池？(y/n，默认y)：").strip().lower()
    load_replay_buffer = True if load_replay_buffer_choice in ["y", "", "yes"] else False

    # 2. 选择是否加载已有模型
    load_model_choice = input("是否加载已有模型？(y/n，默认n)：").strip().lower()
    load_model = True if load_model_choice in ["y", "yes"] else False

    logging.basicConfig(
        level=logging.INFO,
        filename=log_path,
        format='%(asctime)s - %(message)s',
        datefmt = '%m-%d %H:%M:%S'
    )
    model = DQN(128,128,action=ACTION_SPACE,model=model_path,log=log_path,load_replay_buffer=load_replay_buffer)
    if load_model:
        model.load_model()

    print("开始填充经验池...")
    while len(model.replay_buffer) < BATCH:  # 填充经验池直到能启动train
        # timeleft 减少垃圾时间
        time_left = 1
        while time_left>=1:
            screen_state, screen = get_game_state()
            _,_,enemy_count,_,_,time_left = check(screen)
        screen_state,screen = get_game_state()
        _, _, enemy_count, _, _, time_left = check(screen)
        action = random.randint(0, ACTION_SPACE-1)  # 纯随机探索
        move(action)
        n_screen_state,n_screen = get_game_state()
        alive, finish, n_enemy_count,player_box,enemy_boxes,n_time_left = check(n_screen)
        done = not alive or finish
        reward = c_reward(enemy_count, n_enemy_count, alive, finish,action,player_box,enemy_boxes,n_time_left)
        model.replay_data(screen_state, action, reward, n_screen_state, done)
        if done:
            c_action.release_all_key()
            restart()
    print("经验池填充完成，开始训练")

    # 用于判断更新target_net
    target_steps = 0
    #记录最好成绩
    episode_rewards = []    # 存储每轮的总奖励
    best_avg_reward = -float('inf')
    best_model_path = "best_model"
    best_reward = -float('inf') #初始为无限小
    for e in range(EPOCH):
        # timeleft 减少垃圾时间
        time_left = 1
        while time_left >= 1:
            screen_state,screen = get_game_state()
            _,_,enemy_count,_,_,time_left = check(screen)
        screen_state, screen = get_game_state()
        _, _, enemy_count, _, _, time_left = check(screen)
        done = False        #判断游戏是否结束
        all_reward = 0.0
        total_loss = 0
        train_count = 0
        steps = 0   #epoch内步数
        # stop = 0
        # emergence_break = 0     #用于防止重复计算reward




        last_time = time.time()     #记录每epoch训练用时
        logging.info(f"----训练{e+1}/{EPOCH}轮")
        print(f"---训练{e+1}/{EPOCH}轮")

        while not done:

            target_steps += 1
            steps += 1
            #选动作
            action = model.choose_action(screen_state)
            #做动作
            move(action)
            #获取新状态
            n_screen_state,n_screen = get_game_state()
            #游戏是否结束
            alive,finish,n_enemy_count,player_box,enemy_boxes,n_time_left = check(n_screen)
            done = not alive or finish
            if done:
                TIMELEFT= False

            #计算奖励分数
            reward = c_reward(enemy_count,n_enemy_count,alive,finish,action,player_box,enemy_boxes,n_time_left)
            all_reward += reward
            #存储回放经验
            model.replay_data(screen_state,action,reward,n_screen_state,done)
            #训练网络
            loss = model.train()
            if loss is not None:
                total_loss += loss
                train_count +=1
            #更新target_net
            if target_steps % UPDATE_STEP == 0:
                model.update_target()
                model.save_model()
                model.save_buffer()
                logging.info(f"[TargetNet] 更新目标网络 | step={target_steps}")
                print(f"[TargetNet] 更新目标网络 | step={target_steps}")

            #状态刷新
            screen_state = n_screen_state
            enemy_count = n_enemy_count

        if target_steps != 0:
            print("当前循环耗时{:.1f}秒".format(time.time() - last_time))
        episode_rewards.append(all_reward)  # 记录当前轮总奖励
        # 计算最近10轮平均奖励（不足10轮则取已有轮次）
        if len(episode_rewards) >= 10:
            avg_reward = np.mean(episode_rewards[-10:])
        else:
            avg_reward = np.mean(episode_rewards)

        # 更新最优平均奖励并保存最优模型
        if avg_reward > best_avg_reward:
            best_avg_reward = avg_reward
            # 保存eval_net（而非target_net）的最优状态
            torch.save(model.eval_net.state_dict(), best_model_path)
            logging.info(f"[最优模型] 保存最近10轮平均奖励={best_avg_reward:.2f}的模型")
            print(f"[最优模型] 保存最近10轮平均奖励={best_avg_reward:.2f}的模型")
        # 奖励下降50%触发回滚
        elif len(episode_rewards) >= 50 and avg_reward < best_avg_reward * 0.5 and best_avg_reward != -float('inf'):
            model.eval_net.load_state_dict(torch.load(best_model_path))
            # 同步更新target_net（保证双网络一致）
            model.update_target()
            # 临时提高探索率（不超过初始值）
            # model.epsilon = min(model.epsilon + 0.1, INITIAL_EPSILON)
            log_info = (
                f"Epoch {e + 1}: 模型性能下降，回滚到最优模型 | "
                f"当前平均奖励={avg_reward:.2f} \n "
                f"最优平均奖励={best_avg_reward:.2f}  "
            )
            logging.info(log_info)
            print(log_info)
        #日志记录
        avg_loss = total_loss/train_count if train_count>0 else 0
        avg_reward = all_reward/steps if steps>0 else 0
        log_info = (
            f"Epoch {e + 1:2d} | "
            f"总得分={all_reward:6.1f} | "
            f"平均每步奖励={avg_reward:5.2f} | "
            f"平均loss={avg_loss:7.4f} \n "
            f"探索率ε={model.epsilon:5.3f} | "
            f"存活步数={steps}"
        )
        logging.info(log_info)
        print(log_info)
        #Q值记录
        if hasattr(model,'last_q'):
            q_log = f"[Q]：{np.round(model.last_q,2)}"
            logging.info(q_log)
            print(q_log)

        if all_reward > best_reward:
            best_reward = all_reward
            model.update_target()
            model.save_model()
            logging.info(f"[新]保存得分为{best_reward}的模型")
            print(f"保存得分为{best_reward:.1f}的模型")
            model.save_buffer()
        if train_count >0:
            model.scheduler.step()
        c_action.release_all_key()
        restart()












