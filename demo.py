''' Demo SDK for LiveStreaming
    Author Dan Yang
    Time 2018-10-15
    For LiveStreaming Game'''
# import the env from pip
import LiveStreamingEnv.env as env
import LiveStreamingEnv.load_trace as load_trace
#import matplotlib.pyplot as plt
import time
import numpy as np
import random

from keras.models import Sequential
from keras.layers import Dense, Embedding, Dropout
from keras.layers import LSTM
from keras.optimizers import RMSprop
import gym


inputlen=8
# DQN AGENT
class DQNAgent:
    def __init__(self):
        self.memory = []
        self.gamma = 0.9  # decay rate
        self.epsilon = 1  # exploration
        self.epsilon_decay = .999
        self.epsilon_min = 0.2
        self.learning_rate = 0.0001
        self._build_model()
    
    def _build_model(self):
        model = Sequential()
        model.add(Embedding(8, 128))
        model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
        #model.add(LSTM(32))
        model.add(Dense(64,activation='relu'))
        #model.add(Dropout(0.5))
        model.add(Dense(8, activation='linear'))
        model.compile(loss='mse',
                      optimizer=RMSprop(lr=self.learning_rate),
                      metrics=['accuracy'])
        self.model = model
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state, reward_diss,reward_p):
        if np.random.rand() <= self.epsilon:
            return random.randrange(0,7) 
       # print(self.epsilon)
        '''if reward_p < 0:
            if np.random.rand()/3 <=self.epsilon:
                return random.randrange(0,7)
        if reward_p < -1:
            if np.random.rand()/4 <=self.epsilon:
                return random.randrange(0,5)
        if reward_p < -3:
            if np.random.rand()/3 <=self.epsilon:
                return random.randrange(0,3)
        if reward_p < 1:
            if np.random.rand()/2 <=self.epsilon:
                return random.randrange(0,7)
        if reward_diss > 1:
            if np.random.rand()*2/3 <=self.epsilon:
                return random.randrange(2,7)
        if reward_diss > 2:
            if np.random.rand()/2 <=self.epsilon:
                return random.randrange(0,7)'''
        act_values = self.model.predict(state)
        #print(act_values)
        return np.argmax(act_values[0])  # returns action
    
    def replay(self, batch_size):
        batches = min(batch_size, len(self.memory))
        batches = np.random.choice(len(self.memory), batches)
        for i in batches:
            state, action, reward, next_state, done = self.memory[i]
            if action == 0 or action == 1:
              target = reward*2.4
            if action == 2 or action == 3:
              target = reward*1.9
            if action == 4 or action == 5:
              target = reward*1.3
            if action == 6 or action == 7:
              target = reward
            if not done:
              target = -3
            target_f = self.model.predict(state)

            #target_f[0][0] = target_f[0][0]*0.95
            #target_f[0][1] = target_f[0][0]*0.95
            #target_f[0][2] = target_f[0][0]*0.95
            #target_f[0][3] = target_f[0][0]*0.95
            #target_f[0][4] = target_f[0][0]*0.95
            #target_f[0][5] = target_f[0][0]*0.95
            #target_f[0][6] = target_f[0][0]*0.95
            #target_f[0][7] = target_f[0][0]*0.95

            target_f[0][action] = target
            if reward>3:
              target_f[0]=[0,0,0,0,0,0,0,0]
              target_f[0][action]=3
            if reward>2.5 and action<6:
              target_f[0]=[0,0,0,0,0,0,0,0]
              target_f[0][action]=3
            if reward>2.3 and action<4:
              target_f[0]=[0,0,0,0,0,0,0,0]
              target_f[0][action]=3
            if reward>2.0 and action<2:
              target_f[0]=[0,0,0,0,0,0,0,0]
              target_f[0][action]=3
            print(target_f[0])
            history = self.model.fit(state, target_f, nb_epoch=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay 
        print("costc:%f,acc:%f"%(history.history["loss"][0],history.history["acc"][0]))

agent = DQNAgent()
# path setting
TRAIN_TRACES = './network_trace/'   #train trace path setting,
video_size_file = './video_trace/AsianCup_China_Uzbekistan/frame_trace_'      #video trace path setting,
video_size_file2 = './video_trace/YYF_2018_08_12/frame_trace_' 
video_size_file3 = './video_trace/Fengtimo_2018_11_3/frame_trace_' 
LogFile_Path = "./log/"                #log file trace path setting,
# Debug Mode: if True, You can see the debug info in the logfile
#             if False, no log ,but the training speed is high
DEBUG = False
DRAW = False
# load the trace
all_cooked_time, all_cooked_bw, all_file_names = load_trace.load_trace(TRAIN_TRACES)
#random_seed 
random_seed = 2
video_count = 0
FPS = 25
frame_time_len = 0.04
#init the environment
#setting one:
#     1,all_cooked_time : timestamp
#     2,all_cooked_bw   : throughput
#     3,all_cooked_rtt  : rtt
#     4,agent_id        : random_seed
#     5,logfile_path    : logfile_path
#     6,VIDEO_SIZE_FILE : Video Size File Path
#     7,Debug Setting   : Debug
t=0
while t<6:
  net_env = env.Environment(all_cooked_time=all_cooked_time,
  			  all_cooked_bw=all_cooked_bw,
  			  random_seed=random_seed,
  			  logfile_path=LogFile_Path,
  			  VIDEO_SIZE_FILE=video_size_file,
  			  Debug = DEBUG)
  net_env2 = env.Environment(all_cooked_time=all_cooked_time,
  			  all_cooked_bw=all_cooked_bw,
  			  random_seed=random_seed,
  			  logfile_path=LogFile_Path,
  			  VIDEO_SIZE_FILE=video_size_file2,
  			  Debug = DEBUG)
  net_env3 = env.Environment(all_cooked_time=all_cooked_time,
  			  all_cooked_bw=all_cooked_bw,
  			  random_seed=random_seed,
  			  logfile_path=LogFile_Path,
  			  VIDEO_SIZE_FILE=video_size_file3,
  			  Debug = DEBUG)
  
  use=[0,0,0,0,0,0,0,0]
  
  BIT_RATE      = [500.0,850.0,1200.0,1850.0] # kpbs
  TARGET_BUFFER = [2.0,3.0]   # seconds
  # ABR setting
  RESEVOIR = 0.5
  CUSHION  = 2
  reward_p_last=2
  
  cnt = 0
  # defalut setting
  last_bit_rate = 0
  bit_rate = 0
  target_buffer = 0
  # QOE setting
  reward_frame = 0
  reward_all = 0
  SMOOTH_PENALTY= 0.02 
  REBUF_PENALTY = 1.5 
  LANTENCY_PENALTY = 0.005 
  
  action=bit_rate*2+target_buffer
  
  # plot info
  idx = 0
  id_list = []
  bit_rate_record = []
  buffer_record = []
  throughput_record = []
  # plot the real time image
  if DRAW:
      fig = plt.figure()
      plt.ion()
      plt.xlabel("time")
      plt.axis('off')
  
  #network init
  
  
  time_interval_all=0
  send_data_size_all=0
  chunk_len_all=0
  rebuf_all=0
  buffer_size_all=0
  play_time_len_all=0
  end_delay_all=0
  
  last_state=np.array([[0,0,0,0,0,0,0,0]])
  count=0
  done=True
  
  reward_p=0
  
  while True:
          reward_frame = 0
          # input the train steps
          #if cnt > 5000:
              #plt.ioff()
          #    break
          #actions bit_rate  target_buffer
          # every steps to call the environment
          # time           : physical time 
          # time_interval  : time duration in this step
          # send_data_size : download frame data size in this step
          # chunk_len      : frame time len
          # rebuf          : rebuf time in this step          
          # buffer_size    : current client buffer_size in this step          
          # rtt            : current buffer  in this step          
          #play_time_len  : played time len  in this step          
          # end_delay      : end to end latency which means the (upload end timestamp - play end timestamp)
          # decision_flag  : Only in decision_flag is True ,you can choose the new actions, other time can't Becasuse the Gop is consist by the I frame and P frame. Only in I frame you can skip your frame
          # buffer_flag    : If the True which means the video is rebuffing , client buffer is rebuffing, no play the video
          # cdn_flag       : If the True cdn has no frame to get 
          # end_of_video   : If the True ,which means the video is over.
          if t%3==0:
            time, time_interval, send_data_size, chunk_len, rebuf, buffer_size, play_time_len,end_delay, cdn_newest_id, downlaod_id,decision_flag, buffer_flag,cdn_flag, end_of_video = net_env.get_video_frame(bit_rate,target_buffer)
          if t%3==1:
            time, time_interval, send_data_size, chunk_len, rebuf, buffer_size, play_time_len,end_delay, cdn_newest_id, downlaod_id,decision_flag, buffer_flag,cdn_flag, end_of_video = net_env2.get_video_frame(bit_rate,target_buffer)
          if t%3==2:
            time, time_interval, send_data_size, chunk_len, rebuf, buffer_size, play_time_len,end_delay, cdn_newest_id, downlaod_id,decision_flag, buffer_flag,cdn_flag, end_of_video = net_env3.get_video_frame(bit_rate,target_buffer)
          cnt += 1
  
          time_interval_all+=time_interval
          send_data_size_all+=send_data_size
          chunk_len_all+=chunk_len
          rebuf_all+=rebuf
          buffer_size_all+=buffer_size
          play_time_len_all+= play_time_len
          end_delay_all+=end_delay
  
          '''if time_interval != 0:
              # plot bit_rate 
              id_list.append(idx)
              idx += time_interval
              bit_rate_record.append(BIT_RATE[bit_rate])
              # plot buffer 
              buffer_record.append(buffer_size)
              # plot throughput 
              trace_idx = net_env.get_trace_id()
              print(trace_idx, idx,len(all_cooked_bw[trace_idx]))
              throughput_record.append(all_cooked_bw[trace_idx][int(idx/0.5)] * 1000 )'''
  
          if not cdn_flag:
              reward_frame = frame_time_len * float(BIT_RATE[bit_rate]) / 1000  - REBUF_PENALTY * rebuf - LANTENCY_PENALTY * end_delay
              reward_p+=reward_frame
          else:
              reward_frame = -(REBUF_PENALTY * rebuf)
              reward_p+=reward_frame
          if decision_flag or end_of_video:
              # reward formate = play_time * BIT_RATE - 4.3 * rebuf - 1.2 * end_delay
              reward_frame += -1 * SMOOTH_PENALTY * (abs(BIT_RATE[bit_rate] - BIT_RATE[last_bit_rate]) / 1000)
              # last_bit_rate
              last_bit_rate = bit_rate
              reward_p +=  -1 * SMOOTH_PENALTY * (abs(BIT_RATE[bit_rate] - BIT_RATE[last_bit_rate]) / 1000)
              reward_p = reward_p
              count=count+1
  
              state=np.array([[time_interval_all,send_data_size_all,chunk_len_all,rebuf_all,buffer_size_all,play_time_len_all,end_delay_all,reward_p]])
              print(reward_p)
  
              if reward_p > 0:
                  done=True
              else:
                  done=False
              agent.remember(last_state, action, reward_p, state, done)
  
              reward_diss = abs(reward_p - reward_p_last)
              action=agent.act(state,reward_diss,reward_p)  
              use[action]+=1
  
              last_state=state
              time_interval_all=0
              send_data_size_all=0
              chunk_len_all=0
              rebuf_all=0
              buffer_size_all=0
              play_time_len_all=0
              end_delay_all=0
              agent.replay(1) 
  
              bit_rate = int(action/2)
              target_buffer =action%2
              reward_p_last=reward_p
             # print("%d,%d,%d" %(bit_rate,target_buffer,action)) 
              reward_p=0
              
              # draw setting
              #if DRAW:
              #    ax = fig.add_subplot(311)
              #    plt.ylabel("BIT_RATE")
              #    plt.ylim(300,1000)
              #    plt.plot(id_list,bit_rate_record,'-r')
              
              #    ax = fig.add_subplot(312)
              #    plt.ylabel("Buffer_size")
              #    plt.ylim(0,7)
              #    plt.plot(id_list,buffer_record,'-b')
  
              #   ax = fig.add_subplot(313)
              #    plt.ylabel("throughput")
              #    plt.ylim(0,2500)
              #    plt.plot(id_list,throughput_record,'-g')
  
              #   plt.draw()
              #    plt.pause(0.01)
  
              
  
          # -------------------------------------------Your Algorithm ------------------------------------------- 
          # which part is the althgrothm part ,the buffer based , 
          # if the buffer is enough ,choose the high quality
          # if the buffer is danger, choose the low  quality
          # if there is no rebuf ,choose the low target_buffer
          '''if buffer_size < RESEVOIR:
              bit_rate = 0
          elif buffer_size >= RESEVOIR + CUSHION:
              bit_rate = 2
          elif buffer_size >= CUSHION + CUSHION:
              bit_rate = 3
          else: 
             bit_rate = 2'''
  
          # ------------------------------------------- End  ------------------------------------------- 
  
          reward_all += reward_frame
          if end_of_video:
              # Narrow the range of results
              break
              
  if DRAW:
      plt.show()
  t=t+1
print(reward_all)
agent.model.save("firstmodel.h5")
print("%d,%d,%d,%d,%d,%d,%d,%d"%(use[0],use[1],use[2],use[3],use[4],use[5],use[6],use[7]))
