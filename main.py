import gym
import torch
from ddpg import DDPG
import numpy as np
from replaybuffer import ReplayBuffer

episodes_num = 200
batch_size = 256
REPLAY_BUFFER_SIZE = 5000


def main():
    #test or not(set False for training)
    test = True

    envName = 'HalfCheetah-v2'

    if torch.cuda.is_available():
        print("Device Count: {}".format(torch.cuda.device(0)))
        print("Device Name (first): {}".format(torch.cuda.get_device_name(0)))
    print("Environment Used: {}".format(envName))

    env = gym.make(envName, render_mode="human") 

    
    #make replay buffer object
    replay_buffer = ReplayBuffer()

    action_space = env.action_space.shape[0]
    state_space = env.observation_space.shape[0] 
    steps_limit = 800

    model = DDPG(state_space, action_space, replay_buffer)

    if test == True:
        model.load('models/DDPG')

    #train
    for i in range(episodes_num):
        print("--------Episode %d--------" % i)
        reward_per_episode = 0
        observation = env.reset()

        for j in range(steps_limit):
            if test == True:
                env.render()

            state = observation
         
            #Select action off-policy
            action = model.feed_forward_actor(np.expand_dims(state, axis=0))
        
            # Throw action to environment
            observation, reward, done, _, info = env.step(action)

            if len(state)==2:
                state = state[0]

            if len(observation)==2:
                observation = observation[0]   

            if test == False:     
                # For replay buffer. (s_t, a_t, s_t+1, r)
                model.add_experience(action, state, observation, reward, done)

                # Train actor/critic network
                if len(model.replay_buffer.buffer) > batch_size:    
                    model.train()

            reward_per_episode += reward

            if ((i+1)%20 == 0):
                model.save()

            if (done or j == steps_limit -1):
                print("Steps count: %d" % j)
                print("Total reward: %d" % reward_per_episode)

                break


if __name__ == '__main__':
    main()

