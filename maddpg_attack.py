from Maddpg import *
from copy import deepcopy
import torch
import math
from generator import Generator
from utils import * 
from classify import *
import os
from torchvision.utils import save_image
import logging
import sys


class Logger(object):
    def __init__(self, file_name="Default.log", stream=sys.stdout):
        self.terminal = stream
        self.log = open(file_name, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

def rreparameterize(mu, logvar):
    logvar = torch.exp(0.5 * logvar)
    z = torch.normal(mu, logvar)
    return z

def inversion(agent_mu, agent_var, G, T, alpha, label, total_episodes, max_steps_per_episode,model_name):
    print("Target Label : " + str(label))
    best_score = 0
    replay_buffer = ReplayBuffer(1e6)
    batch_size = 256
    best_images = []
    for episode in range(total_episodes):
        episode_reward_mu = 0
        episode_reward_var = 0
        y = torch.tensor([label]).cuda()
        mu = torch.randn((1,100)).cuda()
        log_var = torch.abs(torch.randn((1,100))).cuda()
        state_mu = deepcopy(mu)
        state_var = deepcopy(log_var)
        state = torch.cat((state_mu, state_var), dim=1)
        z = rreparameterize(mu, log_var)
        for step in range(max_steps_per_episode):
            action_mu = agent_mu.get_action(state.cpu().numpy())
            action_var = agent_var.get_action(state.cpu().numpy())
            a_mu = action_mu.detach().numpy()
            a_var = action_var.detach().numpy()
            mu = alpha * mu + (1 - alpha) * action_mu.clone().detach().cuda()
            log_var = alpha * log_var + (1 - alpha) * action_var.clone().detach().cuda()
            next_state_mu = deepcopy(mu.cpu().numpy())
            next_state_var = deepcopy(log_var.cpu().numpy())
            next_state = torch.cat((next_state_mu, next_state_var), dim=1)
            #update all
            z = rreparameterize(mu, log_var).cuda()
            #only action
            action_z = rreparameterize(action_mu.clone().detach().cuda(), action_var.clone().detach().cuda()).detach()
            #action var
            action_z_var = rreparameterize(state_mu.clone().detach().cuda(), log_var.cuda()).detach()
            #action mu
            action_z_mu = rreparameterize(mu.clone().detach().cuda(), state_var.cuda()).detach()

            state_image = G(z)
            action_z_image = G(action_z)
            action_z_mu_image = G(action_z_mu)
            action_z_var_image = G(action_z_var)

            _, state_output = T(state_image)
            _, action_z_output = T(action_z_image)
            _, action_mu_output = T(action_z_mu_image)
            _, action_var_output = T(action_z_var_image)
        
            criterion = nn.NLLLoss().cuda()
            Iden_Loss = 0
            loss_sdt = criterion(state_output, torch.tensor([label]).cuda())
            Iden_Loss = loss_sdt
            Iden_Loss = max(1e-7,Iden_Loss.cpu().detach().numpy())

            score_z_1 = float(torch.mean(torch.diag(torch.index_select(torch.log(F.softmax(state_output, dim=-1)).data, 1, y))))
            score_z_2 = float(torch.mean(torch.diag(torch.index_select(torch.log(F.softmax(action_z_output, dim=-1)).data, 1, y))))
            score_mu = float(torch.mean(torch.diag(torch.index_select(torch.log(F.softmax(action_mu_output, dim=-1)).data, 1, y))))
            score_var = float(torch.mean(torch.diag(torch.index_select(torch.log(F.softmax(action_var_output, dim=-1)).data, 1, y))))
            
            reward_mu =  1 * score_z_1 + 1 * score_z_2    + 1 * score_mu -  8 * Iden_Loss
            reward_var =  1 * score_z_1 + 1 * score_z_2   + 1 * score_var - 8 * Iden_Loss

            episode_reward_mu += reward_mu
            episode_reward_var += reward_var

            if step == max_steps_per_episode - 1 :
                    done = True
            else :
                    done = False

            replay_buffer.add(state.cpu().numpy(), a_mu, next_state.cpu().numpy(), reward_mu, done, state.cpu().numpy(), a_var, next_state.cpu().numpy(), reward_var, done)
            state = deepcopy(next_state)
            if len(replay_buffer) > batch_size:
                agent_mu.update(batch_size, replay_buffer,flag=1)
                agent_var.update(batch_size, replay_buffer,flag=0)
            
            if done:
                break
        
        test_images = []
        test_scores = []
        for _ in range(1):
            with torch.no_grad():
                mu_test = torch.randn((1,100)).cuda()
                var_test = torch.abs(torch.randn((1,100))).cuda()
                state_test = torch.cat((mu_test, var_test),dim=1)
                z_test = rreparameterize(mu_test, var_test).cuda()
                for _ in range(max_steps_per_episode):
                    action_mu_test = agent_mu.get_action(state_test.cpu().numpy())
                    action_var_test = agent_var.get_action(state_test.cpu().numpy())
                    mu_test = alpha * mu_test + (1 - alpha) * action_mu_test.clone().detach().cuda()
                    var_test = alpha * var_test + (1 - alpha) * action_var_test.clone().detach().cuda()
                    z_test = rreparameterize(mu_test, var_test).cuda()
                test_image = G(z_test).detach()
                test_images.append(test_image.cpu())
                _, test_output = T(test_image)
                test_score = float(torch.mean(torch.diag(torch.index_select(F.softmax(test_output, dim=-1).data, 1, y))))
            test_scores.append(test_score)
        mean_score = sum(test_scores) / len(test_scores)
        if mean_score >= best_score:
            best_score = mean_score
            best_images.append(torch.vstack(test_images))
            os.makedirs("./attack/images/{}/{}".format(model_name, label), exist_ok=True)
            save_image(best_images[-1], "./attack/{}/{}_{}.png".format(model_name, label, alpha), nrow=10)
            save_image(best_images[-1], "./attack/{}/{}/{}_{:.2f}.png".format(model_name, label, episode, best_score), nrow=10)
        if episode % 10000 == 0 or episode == total_episodes:
            print('Episodes {}/{}, Confidence score for the target model : {:.4f}'.format(episode, total_episodes,best_score))
    replay_buffer = None
    return best_images