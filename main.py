from generator import Generator
from utils import * 
from classify import *
from Maddpg import *
import random
from maddpg_attack import *
import sys
import datetime


log_file = 'attack.log'
sys.stdout = Logger(log_file)
print(datetime.datetime.today())
state_dim = 200
action_dim = 100
alphas = 0
total_episodes = 40000
max_steps_per_episode = 1
model_name = 'VGG16'
print("Target Model : " + model_name)
G = Generator(100)
G = nn.DataParallel(G).cuda()
G = G.cuda()
ckp_G = torch.load('celeba_G.tar')['state_dict']
load_my_state_dict(G, ckp_G)
G.eval()


E = FaceNet(1000)
path_E = './weights/FaceNet.tar'
E = torch.nn.DataParallel(E).cuda()
ckp_E = torch.load(path_E)
E.load_state_dict(ckp_E['state_dict'], strict=False)
E.eval()

if model_name == "VGG16":
    T = VGG16(1000)
    path_T = './weights/VGG16.tar'
elif model_name == 'ResNet-152':
    T = IR152(1000)
    path_T = './weights/ResNet-152.tar'
elif model_name == "Face64":
    T = FaceNet64(1000)
    path_T = './weights/Face64.tar'


T = torch.nn.DataParallel(T).cuda()
ckp_T = torch.load(path_T)
T.load_state_dict(ckp_T['state_dict'], strict=False)
T.eval()

total = 0
cnt = 0
cnt5 = 0

identities = range(1000)
targets = random.sample(identities, 300)


for i in targets:
    t_flag = False
    t5_flag = False
    agent_mu = MADDPG(state_dim, action_dim)
    agent_var = MADDPG(state_dim, action_dim)
    recover_image = inversion(agent_mu, agent_var, G, T, alphas, label=i, total_episodes=total_episodes,max_steps_per_episode = 1,model_name=model_name,)
    os.makedirs("./attack/images/{}/success/".format(model_name), exist_ok=True)
    total += 1
    for rec_img in recover_image:
        _, outp = E(low2high(rec_img))
        e_pro = F.softmax(outp[0], dim=-1)
        t_idx = torch.argmax(e_pro)
        _, t5_idx = torch.topk(e_pro, 5)
        _, rwout = T(rec_img)
        rw_pro = F.softmax(rwout[0], dim = -1)
        rw_idx = torch.argmax(rw_pro)
        if t_idx == i and rw_idx == i:
            score = float(torch.mean(torch.diag(torch.index_select(F.softmax(outp, dim=-1).data, 1, torch.tensor([i]).cuda()))))
            save_image(rec_img, "./attack/images/{}/success/{}_{:.2f}.png".format(model_name, i, score), nrow=10)
            t_flag = True
        if i in t5_idx:
            t5_flag = True
    if t_flag:
        cnt += 1
    if t5_flag:
        cnt5 += 1
    acc = cnt / total
    acc5 = cnt5 / total
    print("Classes {}/{}, Accuracy : {:.3f}, Top-5 Accuracy : {:.3f}".format(total, 1000, acc, acc5))
