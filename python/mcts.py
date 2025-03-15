import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from collections import defaultdict
import math
import random
from simple_evaluate import DRAW_STOCK, DRAW_DISCARD, DISCARD, KNOCK, GIN
import time

N_ACTIONS = 110
UCB_C = 1.41
GAMMA = 0.8
MAX_CHILDREN = 5
C_PUCT = 1.0

class Net(nn.Module):
    def __init__(self, n_actions=110):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.lstm = nn.LSTM(52, 128, batch_first=True)
        self.opp_net = nn.Linear(52, 64)
        
        self.fc = nn.Sequential(
            nn.Linear(64 * 4 * 13 + 128 + 64, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        self.policy = nn.Linear(128, n_actions)
        self.value = nn.Linear(128, 1)
        
    def forward(self, hand, history, opp_info=None):
        batch = hand.size(0)
        
        if hand.dim() == 3:
            hand = hand.unsqueeze(1)
            
        x = self.conv(hand.float())
        x = x.view(batch, -1)
        
        history = history.float()
        h_out, _ = self.lstm(history)
        h_out = h_out[:, -1]
        
        if opp_info is not None:
            opp = F.relu(self.opp_net(opp_info.float()))
        else:
            opp = torch.zeros(batch, 64, device=hand.device)
            
        x = torch.cat([x, h_out, opp], dim=1)
        x = self.fc(x)
            
        pi = F.softmax(self.policy(x), dim=1)
        v = torch.tanh(self.value(x))
        
        return pi, v
        
    def load(self, path, device=None):
        if not device:
            device = next(self.parameters()).device
        self.load_state_dict(torch.load(path, map_location=device))

class BeliefState:
    def __init__(self, n_particles=100):
        self.n = n_particles
        self.particles = []
        self.weights = np.ones(n_particles) / n_particles
        
    def update(self, obs):
        idx = np.random.choice(self.n, size=self.n, p=self.weights)
        self.particles = [self.particles[i] for i in idx]
        
        for i, p in enumerate(self.particles):
            self.weights[i] *= self._likelihood(p, obs)
            
        self.weights /= np.sum(self.weights)
        
    def _likelihood(self, particle, obs):
        return 1.0

class Node:
    def __init__(self, state, parent=None, move=None):
        self.state = state
        self.parent = parent
        self.move = move
        self.kids = {}
        self.visits = 0
        self.value = 0
        self.prior = 0
        self.moves = None
        self.is_leaf = True
        
    def expand(self, probs, moves=None):
        self.is_leaf = False
        self.moves = moves if moves else list(range(N_ACTIONS))
        
        if isinstance(probs, dict):
            p = torch.zeros(N_ACTIONS)
            for m, prob in probs.items():
                p[m] = prob
            probs = p
            
        mask = torch.zeros_like(probs)
        for m in self.moves:
            mask[m] = 1
            
        p = probs * mask
        if p.sum() > 0:
            p = p / p.sum()
        else:
            p = mask / mask.sum()
            
        for m in self.moves:
            if p[m] > 0:
                self.kids[m] = Node(None, self, m)
                self.kids[m].prior = float(p[m])
                
    def best_child(self):
        best_score = float('-inf')
        best_kid = None
        best_move = None
        
        for m, kid in self.kids.items():
            explore = UCB_C * kid.prior * math.sqrt(self.visits)
            exploit = kid.value / (kid.visits + 1)
            score = exploit + explore / (kid.visits + 1)
            
            if score > best_score:
                best_score = score
                best_kid = kid
                best_move = m
                
        return best_kid, best_move
        
    def update(self, val):
        self.visits += 1
        self.value += val

class MCTS:
    def __init__(self, net=None, n_sims=100, temp=1.0):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else
                                 "cuda" if torch.cuda.is_available() else "cpu")
            
        self.net = net
        self.n_sims = n_sims
        self.temp = temp
        
    def search(self, state):
        root = Node(state)
        
        state = {k: v.unsqueeze(0).to(self.device) for k, v in state.items()}
        
        with torch.no_grad():
            if self.net:
                pi, _ = self.net(state['hand_matrix'], state['discard_history'])
                pi = pi.squeeze()
            else:
                pi = torch.ones(N_ACTIONS, device=self.device) / N_ACTIONS
                
        moves = torch.nonzero(state['valid_actions_mask']).squeeze().tolist()
        if isinstance(moves, int):
            moves = [moves]
            
        root.expand(pi, moves)
        
        for _ in range(self.n_sims):
            node = root
            path = [node]
            
            while not node.is_leaf:
                kid, move = node.best_child()
                node = kid
                path.append(node)
                
            if node.is_leaf and len(path) < 50:
                s = node.state
                s = {k: v.unsqueeze(0).to(self.device) for k, v in s.items()}
                
                with torch.no_grad():
                    if self.net:
                        pi, val = self.net(s['hand_matrix'], s['discard_history'])
                        pi = pi.squeeze()
                        val = val.item()
                    else:
                        pi = torch.ones(N_ACTIONS, device=self.device) / N_ACTIONS
                        val = self._rollout(node.state)
                        
                valid = torch.nonzero(s['valid_actions_mask']).squeeze().tolist()
                if isinstance(valid, int):
                    valid = [valid]
                    
                node.expand(pi, valid)
                
            for node in reversed(path):
                node.update(val)
                val = -val
                
        visits = np.array([root.kids[m].visits if m in root.kids else 0 for m in moves])
        
        if self.temp == 0:
            best = moves[np.argmax(visits)]
            probs = {m: 1.0 if m == best else 0.0 for m in moves}
        else:
            visits = np.power(visits, 1.0 / self.temp)
            visits = visits / np.sum(visits)
            probs = {moves[i]: visits[i] for i in range(len(moves))}
            
        return probs
        
    def _rollout(self, state):
        steps = 0
        max_steps = 50
        
        while steps < max_steps:
            moves = torch.nonzero(state['valid_actions_mask']).squeeze().tolist()
            if isinstance(moves, int):
                moves = [moves]
            return np.random.uniform(-1, 1)
            steps += 1
            
        return 0.0

class MCTSAgent:
    def __init__(self, net=None, n_sims=100):
        self.net = net
        self.n_sims = n_sims
        self.mcts = MCTS(net, n_sims)
        
    def act(self, state, temp=1.0, time_limit=5.0):
        start = time.time()
        probs = self.mcts.search(state)
        
        if time.time() - start > time_limit:
            moves = torch.nonzero(state['valid_actions_mask']).squeeze().tolist()
            if isinstance(moves, int):
                moves = [moves]
            return random.choice(moves)
            
        moves = list(probs.keys())
        p = list(probs.values())
        
        if temp == 0:
            return moves[np.argmax(p)]
        else:
            return np.random.choice(moves, p=p)
            
    def save(self, path):
        if self.net:
            torch.save(self.net.state_dict(), path)
            
    def load(self, path):
        if self.net:
            self.net.load(path) 
