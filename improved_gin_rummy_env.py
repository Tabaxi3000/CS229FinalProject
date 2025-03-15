class GinRummy:
    def __init__(self, seed=None, agent_type='dqn'):
        self.rng = np.random.RandomState(seed)
        self.agent_type = agent_type
        # Set reward scaling based on agent type
        if agent_type == 'mcts':
            self.alpha_gin = 3.0
            self.alpha_win = 2.0
            self.alpha_knock = 1.0
            self.beta_deadwood = 0.03
        else:  # DQN or REINFORCE
            self.alpha_gin = 1.5
            self.alpha_win = 1.0
            self.alpha_knock = 0.5
            self.beta_deadwood = 0.01
        self.reset()

    def step(self, action):
        if not self._is_valid(action):
            return self._get_state(), -50, True  # Significant penalty for invalid moves
            
        reward = 0
        done = False
        
        # Store previous deadwood for intermediate reward calculation
        prev_deadwood = self._get_deadwood(self.hands[self.current])
        
        if action == DRAW_STOCK:
            if not self.stock:
                return self._get_state(), 0, True
            card = self.stock.pop(0)
            self.hands[self.current].append(card)
            self.last_draw = card
            self.phase = 'discard'
            
            # Check if drawing completes a meld
            new_deadwood = self._get_deadwood(self.hands[self.current])
            if new_deadwood < prev_deadwood:
                reward += self.beta_deadwood * (prev_deadwood - new_deadwood)  # Meld formation bonus
            
        elif action == DRAW_DISCARD:
            if not self.discard:
                return self._get_state(), 0, True
            card = self.discard.pop()
            self.hands[self.current].append(card)
            self.last_draw = card
            self.phase = 'discard'
            
            # Bonus for drawing from discard to complete meld
            new_deadwood = self._get_deadwood(self.hands[self.current])
            if new_deadwood < prev_deadwood:
                reward += self.beta_deadwood * (prev_deadwood - new_deadwood) * 1.5  # Extra bonus for using discard
            
        elif DISCARD <= action < KNOCK:
            card = action - DISCARD
            if card not in self.hands[self.current]:
                return self._get_state(), -50, True
            self.hands[self.current].remove(card)
            self.discard.append(card)
            self.history.append(card)
            self.phase = 'draw'
            self.current = 1 - self.current
            
            # Deadwood reduction reward
            new_deadwood = self._get_deadwood(self.hands[self.current])
            if new_deadwood < prev_deadwood:
                reward += self.beta_deadwood * (prev_deadwood - new_deadwood)
            
        elif action in (KNOCK, GIN):
            deadwood = [self._get_deadwood(h) for h in self.hands]
            
            if action == GIN and deadwood[self.current] == 0:
                reward = self.alpha_gin * 25  # Gin bonus
                if deadwood[self.current] == 0:  # Perfect gin
                    reward *= 1.2  # Extra bonus for perfect gin
            elif action == KNOCK and deadwood[self.current] <= 10:
                if deadwood[self.current] < deadwood[1 - self.current]:
                    reward = self.alpha_win * 10  # Successful knock
                    # Low deadwood bonus
                    if deadwood[self.current] <= 5:
                        reward *= 1.1  # Extra bonus for very low deadwood
                else:
                    reward = -self.alpha_knock * 10  # Failed knock
            else:
                reward = -25  # Invalid knock/gin attempt
                
            done = True
            
        return self._get_state(), reward, done 