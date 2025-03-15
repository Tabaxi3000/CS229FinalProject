class Net(nn.Module):
    def __init__(self, n_actions=110):
        super().__init__()
        
        # Hand evaluation stream
        self.hand_conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Discard pile analysis stream
        self.discard_conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Sequential processing
        self.lstm = nn.LSTM(52, 128, batch_first=True)
        
        # Strategy integration
        self.fc = nn.Sequential(
            nn.Linear(64 * 4 * 13 + 64 * 4 * 13 + 128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # Policy and value heads
        self.policy = nn.Linear(128, n_actions)
        self.value = nn.Linear(128, 1)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            nn.init.zeros_(m.bias)
            
    def forward(self, cards, history):
        batch = cards.size(0)
        
        # Process hand
        if cards.dim() == 3:
            cards = cards.unsqueeze(1)
        hand_features = self.hand_conv(cards.float())
        hand_features = hand_features.view(batch, -1)
        
        # Process discard history
        history = history.float()
        if history.dim() == 4:
            history = history.view(batch, -1, 52)
        discard_features = self.discard_conv(history[:, -1].view(batch, 1, 4, 13))
        discard_features = discard_features.view(batch, -1)
        
        # Process sequential information
        h_out, _ = self.lstm(history)
        h_out = h_out[:, -1]
        
        # Combine features
        x = torch.cat([hand_features, discard_features, h_out], dim=1)
        x = self.fc(x)
            
        # Policy and value outputs
        pi = F.softmax(self.policy(x), dim=1)
        v = self.value(x)
        
        return pi, v 