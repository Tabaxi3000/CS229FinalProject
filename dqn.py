class DQN(nn.Module):
    def __init__(self, input_shape=(1, 4, 13), hidden_size=128):
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
        
        self.norm = nn.LayerNorm(64 * 4 * 13)
        self.lstm = nn.LSTM(52, 128, batch_first=True)
        
        # Strategy integration layers
        self.fc = nn.Sequential(
            nn.Linear(64 * 4 * 13 + 64 * 4 * 13 + 128, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, N_ACTIONS)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            nn.init.zeros_(m.bias)
        
    def forward(self, cards, history, mask=None):
        batch = cards.size(0)
        
        # Process hand
        if cards.dim() == 3:
            cards = cards.unsqueeze(1)
        hand_features = self.hand_conv(cards.float())
        hand_features = hand_features.view(batch, -1)
        hand_features = self.norm(hand_features)
        
        # Process discard history
        history = history.float()
        if history.dim() == 4:
            history = history.view(batch, -1, 52)
        discard_features = self.discard_conv(history[:, -1].view(batch, 1, 4, 13))
        discard_features = discard_features.view(batch, -1)
        
        # Process sequential information
        lstm_out, _ = self.lstm(history)
        lstm_out = lstm_out[:, -1]
        
        # Combine features
        combined = torch.cat([hand_features, discard_features, lstm_out], dim=1)
        q_values = self.fc(combined)
        
        # Apply action mask
        if mask is not None:
            if mask.dim() == 1:
                mask = mask.unsqueeze(0)
            q_values[~mask.bool()] = float('-inf')
            
        return q_values 