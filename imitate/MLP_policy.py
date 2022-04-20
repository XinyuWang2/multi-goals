import numpy as np
import torch
import torch.nn as nn

class MLPPolicy(nn.Module):

    def __init__(self,
        action_dim,
        obs_dim,
        n_layers, #number of hidden layers
        hidden_size,
        device,
        lr = 1e-4,
        training=True
        ):
        super().__init__()

        # init vars
        self.training = training
        self.device = device

        # network architecture
        self.mlp = nn.ModuleList([
            nn.Linear(obs_dim, hidden_size), #first hidden layer
            nn.ReLU()
        ])

        for h in range(n_layers - 1): #additional hidden layers
            self.mlp.append(nn.Linear(hidden_size, hidden_size))
            self.mlp.append(nn.ReLU())

        self.mlp.append(nn.Linear(hidden_size, action_dim)) #output layer, no activation function

        #loss and optimizer
        if self.training:
            self.loss_func = nn.CrossEntropyLoss()
            self.optimizer = torch.optim.Adam(self.parameters(), lr)

        self.to(device)


    def forward(self, x):

        for layer in self.mlp:
            x = layer(x)
        return x


    def save(self, filepath):
        torch.save(self.state_dict(), filepath)

    def restore(self, filepath):
        self.load_state_dict(torch.load(filepath))


    # query this policy with observation(s) to get selected action(s)
    def get_action(self, obs):
        if len(obs.shape)>1:
            observation = obs
        else:
            observation = obs[None]

        return self.forward(torch.Tensor(observation).to(self.device)).cpu().detach().numpy()

    # update/train this policy
    def update(self, observations, actions):
        assert self.training, 'Policy must be created with training = true in order to perform training updates...'
        self.optimizer.zero_grad()
        observations = torch.tensor(observations).to(torch.float32).to(self.device)
        gt_actions = torch.tensor(actions).to(self.device)
        pred_actions_scores = self.forward(observations)

        loss = self.loss_func(pred_actions_scores, gt_actions)
        # print("loss", loss)

        loss.backward()
        self.optimizer.step()

        return loss.cpu().detach().numpy()


