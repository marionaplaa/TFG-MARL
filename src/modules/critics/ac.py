import torch as th
import torch.nn as nn
import torch.nn.functional as F


class ACCritic(nn.Module):
    def __init__(self, scheme, args):
        super(ACCritic, self).__init__()

        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents

        input_shape = self._get_input_shape(scheme)
        self.output_type = "v"

        # Set up network layers
        self.fc1 = nn.Linear(input_shape, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc3 = nn.Linear(args.hidden_dim, 1)

    def forward(self, batch, t=None):
        inputs, bs, max_t = self._build_inputs(batch, t=t)
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        q = self.fc3(x)
        return q

    # def _build_inputs(self, batch, t=None):
    #     bs = batch.batch_size
    #     max_t = batch.max_seq_length if t is None else 1
    #     ts = slice(None) if t is None else slice(t, t+1)
    #     inputs = []
    #     # observations
    #     inputs.append(batch["obs"][:, ts])

    #     inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).unsqueeze(0).expand(bs, max_t, -1, -1))

    #     inputs = th.cat(inputs, dim=-1)
    #     return inputs, bs, max_t

    # def _get_input_shape(self, scheme):
    #     # observations
    #     input_shape = scheme["obs"]["vshape"]
    #     # agent id
    #     input_shape += self.n_agents
    #     return input_shape
    def _build_inputs(self, batch, t=None):
        bs = batch.batch_size
        max_t = batch.max_seq_length if t is None else 1
        ts = slice(None) if t is None else slice(t, t+1)
        inputs = []
        
        # observations
        inputs.append(batch["obs"][:, ts])
        
        # last actions
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, 0:1]).view(bs, max_t, self.n_agents, -1))
            elif isinstance(t, int):
                inputs.append(batch["actions_onehot"][:, slice(t-1, t)].view(bs, max_t, self.n_agents, -1))
            else:
                last_actions = th.cat([th.zeros_like(batch["actions_onehot"][:, 0:1]), batch["actions_onehot"][:, :-1]], dim=1)
                inputs.append(last_actions.view(bs, max_t, self.n_agents, -1))

        # target
        if self.args.obs_target:
            obs = batch["obs"][:, ts]           # [bs, max_t, n_agents, obs_dim]
            leader_obs = obs[:, :, 0, :]        # [bs, max_t, obs_dim]
            obs_dim = leader_obs.shape[-1]

            targets = th.full((bs, max_t, 2), -1.0, device=batch.device)
            for b in range(bs):
                for step in range(max_t):
                    for i in range(0, obs_dim - 1, 3):
                        x = leader_obs[b, step, i]
                        y = leader_obs[b, step, i + 1]
                        if x >= 0 and y >= 0:
                            targets[b, step, 0] = x
                            targets[b, step, 1] = y
                            break

            target_tensor = targets.unsqueeze(2).expand(bs, max_t, self.n_agents, 2)
            inputs.append(target_tensor)

        # agent id
        inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).unsqueeze(0).expand(bs, max_t, -1, -1))
        
        inputs = th.cat(inputs, dim=-1)
        return inputs, bs, max_t

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_target:
            input_shape += 2
        input_shape += self.n_agents
        return input_shape