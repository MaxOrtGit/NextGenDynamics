
import torch
import torch.nn as nn

# Assuming these are imported from your RL library (e.g., skrl)
from skrl.models.torch import Model, GaussianMixin, DeterministicMixin
from skrl.utils.spaces.torch import unflatten_tensorized_space

def get_coordinate_grid(batch_size, h, w, device):
    # --- Generate Coordinate Channels ---
    # Create linear gradients from -1 to 1
    x_range = torch.linspace(-1, 1, steps=w, device=device)
    y_range = torch.linspace(-1, 1, steps=h, device=device)
    
    # Create meshgrid (Y, X)
    # indexing='ij' means first dim is rows (Y), second is cols (X)
    y_grid, x_grid = torch.meshgrid(y_range, x_range, indexing='ij')
    
    # Expand to batch size: (Batch, 1, H, W)
    x_channel = x_grid.unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1, -1)
    y_channel = y_grid.unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1, -1)

    return x_channel, y_channel

class HeightMapEncoder(nn.Module):
    def __init__(self, input_channels=1):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Conv2d(input_channels + 2, 8, kernel_size=5, stride=2, padding=0), # 25x25 -> 11x11, 968
            nn.ELU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=0), # 11x11 -> 5x5, 400
            nn.ELU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=0), # 5x5 -> 2x2, 64
            nn.ELU(),
            nn.Flatten(), # 64
        )

    def forward(self, x):
        batch_size, _,  h, w = x.shape
        device = x.device
        x_channel, y_channel = get_coordinate_grid(batch_size, h, w, device)
        
        return self.net(torch.cat([x, x_channel, y_channel], dim=1))


class NavigationMapEncoder(nn.Module):
    def __init__(self, input_channels=3, ):
        super().__init__()

        self.net = nn.Sequential( #33x33
            nn.Conv2d(input_channels + 2, 8, kernel_size=9, stride=2, padding=0), # 33x33 -> 13x13
            nn.ELU(),
            nn.Conv2d(8, 16, kernel_size=5, stride=2, padding=0), # 13x13 -> 5x5
            nn.ELU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=0), # 5x5 -> 2x2
            nn.ELU(),
            nn.Flatten(),
        )

    def forward(self, x):
        # x shape: (Batch, 3, H, W) -> Staleness, Ray Cast Density, Height

        batch_size, _, h, w = x.shape
        device = x.device
        
        x_channel, y_channel = get_coordinate_grid(batch_size, h, w, device)
        
        # Concatenate: (Batch, 3, H, W) + (Batch, 1, H, W) + (Batch, 1, H, W) -> (Batch, 5, H, W)
        x_with_coords = torch.cat([x, x_channel, y_channel], dim=1)
        
        return self.net(x_with_coords)

class SharedRecurrentModel(GaussianMixin,DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(
            self,
            clip_actions=False,
            clip_log_std=True,
            min_log_std=-20.0,
            max_log_std=2.0,
            reduction="sum",
            role="policy",
        )
        DeterministicMixin.__init__(self, clip_actions=False, role="value")
        
        self.height_encoder = HeightMapEncoder()
        self.nav_encoder = NavigationMapEncoder()
        self.net_container = nn.Sequential(
            nn.LazyLinear(out_features=512),
            nn.ELU(),
            nn.LazyLinear(out_features=256),
            nn.ELU(),
            nn.LazyLinear(out_features=128),
            nn.ELU(),
        )
        self.policy_layer = nn.LazyLinear(out_features=self.num_actions)
        self.log_std_parameter = nn.Parameter(torch.full(size=(self.num_actions,), fill_value=0.0), requires_grad=True)
        self.value_layer = nn.LazyLinear(out_features=1)

        self._shared_output = None

    def act(self, inputs, role):
        if role == "policy":
            return GaussianMixin.act(self, inputs, role)
        elif role == "value":
            return DeterministicMixin.act(self, inputs, role)

    def compute(self, inputs, role=""):
        if self._shared_output is None:
            # height map now is in states["height_data"]
            states = unflatten_tensorized_space(self.observation_space, inputs.get("states"))
            height_out = self.height_encoder(states["height_data"].unsqueeze(1))
            nav_out = self.nav_encoder(states["nav_data"])
            net = self.net_container(torch.concatenate([states["observations"], height_out, nav_out], dim=1))
            self._shared_output = net

        if role == "policy":
            output = self.policy_layer(self._shared_output)
            return output, self.log_std_parameter, {}
        elif role == "value":
            output = self.value_layer(self._shared_output)
            self._shared_output = None
            return output, {}
        
"""
def get_coordinate_grid(batch_size, h, w, device):
    # --- Generate Coordinate Channels ---
    # Create linear gradients from -1 to 1
    x_range = torch.linspace(-1, 1, steps=w, device=device)
    y_range = torch.linspace(-1, 1, steps=h, device=device)
    
    # Create meshgrid (Y, X)
    # indexing='ij' means first dim is rows (Y), second is cols (X)
    y_grid, x_grid = torch.meshgrid(y_range, x_range, indexing='ij')
    
    # Expand to batch size: (Batch, 1, H, W)
    x_channel = x_grid.unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1, -1)
    y_channel = y_grid.unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1, -1)

    return x_channel, y_channel

class HeightMapEncoder(nn.Module):
    def __init__(self, input_channels=1):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Conv2d(input_channels + 2, 8, kernel_size=3, stride=2, padding=0), # 25x25 -> 12x12, 1152
            nn.ELU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1), # 12x12 -> 6x6, 576
            nn.ELU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # 6x6 -> 3x3, 288
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.Flatten(),
            nn.Linear(32 * 3 * 3, 128),
        )

    def forward(self, x):
        batch_size, _,  h, w = x.shape
        device = x.device
        x_channel, y_channel = get_coordinate_grid(batch_size, h, w, device)
        
        return self.net(torch.cat([x, x_channel, y_channel], dim=1))

class NavigationMapEncoder(nn.Module):
    def __init__(self, input_channels=3, ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(input_channels + 2, 16, kernel_size=3, stride=2), # 33x33 -> 16x16, 4096
            nn.ELU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # 16x16 -> 8x8, 2048
            nn.ELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # 8x8 -> 4x4, 1024
            nn.ELU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1), # 4x4 -> 2x2, 256
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.Flatten(),
            nn.Linear(64 * 2 * 2, 128),
        )

    def forward(self, x):
        # x shape: (Batch, 3, H, W) -> Staleness, Ray Cast Density, Height

        batch_size, _, h, w = x.shape
        device = x.device
        
        x_channel, y_channel = get_coordinate_grid(batch_size, h, w, device)
        
        # Concatenate: (Batch, 3, H, W) + (Batch, 1, H, W) + (Batch, 1, H, W) -> (Batch, 5, H, W)
        x_with_coords = torch.cat([x, x_channel, y_channel], dim=1)
        
        return self.net(x_with_coords)


class SharedRecurrentModel(GaussianMixin, DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, num_envs, init_log_std=0.0):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(
            self,
            clip_actions=False,
            clip_log_std=True,
            min_log_std=-20.0,
            max_log_std=2.0,
            reduction="sum",
            role="policy",
        )
        DeterministicMixin.__init__(self, 
            clip_actions=False, 
            role="value"
        )

        obs_dim = self.observation_space["observations"].shape[0]
        height_dim = self.observation_space["height_data"].shape
        assert height_dim == (25, 25), "Expected height_data to be of shape (25, 25)"
        nav_dim = self.observation_space["nav_data"].shape
        assert nav_dim == (3, 33, 33), "Expected nav_data to be of shape (33, 33)"
        act_dim = self.num_actions
        self.num_envs = num_envs


        # Observation encoder
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
        )

        # Height_data encoder CNN (25x25 to 288)
        self.height_encoder = HeightMapEncoder()

        # Nav_data encoder CNN (33x33 to 256)
        self.nav_encoder = NavigationMapEncoder()

        # For fusion of both encoders
        self.fusion = nn.Sequential(
            nn.Linear(128+128+128, 256),
            nn.ELU(),
        )

        self.num_layers = 1
        self.input_size = 256
        self.hidden_size = 256
        self.sequence_length = 64
        self.gru = nn.GRU(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            batch_first=True,    # Input/output tensors are (batch, seq, feature)
        )

        self.net = nn.Sequential(
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU()
        )

        self.policy_layer = nn.Linear(128, act_dim)
        self.value_layer = nn.Linear(128, 1)
        self.log_std_parameter = nn.Parameter(torch.full(size=(self.num_actions,), fill_value=float(init_log_std)), requires_grad=True)

        self._shared_output = None
        
        #for m in self.modules():
        #    if isinstance(m, nn.Linear):
        #        nn.init.orthogonal_(m.weight, gain=0.6)
        #        nn.init.constant_(m.bias, 0)

    
    def get_specification(self):
        return {
            "rnn": {
                "sequence_length": self.sequence_length,
                "sizes": [
                    (self.num_layers, self.num_envs, self.hidden_size),  # gru memory
                ]
            }
        }
    
    def act(self, inputs, role):
        if role == "policy":
            return GaussianMixin.act(self, inputs, role)
        elif role == "value":
            return DeterministicMixin.act(self, inputs, role)

    def compute(self, inputs, role=""):
        if self._shared_output is None:
            states = unflatten_tensorized_space(self.observation_space, inputs["states"])
            observations = states["observations"]
            height_data = states["height_data"]
            nav_data = states["nav_data"]
            
            terminated = inputs.get("terminated", None)
            rnn_dict = {}
            
            # Encode observations
            obs_encoded = self.obs_encoder(observations)
            height_encoded = self.height_encoder(height_data.unsqueeze(1)) # Add channel dimension
            nav_encoded = self.nav_encoder(nav_data)
            encoded = torch.cat([obs_encoded, height_encoded, nav_encoded], dim=-1)

            # Fusion
            fused = self.fusion(encoded)

            # LSTM
            #rnn_output, rnn_dict = self.lstm_rollout(self.lstm, fused, terminated, inputs["rnn"])
            # GRU
            rnn_output, rnn_dict = self.gru_rollout(self.gru, fused, terminated, inputs["rnn"])

            # Final layers
            net = self.net(rnn_output)

            self._shared_output = net, rnn_dict

        if role == "policy":
            mean = self.policy_layer(net)
            return mean, self.log_std_parameter, rnn_dict

        elif role == "value":
            net, rnn_dict = self._shared_output
            self._shared_output = None
            output = self.value_layer(net)
            return output, rnn_dict

    

    def gru_rollout(self, model, states, terminated, hidden_states):
        #print(f"states shape: {states.shape}, hidden_states shapes: {[h.shape for h in hidden_states]}")
        if self.training:
            # reshape to (batch, seq, features)
            
            rnn_input = states.view(-1, self.sequence_length, states.shape[-1])
            

            hidden_states[0] = hidden_states[0].view(
                self.num_layers, -1, self.sequence_length, hidden_states[0].shape[-1]
            )[:, :, 0, :].contiguous()

            if terminated is not None and torch.any(terminated):
                # handle resets within the sequence
                rnn_outputs = []
                terminated = terminated.view(-1, self.sequence_length)
                indexes = (
                    [0]
                    + (terminated[:, :-1].any(dim=0).nonzero(as_tuple=True)[0] + 1).tolist()
                    + [self.sequence_length]
                )
                for i in range(len(indexes) - 1):
                    i0, i1 = indexes[i], indexes[i + 1]
                    rnn_output, hidden_states[0] = model(
                        rnn_input[:, i0:i1, :], hidden_states[0]
                    )
                    hidden_states[0][:, (terminated[:, i1 - 1]), :] = 0
                    rnn_outputs.append(rnn_output)
                rnn_output = torch.cat(rnn_outputs, dim=1)
            else:
                rnn_output, hidden_states[0] = model(rnn_input, hidden_states[0])
        else:
            # evaluation mode: one step at a time
            rnn_input = states.view(-1, 1, states.shape[-1])
            # Make h contiguous
            hidden_states[0] = hidden_states[0].contiguous()
            rnn_output, hidden_states[0] = model(rnn_input, hidden_states[0])
        # flatten batch + sequence
        rnn_output = torch.flatten(rnn_output, start_dim=0, end_dim=1)

        return rnn_output, {"rnn": hidden_states}
    
    
    def gru_rollout_no_term(self, model, states, terminated, hidden_states):
        #print(f"states shape: {states.shape}, hidden_states shapes: {[h.shape for h in hidden_states]}")
        # evaluation mode: one step at a time
        rnn_input = states.view(-1, 1, states.shape[-1])
        # Make h contiguous
        hidden_states[0] = hidden_states[0].contiguous()
        rnn_output, hidden_states[0] = model(rnn_input, hidden_states[0])
        # flatten batch + sequence
        rnn_output = torch.flatten(rnn_output, start_dim=0, end_dim=1)

        return rnn_output, {"rnn": hidden_states}
    
    # === LSTM rollout logic ===
    def lstm_rollout(self, model, states, terminated, hidden_states):
        #print(f"states shape: {states.shape}, hidden_states shapes: {[h.shape for h in hidden_states]}")
        if self.training:
            # reshape to (batch, seq, features)
            rnn_input = states.view(-1, self.sequence_length, states.shape[-1])
            
            h, c = hidden_states
            h = h.view(
                self.num_layers, -1, self.sequence_length, h.shape[-1]
            )[:, :, 0, :].contiguous()
            c = c.view(
                self.num_layers, -1, self.sequence_length, c.shape[-1]
            )[:, :, 0, :].contiguous()
            hidden_states = (h.contiguous(), c.contiguous())

            if terminated is not None and torch.any(terminated):
                # handle resets within the sequence
                rnn_outputs = []
                terminated = terminated.view(-1, self.sequence_length)
                indexes = (
                    [0]
                    + (terminated[:, :-1].any(dim=0).nonzero(as_tuple=True)[0] + 1).tolist()
                    + [self.sequence_length]
                )
                for i in range(len(indexes) - 1):
                    i0, i1 = indexes[i], indexes[i + 1]
                    rnn_output, (h, c) = model(
                        rnn_input[:, i0:i1, :], hidden_states
                    )
                    h[:, (terminated[:, i1 - 1]), :] = 0
                    c[:, (terminated[:, i1 - 1]), :] = 0
                    hidden_states = (h, c)
                    rnn_outputs.append(rnn_output)
                rnn_output = torch.cat(rnn_outputs, dim=1)
            else:
                rnn_output, hidden_states = model(rnn_input, hidden_states)
        else:
            # evaluation mode: one step at a time
            rnn_input = states.view(-1, 1, states.shape[-1])
            # Make h, c contiguous
            h, c = hidden_states
            h = h.contiguous()
            c = c.contiguous()
            hidden_states = (h, c)
            rnn_output, hidden_states = model(rnn_input, hidden_states)

        # flatten batch + sequence
        rnn_output = torch.flatten(rnn_output, start_dim=0, end_dim=1)

        return rnn_output, {"rnn": hidden_states}
"""