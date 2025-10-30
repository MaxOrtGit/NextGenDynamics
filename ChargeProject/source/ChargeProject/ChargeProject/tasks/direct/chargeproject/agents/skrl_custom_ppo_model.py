
import torch
import torch.nn as nn

# Assuming these are imported from your RL library (e.g., skrl)
from skrl.models.torch import Model, GaussianMixin, DeterministicMixin
from skrl.utils.spaces.torch import unflatten_tensorized_space


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
        assert height_dim == (16, 16), "Expected height_data to be of shape (16, 16)"
        act_dim = self.num_actions
        self.num_envs = num_envs


        # Observation encoder
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
        )

        # Height_data encoder CNN (16x16 to 256)
        self.height_encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1),   # 16x16 -> 8x8, 512
            nn.ELU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),  # 8x8 -> 4x4, 256
            nn.ELU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # 4x4 -> 2x2, 128
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.Flatten(), # 32 * 2 * 2 = 128
        )

        # For fusion of both encoders
        self.fusion = nn.Sequential(
            nn.Linear(256, 128),
            nn.ELU(),
        )


        self.num_layers = 1
        self.input_size = 128
        self.hidden_size = 128
        self.sequence_length = 32
        self.gru = nn.GRU(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            batch_first=True,    # Input/output tensors are (batch, seq, feature)
        )

        self.net = nn.Sequential(
            nn.ELU()
        )# check if end with relu

        self.policy_layer = nn.Linear(128, act_dim)
        self.value_layer = nn.Linear(128, 1)
        self.log_std_parameter = nn.Parameter(torch.full(size=(self.num_actions,), fill_value=init_log_std), requires_grad=True)

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
            
            #terminated = inputs.get("terminated", None)
            rnn_dict = {}
            
            # Encode observations
            obs_encoded = self.obs_encoder(observations)
            height_encoded = self.height_encoder(height_data.unsqueeze(1))  # Add channel dimension
            encoded = torch.cat([obs_encoded, height_encoded], dim=-1)

            # Fusion
            fused = self.fusion(encoded)

            # LSTM
            #rnn_output, rnn_dict = self.lstm_rollout(self.lstm, fused, terminated, inputs["rnn"])
            # GRU
            rnn_output, rnn_dict = self.gru_rollout(self.gru, fused, None, inputs["rnn"])

            # Final layers
            net = self.net(rnn_output)

            self._shared_output = net, rnn_dict

        if role == "policy":
            mean = torch.tanh(self.policy_layer(net))
            return mean, self.log_std_parameter, rnn_dict

        elif role == "value":
            net, rnn_dict = self._shared_output
            self._shared_output = None
            output = self.value_layer(net)
            return output, rnn_dict

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

"""
import torch
import torch.nn as nn

# Assuming these are imported from your RL library (e.g., skrl)
from skrl.models.torch import Model, GaussianMixin, DeterministicMixin
from skrl.utils.spaces.torch import unflatten_tensorized_space


class SharedRecurrentModel(GaussianMixin, DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, num_envs, init_log_std=0.0, gain=1.0):
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

        base_dim = self.observation_space["base_obs"].shape[0]
        leg_dims = self.observation_space["leg_obs"].shape
        self.num_legs = leg_dims[0]
        self.num_leg_joints = action_space.shape[1]
        height_dim = self.observation_space["height_data"].shape
        assert height_dim == (16, 16), "Expected height_data to be of shape (16, 16)"
        self.num_envs = num_envs


        # Observation encoder
        self.base_encoder = nn.Sequential(
            nn.Linear(base_dim, 64),
            nn.ReLU(),
        )

        # Height_data encoder CNN (16x16 to 128)
        self.height_encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1),   # 16x16 -> 8x8
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),  # 8x8 -> 4x4
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # 4x4 -> 2x2
            nn.ReLU(),
            nn.Flatten(),  # 32 * 2 * 2 = 128
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        
        # Leg encoder (shared across legs)
        self.leg_encoder = nn.Sequential(
            nn.Linear(leg_dims[1], 64),
            nn.ReLU(),
        )


        # ----- Policy Network -----

        leg_input_size = 64 + 128 + 64 # 224 (base + height + leg_encoded)
        
        self.leg_pre_net = nn.Sequential(
            nn.Linear(leg_input_size, 128),
            nn.ReLU(),
        )

        # Leg gru (weights shared, memory per leg)
        self.sequence_length = 32
        self.leg_num_layers = 1
        self.leg_hidden_size = 128 
        self.leg_gru = nn.GRU(
            input_size=self.leg_hidden_size,
            num_layers=self.leg_num_layers,
            hidden_size=self.leg_hidden_size,
            batch_first=True,
        )

        # Policy Head
        self.leg_policy_head = nn.Sequential(
            nn.Linear(self.leg_hidden_size, self.num_leg_joints),
        )


        # ----- Value Network -----
        self.value_leg_conv = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=3, stride=1, padding=1, padding_mode='circular'),  # 6 → 6
            nn.ReLU(),
            nn.Conv1d(64, 32, kernel_size=3, stride=2, padding=1, padding_mode='circular'),  # 6 → 3
            nn.ReLU(),
            nn.Flatten(), # 32*3=96
            nn.Linear(96, 64),
            nn.ReLU(),
        )

        value_input_size = 64 + 128 + 64 # base + height + leg_attention (256)

        self.value_layer = nn.Sequential(
            nn.Linear(value_input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        self.log_std_parameter = nn.Parameter(torch.full(size=(self.num_leg_joints,), fill_value=init_log_std), requires_grad=True)

        self._policy_out = None
        self._value_out = None

        self.debug_i = 0 

        #for m in self.modules():
        #    if isinstance(m, nn.Linear):
        #        nn.init.orthogonal_(m.weight, gain=gain)
        #        nn.init.constant_(m.bias, 0)


    
    def get_specification(self):
        return {
            "rnn": {
                "sequence_length": self.sequence_length,
                "sizes": [
                    (self.leg_num_layers, self.num_envs, self.leg_hidden_size) for _ in range(self.num_legs)
                ] +
                [
                    #(self.value_num_layers, self.num_envs, self.value_hidden_size)
                ]
            }
        }
    
    def act(self, inputs, role):
        if role == "policy":
            return GaussianMixin.act(self, inputs, role)
        elif role == "value":
            return DeterministicMixin.act(self, inputs, role)

    def compute(self, inputs, role=""):
        rnn_data = inputs["rnn"]

        if self._value_out is None:
            states = unflatten_tensorized_space(self.observation_space, inputs["states"])
            base_obs = states["base_obs"]
            leg_obs_all = states["leg_obs"]  # [B, num_legs, leg_dim]
            height_data = states["height_data"]

            terminated = inputs.get("terminated", None)
            rnn_dict = {"rnn": []}

            # --- Base and height encoders ---
            base_encoded = self.base_encoder(base_obs)
            height_encoded = self.height_encoder(height_data.unsqueeze(1))  # add channel dim

            # --- Per-leg encoding and GRU ---
            # Flatten so each leg acts like different env
            leg_obs_flat = leg_obs_all.reshape(-1, leg_obs_all.shape[-1])  # [B * num_legs, leg_dim]
            # Fuse and expand base to match leg count
            fused_base = torch.cat([base_encoded, height_encoded], dim=-1)
            fused_base_exp = fused_base.unsqueeze(1).expand(-1, self.num_legs, -1).contiguous()
            fused_base_flat = fused_base_exp.reshape(-1, fused_base_exp.shape[-1])  # [B * num_legs, 128]
            # expand terminated to match leg count
            terminated_flat = terminated.expand(-1, self.num_legs).reshape(-1) if terminated is not None else None
            # Flatten rnn_data for legs
            leg_data_flat = torch.cat(rnn_data[:self.num_legs], dim=1)

            leg_encoded = self.leg_encoder(leg_obs_flat)
            
            # --- Policy SPECIFIC ---
            leg_input = torch.cat([fused_base_flat, leg_encoded], dim=-1)
            
            leg_gru_input = self.leg_pre_net(leg_input)
            # rnn_data[0]-rnn_data[5]
            leg_rnn_output, leg_rnn_hidden = self.gru_rollout(
                self.leg_gru,
                self.leg_num_layers,
                leg_gru_input,
                terminated_flat,
                leg_data_flat,
            )

            leg_action_mean = self.leg_policy_head(leg_rnn_output)  # [B * num_legs, action_dim_per_leg]
            actions_flat = leg_action_mean.view(-1, self.num_legs *self.num_leg_joints)  # [B, num_legs *action_dim_per_leg]

            # Reshape leg hidden states back to per-leg
            for i in range(self.num_legs):
                rnn_dict["rnn"].append(leg_rnn_hidden[:, i::self.num_legs, :])


            # ----- Value Specific -----
            leg_features = leg_rnn_output.view(-1, self.num_legs, self.leg_hidden_size)
            leg_context = self.value_leg_conv(leg_features.permute(0, 2, 1)).squeeze(-1)

            value_input = torch.cat(
                [fused_base, leg_context],
                dim=-1,
            )  # base + height + leg_encoded
            #value_gru_input = self.value_pre_net(value_input)

            #value_rnn_output, value_rnn_hidden = self.gru_rollout(
            #    self.value_gru,
            #    self.value_num_layers,
            #    value_gru_input,
            #    terminated,
            #    rnn_data[-1],  # last slot for value GRU
            #)
            #rnn_dict["rnn"].append(value_rnn_hidden)

            # Final value prediction
            value = self.value_layer(value_input)  # [B, 1]

            log_flat = self.log_std_parameter.repeat(self.num_legs)
            self._policy_out = actions_flat, log_flat, rnn_dict
            self._value_out = value, rnn_dict


                    

        if role == "policy":
            out = self._policy_out
            self._policy_out = None
            return out
        elif role == "value":
            out = self._value_out
            self._value_out = None
            return out

    def gru_rollout(self, model, num_layers, states, terminated, hidden_state):
        if self.training:
            # reshape to (batch, seq, features)
            rnn_input = states.view(-1, self.sequence_length, states.shape[-1])
            

            hidden_state = hidden_state.view(
                num_layers, -1, self.sequence_length, hidden_state.shape[-1]
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
                    rnn_output, hidden_state = model(
                        rnn_input[:, i0:i1, :], hidden_state
                    )
                    hidden_state[:, (terminated[:, i1 - 1]), :] = 0
                    rnn_outputs.append(rnn_output)
                rnn_output = torch.cat(rnn_outputs, dim=1)
            else:
                rnn_output, hidden_state = model(rnn_input, hidden_state)
        else:
            # evaluation mode: one step at a time
            rnn_input = states.view(-1, 1, states.shape[-1])
            # Make h contiguous
            hidden_state = hidden_state.contiguous()
            rnn_output, hidden_state = model(rnn_input, hidden_state)
        # flatten batch + sequence
        rnn_output = torch.flatten(rnn_output, start_dim=0, end_dim=1)

        return rnn_output, hidden_state
        """