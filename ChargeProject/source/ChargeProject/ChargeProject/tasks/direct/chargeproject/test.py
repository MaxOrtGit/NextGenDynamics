import torch

# -----------------------------
# CONFIG
# -----------------------------
checkpoint_path = r"G:\Code\NextGenDynamics\ChargeProject\logs\skrl\spiderbot\2025-10-12_18-08-46_ppo_torch\checkpoints\agent_1000.pt"
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

# -----------------------------
# LOAD CHECKPOINT
# -----------------------------
checkpoint = torch.load(checkpoint_path, map_location=device)
print("Checkpoint keys:", checkpoint.keys())

# -----------------------------
# FUNCTION TO INSPECT A STATE DICT
# -----------------------------
def inspect_state_dict(state_dict, name):
    print(f"\n{name} state dict:")
    lstm_detected = False
    for param_name, tensor in state_dict.items():
        print(f"{param_name}: {tensor.shape}, device={tensor.device}")
        # Detect if this is an LSTM layer
        if "weight_ih" in param_name or "weight_hh" in param_name:
            lstm_detected = True
    if lstm_detected:
        print(f"✅ LSTM detected in {name}")
    else:
        print(f"⚠️ No LSTM detected in {name}")

# -----------------------------
# INSPECT POLICY AND VALUE
# -----------------------------
policy_state = checkpoint['policy']
value_state = checkpoint['value']

inspect_state_dict(policy_state, "Policy model")
inspect_state_dict(value_state, "Value model")



"""
class SharedModel(GaussianMixin,DeterministicMixin, Model):
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

        self.rnn_container = nn.Sequential(
        )
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

    def act(self, inputs, role):
        if role == "policy":
            return GaussianMixin.act(self, inputs, role)
        elif role == "value":
            return DeterministicMixin.act(self, inputs, role)

    def compute(self, inputs, role=""):
        if role == "policy":
            states = unflatten_tensorized_space(self.observation_space, inputs.get("states"))
            taken_actions = unflatten_tensorized_space(self.action_space, inputs.get("taken_actions"))
            rnn = self.rnn_container(states)
            net = self.net_container(rnn)
            self._shared_output = net
            output = self.policy_layer(net)
            return output, self.log_std_parameter, {}
        elif role == "value":
            if self._shared_output is None:
                states = unflatten_tensorized_space(self.observation_space, inputs.get("states"))
                taken_actions = unflatten_tensorized_space(self.action_space, inputs.get("taken_actions"))
                rnn = self.rnn_container(states)
                shared_output = net
                shared_output = net
                shared_output = net
                shared_output = net
            else:
                shared_output = self._shared_output
            self._shared_output = None
            output = self.value_layer(shared_output)
            return output, {}

"""