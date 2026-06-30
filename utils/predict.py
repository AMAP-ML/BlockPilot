# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import json

# ----------------- 0.  -----------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype = torch.bfloat16

dropout_prob = 0.2
hidden_dim = 2048
num_classes = 5
n_layers = 2   # ⭐ （=1）

# test dataset
test_dataset = torch.load("Predictor/utils/dataset.pt")
test_inputs = torch.stack([d['input'].view(-1) for d in test_dataset])
_, input_dim = test_inputs.shape

# ----------------- 2.  MLP -----------------
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, n_layers=1, dropout_prob=0.2):
        super().__init__()

        self.layers = nn.ModuleList()

        # 
        self.layers.append(nn.Linear(input_dim, hidden_dim))

        # （ n_layers > 1 ）
        for _ in range(n_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))

        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout_prob)
        self.fc_out = nn.Linear(hidden_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        for layer in self.layers:
            nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
            nn.init.zeros_(layer.bias)

        nn.init.xavier_uniform_(self.fc_out.weight)
        nn.init.zeros_(self.fc_out.bias)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            residual = x
            x = layer(x)
            x = self.act(x)
            x = self.dropout(x)
            if i > 0:
                x = x + residual
        x = self.fc_out(x)
        return x

# ----------------- 3.  -----------------
model = MLP(
    input_dim=input_dim,
    hidden_dim=hidden_dim,
    num_classes=num_classes,
    n_layers=n_layers,
    dropout_prob=dropout_prob
).to(device=device, dtype=dtype)

# ----------------- load checkpoint -----------------
state_dict = torch.load("Predictor/utils/model.pt", map_location=device)
model.load_state_dict(state_dict)
# ----------------- inference -----------------
model.eval()
with torch.no_grad():
    logits = model(test_inputs.to(device=device, dtype=dtype))
    preds = logits.argmax(dim=1)

# _zh
# ============================================================
dir_list = ["Predictor/utils"]
if all("task" in d for d in test_dataset):
    dataset_list = sorted({d["task"] for d in test_dataset})
else:
    dataset_list = ["gsm8k"]
temp_list = ["0.0"]
# ============================================================
# _zh
if all("task" in d and "index" in d for d in test_dataset):
    final_result = {}
    for i, d in enumerate(test_dataset):
        task = d['task']
        index = d['index']
        k = int(preds[i].item()) + 14  # preds+14  result  key
        value = k

        if task not in final_result:
            final_result[task] = {}
        final_result[task][index] = value
    for task, task_dict in final_result.items():
        out_dir = f"{dir_list[0]}/{task}_0T{temp_list[0]}"
        out_path = f"{out_dir}/1.json"
        task_dict_strkey = {str(k): v for k, v in task_dict.items()}
        with open(out_path, "w") as f:
            json.dump(task_dict_strkey, f, indent=4)
else:
    print("Skip writing 1.json because test dataset has no task/index fields.")