import torch
import torch.nn as nn

class Dummy(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 10)
    def forward(self, x):
        return self.fc(x)

model = Dummy().to("cpu")
try:
    c_model = torch.compile(model)
    print("Compilation successful, testing forward...")
    x = torch.randn(2, 10)
    out = c_model(x)
    print("Forward successful!")
except Exception as e:
    import traceback
    traceback.print_exc()
