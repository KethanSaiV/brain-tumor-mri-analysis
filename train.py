import torch
from torch.utils.data import DataLoader
from model import TumorCNN
from dataset import MRIDataset

device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = MRIDataset("data")
loader = DataLoader(dataset, batch_size=16, shuffle=True)

model = TumorCNN().to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.CrossEntropyLoss()

for epoch in range(10):
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = loss_fn(pred, y)

        opt.zero_grad()
        loss.backward()
        opt.step()

    print(f"Epoch {epoch} Loss {loss.item():.4f}")

torch.save(model.state_dict(), "tumor_model.pth")
