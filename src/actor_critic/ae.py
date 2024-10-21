import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

# Define the autoencoder architecture
class AutoEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(AutoEncoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

        self.latent_dim = latent_dim

    def forward(self, x):
        z = self.encoder(x)
        reconstructed = self.decoder(z)
        return reconstructed
    
    def train_ae(self, observations, n_epochs = 50, lr = 1e-3, device='cuda'):

        input_dim = observations.shape[1]

        self.to(device)
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        # Prepare DataLoader
        dataset = TensorDataset(torch.from_numpy(observations).float())
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

        # Training loop
        num_epochs = n_epochs

        for epoch in tqdm(range(num_epochs)):
            for data in dataloader:
                inputs = data[0].to(device)
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, inputs)
                loss.backward()
                optimizer.step()
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

        # Save the trained autoencoder and scaler
        torch.save(self.state_dict(), 'autoencoder.pth')