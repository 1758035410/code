import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, args):
        super(Discriminator, self).__init__()
        self.num_nodes = args.num_nodes
        self.model = nn.Sequential(
            nn.Linear(self.num_nodes, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):  
        x = x.squeeze() # [B, (W+H), N]
        x_flat = x.view(-1, x.shape[2]) # [B*(W+H), N]

        validity = self.model(x_flat)

        return validity

def gradient_penalty(discriminator, real_data, fake_data):
    alpha = torch.rand(real_data.size(0), 1, 1).to(real_data.device)
    interpolated_data = alpha * real_data + (1 - alpha) * fake_data
    interpolated_data.requires_grad_(True)

    disc_output = discriminator(interpolated_data)

    gradients = torch.autograd.grad(outputs=disc_output, inputs=interpolated_data,
                                    grad_outputs=torch.ones_like(disc_output),
                                    create_graph=True, retain_graph=True)[0]

    gradients_norm = gradients.view(gradients.size(0), -1).norm(p=2, dim=1)
    penalty = ((gradients_norm - 1) ** 2).mean()

    return penalty


