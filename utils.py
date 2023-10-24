import torch


def gradient_penalty(critic, real, fake, device="CPU"):
    BATCH_SIZE, C, H, W = real.shape
    epsilon = torch.rand(BATCH_SIZE, 1, 1, 1).repeat(1, C, H, W).to(device)
    interpolated_images = real * epsilon + fake * (1-epsilon)

    mixed_score = critic(interpolated_images)

    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_score,
        grad_outputs=torch.ones_like(mixed_score),
        create_graph=True,
        retain_graph=True,
    )[0]

    gradient = gradient.view(gradient.shape[0], -1)
    grad_norm = gradient.norm(2, dim=1)
    grad_penalty = torch.mean((grad_norm - 1) ** 2)
    return grad_penalty
