import torch


def lambda_returns(
    rewards, continuations, next_values, discount=0.99, lam=0.95, valid=None
):
    next_return = next_values[:, -1]
    returns = []
    for step in reversed(range(rewards.shape[1])):
        if valid is not None:
            has_next_step = (
                valid[:, step + 1].bool()
                if step + 1 < rewards.shape[1]
                else torch.zeros_like(valid[:, step], dtype=torch.bool)
            )
            next_return = torch.where(has_next_step, next_return, next_values[:, step])
        target = rewards[:, step] + discount * continuations[:, step] * (
            (1 - lam) * next_values[:, step] + lam * next_return
        )
        next_return = (
            target
            if valid is None
            else torch.where(valid[:, step].bool(), target, next_values[:, step])
        )
        returns.append(next_return)
    return torch.stack(returns[::-1], 1)


def twohot(target, count=255):
    transformed = target.sign() * target.abs().log1p()
    location = ((transformed.clamp(-5, 5) + 5) * (count - 1) / 10).clamp(0, count - 1)
    lower = location.floor().long().clamp(max=count - 2)
    fraction = location - lower
    result = torch.zeros((*target.shape, count), device=target.device)
    result.scatter_add_(-1, lower[..., None], (1 - fraction)[..., None])
    result.scatter_add_(-1, (lower + 1)[..., None], fraction[..., None])
    return result


def distribution_loss(logits, target):
    return -(twohot(target, logits.shape[-1]) * logits.log_softmax(-1)).sum(-1)


def distribution_mean(logits):
    bins = torch.linspace(-5, 5, logits.shape[-1], device=logits.device)
    bins = bins.sign() * bins.abs().expm1()
    return (logits.softmax(-1) * bins).sum(-1)
