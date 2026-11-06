import torch


# same lambda return as our previous.
# G₂ = reward₂ + 0.99 × continuation₂ × (0.05 × critic(s₃) + 0.95 × G₃)
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
