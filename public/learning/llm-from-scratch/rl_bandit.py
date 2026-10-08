"""A four-action policy-gradient lab, not LLM PPO/GRPO training."""
import json
import torch


def main():
    torch.set_num_threads(1)
    torch.manual_seed(17)
    rewards = torch.tensor([0., 0., 0., 1.], dtype=torch.float64)
    logits = torch.zeros(4, dtype=torch.float64, requires_grad=True)
    probabilities = logits.softmax(-1)
    expected_reward = (probabilities * rewards).sum()
    exact_gradient = torch.autograd.grad(expected_reward, logits)[0]
    torch.testing.assert_close(exact_gradient, torch.tensor([-0.0625, -0.0625, -0.0625, 0.1875], dtype=torch.float64))
    # Enumerate actions: sum pi(a) * (R(a)-b) * grad log pi(a).
    score_gradient = probabilities.detach() * (rewards - expected_reward.detach())
    torch.testing.assert_close(exact_gradient, score_gradient)
    baseline_start = expected_reward.item()
    optimizer = torch.optim.SGD([logits], lr=0.3)
    for step in range(250):
        p = logits.softmax(-1)
        baseline = (p.detach() * rewards).sum()
        actions = torch.multinomial(p.detach(), 128, replacement=True)
        advantage = rewards[actions] - baseline
        loss = -(advantage * logits.log_softmax(-1)[actions]).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    final_p = logits.detach().softmax(-1)
    final_reward = (final_p * rewards).sum().item()
    assert final_reward > 0.9
    print(json.dumps({"experiment": "four-action REINFORCE", "seed": 17,
                      "initial_expected_reward": baseline_start, "final_expected_reward": final_reward,
                      "final_probabilities": final_p.tolist(), "steps": 250}, indent=2))
    print("PASS exact score-function identity and sampled policy improvement; not LLM RL.")


if __name__ == "__main__":
    main()
