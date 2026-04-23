def train(config: PPOConfig) -> tuple[str, dict[str, float]]:
    run_dir = Path(config.results_dir) / f"{config.env_name}_seed{config.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "log.csv"
    model_path = run_dir / "model.pt"

    env = gym.make(config.env_name)
    eval_env = gym.make(config.env_name)
    set_seed(config.seed, env)
    set_seed(config.seed + 1234, eval_env)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ActorCritic(obs_dim, act_dim, config.hidden_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    state = extract_state(env.reset())
    episode_return = 0.0
    train_rewards = deque(maxlen=config.train_reward_window)
    train_reward_series: list[float] = []
    eval_reward_series: list[float] = []
    max_q_series: list[float] = []
    loss_series: list[float] = []
    loss_ema: float | None = None
    rollout_max_q_values: list[float] = []

    fieldnames = ["Timestep", "Training Rewards", "Eval Rewards", "Max Q", "Loss", "Entropy", "KL"]
    with log_path.open("w", newline="") as log_file:
        writer = csv.DictWriter(log_file, fieldnames=fieldnames)
        writer.writeheader()

        timesteps = 0
        next_eval = config.eval_freq
        while timesteps < config.total_timesteps:
            rollout = RolloutBuffer(config.rollout_steps, obs_dim)
            for _ in range(config.rollout_steps):
                obs_tensor = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    action_tensor, log_prob_tensor, _, value_tensor = model.get_action_value(obs_tensor)
                    logits, value_pred = model.forward(obs_tensor)
                action = int(action_tensor.item())
                log_prob = float(log_prob_tensor.item())
                value = float(value_tensor.item())
                rollout_max_q_values.append(float(value_pred.item()))

                next_state, reward, done, _ = extract_step(env.step(action))
                rollout.add(state, action, log_prob, float(reward), done, value)

                timesteps += 1
                episode_return += float(reward)
                state = next_state

                if done:
                    train_rewards.append(episode_return)
                    episode_return = 0.0
                    state = extract_state(env.reset())

                if timesteps >= config.total_timesteps:
                    break

            with torch.no_grad():
                last_obs = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                _, last_value = model.forward(last_obs)
                last_value = float(last_value.item())

            rollout.compute_returns_advantages(
                last_value=last_value,
                last_done=False,
                gamma=config.gamma,
                gae_lambda=config.gae_lambda,
            )

            batch = rollout.get_tensors(device)
            num_samples = batch["obs"].shape[0]
            indices = np.arange(num_samples)

            progress = min(float(timesteps) / float(config.total_timesteps), 1.0)
            lr_now = max(config.learning_rate * (1.0 - progress),
                         config.learning_rate * 0.1)
            for group in optimizer.param_groups:
                group["lr"] = max(lr_now, 1e-6)
            entropy_coef_now = (
                config.entropy_coef
                + (config.entropy_coef_end - config.entropy_coef) * progress
            )
            value_coef_now = (
                config.value_coef
                * ((1.0 - progress) + config.value_coef_end_ratio * progress)
            )

            epoch_losses = []
            epoch_value_losses = []
            epoch_entropies = []
            epoch_kls = []
            stop_early = False

            for _ in range(config.update_epochs):
                np.random.shuffle(indices)
                for start in range(0, num_samples, config.minibatch_size):
                    end = start + config.minibatch_size
                    mb_idx = indices[start:end]
                    mb_obs = batch["obs"][mb_idx]
                    mb_actions = batch["actions"][mb_idx]
                    mb_old_log_probs = batch["log_probs"][mb_idx]
                    mb_advantages = batch["advantages"][mb_idx]
                    mb_returns = batch["returns"][mb_idx]
                    mb_old_values = batch["old_values"][mb_idx]

                    _, new_log_probs, entropy, values = model.get_action_value(mb_obs, mb_actions)
                    ratio = (new_log_probs - mb_old_log_probs).exp()
                    clipped_ratio = torch.clamp(ratio, 1.0 - config.clip_ratio, 1.0 + config.clip_ratio)

                    policy_loss = -torch.min(ratio * mb_advantages, clipped_ratio * mb_advantages).mean()
                    unclipped_value_loss = (values - mb_returns) ** 2
                    value_clipped = mb_old_values + torch.clamp(
                        values - mb_old_values,
                        -config.value_clip_ratio,
                        config.value_clip_ratio,
                    )
                    clipped_value_loss = (value_clipped - mb_returns) ** 2
                    value_loss = 0.5 * torch.max(unclipped_value_loss, clipped_value_loss).mean()
                    if config.normalize_value_loss:
                        value_scale = mb_returns.detach().var(unbiased=False).clamp_min(config.value_loss_eps)
                        value_loss_for_opt = value_loss / value_scale
                    else:
                        value_loss_for_opt = value_loss
                    entropy_loss = entropy.mean()

                    loss = (
                        policy_loss
                        + value_coef_now * value_loss_for_opt
                        - entropy_coef_now * entropy_loss
                    )

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                    optimizer.step()

                    approx_kl = (mb_old_log_probs - new_log_probs).mean().item()
                    epoch_losses.append(float(loss.item()))
                    epoch_value_losses.append(float((value_coef_now * value_loss_for_opt).item()))
                    epoch_entropies.append(float(entropy_loss.item()))
                    epoch_kls.append(float(approx_kl))
                    if approx_kl > config.target_kl:
                        stop_early = True
                        break
                if stop_early:
                    break

            mean_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
            mean_value_loss = float(np.mean(epoch_value_losses)) if epoch_value_losses else 0.0
            mean_entropy = float(np.mean(epoch_entropies)) if epoch_entropies else 0.0
            mean_kl = float(np.mean(epoch_kls)) if epoch_kls else 0.0
            if loss_ema is None:
                loss_ema = mean_value_loss
            else:
                loss_ema = config.loss_ema_decay * loss_ema + (1.0 - config.loss_ema_decay) * mean_value_loss
            loss_series.append(float(loss_ema))

            if timesteps >= next_eval or timesteps >= config.total_timesteps:
                eval_reward = evaluate_policy(eval_env, model, config.eval_episodes, device)
                avg_train_reward = float(np.mean(train_rewards)) if len(train_rewards) > 0 else 0.0
                avg_max_q = float(np.percentile(rollout_max_q_values, 95)) if len(rollout_max_q_values) > 0 else 0.0
                train_reward_series.append(avg_train_reward)
                eval_reward_series.append(eval_reward)
                max_q_series.append(avg_max_q)
                row = {
                    "Timestep": timesteps,
                    "Training Rewards": avg_train_reward,
                    "Eval Rewards": eval_reward,
                    "Max Q": avg_max_q,
                    "Loss": float(loss_ema),
                    "Entropy": mean_entropy,
                    "KL": mean_kl,
                }
                writer.writerow(row)
                log_file.flush()
                print(
                    f"Timestep {timesteps} | Train R {avg_train_reward:.2f} | "
                    f"Eval R {eval_reward:.2f} | Max Q {avg_max_q:.4f} | "
                    f"Loss {loss_ema:.4f} | "
                    f"Entropy {mean_entropy:.4f} | KL {mean_kl:.4f}"
                )
                rollout_max_q_values = []
                next_eval += config.eval_freq

    torch.save(model.state_dict(), model_path)

    export_plot(train_reward_series, "Training Rewards", str(run_dir / "training_rewards.png"))
    export_plot(eval_reward_series, "Eval Rewards", str(run_dir / "eval_rewards.png"))
    export_plot(max_q_series, "Max Q", str(run_dir / "max_q.png"))
    export_plot(loss_series, "Loss", str(run_dir / "loss.png"))

    env.close()
    eval_env.close()

    summary = {
        "final_training_reward": float(train_reward_series[-1]) if train_reward_series else 0.0,
        "best_training_reward": float(np.max(train_reward_series)) if train_reward_series else 0.0,
        "final_eval_reward": float(eval_reward_series[-1]) if eval_reward_series else 0.0,
        "best_eval_reward": float(np.max(eval_reward_series)) if eval_reward_series else 0.0,
        "final_max_q": float(max_q_series[-1]) if max_q_series else 0.0,
        "peak_max_q": float(np.max(max_q_series)) if max_q_series else 0.0,
        "final_loss": float(loss_series[-1]) if loss_series else 0.0,
        "min_loss": float(np.min(loss_series)) if loss_series else 0.0,
    }
    return str(run_dir), summary