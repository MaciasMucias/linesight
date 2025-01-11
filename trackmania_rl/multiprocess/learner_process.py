"""
This file implements the main training loop, tensorboard statistics tracking, etc...
"""

import copy
import importlib
import math
import random
import sys
import time
import typing
from collections import defaultdict
from copy import deepcopy
from datetime import datetime
from multiprocessing.connection import wait
from pathlib import Path

import joblib
import numpy as np
import torch
from torch import multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
from torchrl.data.replay_buffers import PrioritizedSampler

from config_files import config_copy
from trackmania_rl import buffer_management, utilities
from trackmania_rl.agents import sac
from trackmania_rl.agents.sac import make_untrained_sac_network
from trackmania_rl.analysis_metrics import (
    distribution_curves,
    highest_prio_transitions,
    loss_distribution,
    race_time_left_curves,
    tau_curves,
)
from trackmania_rl.buffer_utilities import make_buffers, resize_buffers
from trackmania_rl.map_reference_times import reference_times

torch.autograd.set_detect_anomaly(True)

def learner_process_fn(
    rollout_queues,
    uncompiled_shared_network,
    shared_network_lock,
    shared_steps: mp.Value,
    base_dir: Path,
    save_dir: Path,
    tensorboard_base_dir: Path,
):
    layout_version = "lay_mono"
    SummaryWriter(log_dir=str(tensorboard_base_dir / layout_version)).add_custom_scalars(
        {
            layout_version: {
                "single_zone_reached": [
                    "Multiline",
                    [
                        "^single_zone_reached",
                    ],
                ],
                "avg_Q": [
                    "Multiline",
                    ["avg_Q"],
                ],
                "critic_loss": [
                    "Multiline",
                    [
                        "^critic_loss$",
                        "^critic_loss_test$",
                    ],
                ],
                "policy_loss": [
                    "Multiline", [
                        "^policy_loss$",
                        "^policy_loss_test$"
                    ]
                ],
                "entropy_loss": [
                    "Multiline",
                    [
                        "^alpha_loss$",
                        "^alpha_loss_test$",
                    ],
                ],
                "entropy": [
                    "Multiline",
                    [
                        "^alpha_t$",
                        "^alpha_t_test$",
                    ]
                ],
                "eval_race_time_robust": [
                    "Multiline",
                    [
                        "eval_race_time_robust",
                    ],
                ],
                "explo_race_time_finished": [
                    "Multiline",
                    [
                        "explo_race_time_finished",
                    ],
                ]
            },
        }
    )

    # ========================================================
    # Create new stuff
    # ========================================================

    online_network, uncompiled_online_network = make_untrained_sac_network(config_copy.use_jit, is_inference=False)

    print(online_network)
    utilities.count_parameters(online_network)

    accumulated_stats: defaultdict[str, typing.Any] = defaultdict(int)
    accumulated_stats["alltime_min_ms"] = {}
    accumulated_stats["rolling_mean_ms"] = {}
    previous_alltime_min = None
    time_last_save = time.perf_counter()
    queue_check_order = list(range(len(rollout_queues)))
    rollout_queue_readers = [q._reader for q in rollout_queues]
    time_waited_for_workers_since_last_tensorboard_write = 0
    time_training_since_last_tensorboard_write = 0
    time_testing_since_last_tensorboard_write = 0

    # ========================================================
    # Load existing stuff
    # ========================================================
    # noinspection PyBroadException
    try:
        online_network.load_state_dict(torch.load(f=save_dir / "weights1.torch", weights_only=False))
        print(" =====================     Learner weights loaded !     ============================")
    except:
        print(" Learner could not load weights")

    with shared_network_lock:
        uncompiled_shared_network.load_state_dict(uncompiled_online_network.state_dict())

    # noinspection PyBroadException
    try:
        accumulated_stats = joblib.load(save_dir / "accumulated_stats.joblib")
        shared_steps.value = accumulated_stats["cumul_number_memories_generated"]
        print(" =====================      Learner stats loaded !      ============================")
    except:
        print(" Learner could not load stats")

    if "rolling_mean_ms" not in accumulated_stats.keys():
        # Temporary to preserve compatibility with old runs that doesn't have this feature. To be removed later.
        accumulated_stats["rolling_mean_ms"] = {}

    accumulated_stats["cumul_number_single_memories_should_have_been_used"] = accumulated_stats["cumul_number_single_memories_used"]
    transitions_learned_last_save = accumulated_stats["cumul_number_single_memories_used"]
    neural_net_reset_counter = 0
    single_reset_flag = config_copy.single_reset_flag


    critic_lr = utilities.from_exponential_schedule(
        config_copy.critic_lr_schedule, accumulated_stats["cumul_number_memories_generated"],
    )
    policy_lr = utilities.from_exponential_schedule(
        config_copy.policy_lr_schedule, accumulated_stats["cumul_number_memories_generated"]
    )
    alpha_lr = utilities.from_exponential_schedule(
        config_copy.alpha_lr_schedule, accumulated_stats["cumul_number_memories_generated"]
    )

    log_alpha = torch.tensor(config_copy.alpha_initial_value,
                             dtype=torch.float32,
                             device="cuda",
                             requires_grad=True)

    critic_optimizer = torch.optim.Adam([p for name, p in online_network.named_parameters()
        if any(x in name for x in ['q1_net', 'q2_net'])],
                                        lr=critic_lr,
                                        eps=config_copy.adam_epsilon,
                                        betas=(config_copy.adam_beta1, config_copy.adam_beta2),
                                        weight_decay=1e-5)
    critic_scaler = torch.amp.GradScaler(
        "cuda",
        init_scale=2**10,  # Start smaller
        growth_factor=1.5,  # Grow more slowly (default is 2)
        backoff_factor=0.5,  # Back off more aggressively
        growth_interval=2000  # Increase scale less frequently
    )

    policy_optimizer = torch.optim.Adam([p for name, p in online_network.named_parameters()
        if any(x in name for x in ['action_logits'])],
                                        lr=policy_lr,
                                        eps=config_copy.adam_epsilon,
                                        betas=(config_copy.adam_beta1, config_copy.adam_beta2),
                                        weight_decay=1e-5)
    policy_scaler = torch.amp.GradScaler(
        "cuda",
        init_scale=2**10,  # Start smaller
        growth_factor=1.5,  # Grow more slowly (default is 2)
        backoff_factor=0.5,  # Back off more aggressively
        growth_interval=2000  # Increase scale less frequently
    )

    alpha_optimizer = torch.optim.Adam([log_alpha],
                                       lr=alpha_lr,
                                       eps=config_copy.adam_epsilon,
                                       betas=(config_copy.adam_beta1, config_copy.adam_beta2))
    alpha_scaler = torch.amp.GradScaler(
        "cuda",
        init_scale=2**10,  # Start smaller
        growth_factor=1.5,  # Grow more slowly (default is 2)
        backoff_factor=0.5,  # Back off more aggressively
        growth_interval=2000  # Increase scale less frequently
    )

    memory_size, memory_size_start_learn = utilities.from_staircase_schedule(
        config_copy.memory_size_schedule, accumulated_stats["cumul_number_memories_generated"]
    )
    buffer, buffer_test = make_buffers(memory_size)
    offset_cumul_number_single_memories_used = memory_size_start_learn * config_copy.number_times_single_memory_is_used_before_discard

    # noinspection PyBroadException
    try:
        policy_optimizer.load_state_dict(torch.load(f=save_dir / "policy_optimizer.torch", weights_only=False))
        critic_optimizer.load_state_dict(torch.load(f=save_dir / "critic_optimizer.torch", weights_only=False))
        alpha_optimizer.load_state_dict(torch.load(f=save_dir / "alpha_optimizer.torch", weights_only=False))
        policy_scaler.load_state_dict(torch.load(f=save_dir / "policy_scaler.torch", weights_only=False))
        critic_scaler.load_state_dict(torch.load(f=save_dir / "critic_scaler.torch", weights_only=False))
        alpha_scaler.load_state_dict(torch.load(f=save_dir / "alpha_scaler.torch", weights_only=False))
        print(" =========================     Optimizers loaded !     ================================")
    except:
        print(" Could not load optimizer")

    tensorboard_suffix = utilities.from_staircase_schedule(
        config_copy.tensorboard_suffix_schedule,
        accumulated_stats["cumul_number_memories_generated"],
    )
    tensorboard_writer = SummaryWriter(log_dir=str(tensorboard_base_dir / (config_copy.run_name + tensorboard_suffix)))

    critic_loss_history = []
    policy_loss_history = []
    alpha_loss_history = []
    entropy_history = []
    critic_loss_test_history = []
    policy_loss_test_history = []
    alpha_loss_test_history = []
    entropy_test_history = []
    train_on_batch_duration_history = []

    # ========================================================
    # Make the trainer
    # ========================================================
    trainer = sac.Trainer(
        network=online_network,
        policy_optimizer=policy_optimizer,
        critic_optimizer=critic_optimizer,
        alpha_optimizer=alpha_optimizer,
        policy_scaler=policy_scaler,
        critic_scaler=critic_scaler,
        alpha_scaler=alpha_scaler,
        batch_size=config_copy.batch_size,
        log_alpha=log_alpha,
    )

    inferer = sac.Inferer(
        inference_network=online_network
    )

    while True:  # Trainer loop
        before_wait_time = time.perf_counter()
        wait(rollout_queue_readers)
        time_waited = time.perf_counter() - before_wait_time
        if time_waited > 1:
            print(f"Warning: learner waited {time_waited:.2f} seconds for workers to provide memories")
        time_waited_for_workers_since_last_tensorboard_write += time_waited
        for idx in queue_check_order:
            if not rollout_queues[idx].empty():
                (
                    rollout_results,
                    end_race_stats,
                    fill_buffer,
                    is_explo,
                    map_name,
                    map_status,
                    rollout_duration,
                    loop_number,
                ) = rollout_queues[idx].get()
                queue_check_order.append(queue_check_order.pop(queue_check_order.index(idx)))
                break

        importlib.reload(config_copy)

        new_tensorboard_suffix = utilities.from_staircase_schedule(
            config_copy.tensorboard_suffix_schedule,
            accumulated_stats["cumul_number_memories_generated"],
        )
        if new_tensorboard_suffix != tensorboard_suffix:
            tensorboard_suffix = new_tensorboard_suffix
            tensorboard_writer = SummaryWriter(log_dir=str(tensorboard_base_dir / (config_copy.run_name + tensorboard_suffix)))

        (
            new_memory_size,
            new_memory_size_start_learn,
        ) = utilities.from_staircase_schedule(
            config_copy.memory_size_schedule,
            accumulated_stats["cumul_number_memories_generated"],
        )
        if new_memory_size != memory_size:
            buffer, buffer_test = resize_buffers(buffer, buffer_test, new_memory_size)
            offset_cumul_number_single_memories_used += (
                new_memory_size_start_learn - memory_size_start_learn
            ) * config_copy.number_times_single_memory_is_used_before_discard
            memory_size_start_learn = new_memory_size_start_learn
            memory_size = new_memory_size
        # ===============================================
        #   VERY BASIC TRAINING ANNEALING
        # ===============================================

        gamma = utilities.from_linear_schedule(config_copy.gamma_schedule, accumulated_stats["cumul_number_memories_generated"])

        # ===============================================
        #   RELOAD
        # ===============================================

        for param_group in policy_optimizer.param_groups:
            param_group["lr"] = policy_lr
            param_group["epsilon"] = config_copy.adam_epsilon
            param_group["betas"] = (config_copy.adam_beta1, config_copy.adam_beta2)
        for param_group in critic_optimizer.param_groups:
            param_group["lr"] = critic_lr
            param_group["epsilon"] = config_copy.adam_epsilon
            param_group["betas"] = (config_copy.adam_beta1, config_copy.adam_beta2)
        for param_group in alpha_optimizer.param_groups:
            param_group["lr"] = alpha_lr
            param_group["epsilon"] = config_copy.adam_epsilon
            param_group["betas"] = (config_copy.adam_beta1, config_copy.adam_beta2)

        if isinstance(buffer._sampler, PrioritizedSampler):
            buffer._sampler._alpha = config_copy.prio_alpha
            buffer._sampler._beta = config_copy.prio_beta
            buffer._sampler._eps = config_copy.prio_epsilon

        if config_copy.plot_race_time_left_curves and not is_explo and (loop_number // 5) % 17 == 0:
            race_time_left_curves(rollout_results, inferer, save_dir, map_name)
            tau_curves(rollout_results, inferer, save_dir, map_name)
            # distribution_curves(buffer, save_dir, online_network, target_network)
            # loss_distribution(buffer, save_dir, online_network, target_network)
            # patrick_curves(rollout_results, trainer, save_dir, map_name)

        accumulated_stats["cumul_number_frames_played"] += len(rollout_results["frames"])

        # ===============================================
        #   WRITE SINGLE RACE RESULTS TO TENSORBOARD
        # ===============================================
        race_stats_to_write = {
            f"race_time_ratio_{map_name}": end_race_stats["race_time_for_ratio"] / (rollout_duration * 1000),
            f"explo_race_time_{map_status}_{map_name}" if is_explo else f"eval_race_time_{map_status}_{map_name}": end_race_stats[
                "race_time"
            ]
            / 1000,
            f"explo_race_finished_{map_status}_{map_name}" if is_explo else f"eval_race_finished_{map_status}_{map_name}": end_race_stats[
                "race_finished"
            ],
            f"single_zone_reached_{map_status}_{map_name}": rollout_results["furthest_zone_idx"],
            "instrumentation__answer_normal_step": end_race_stats["instrumentation__answer_normal_step"],
            "instrumentation__answer_action_step": end_race_stats["instrumentation__answer_action_step"],
            "instrumentation__between_run_steps": end_race_stats["instrumentation__between_run_steps"],
            "instrumentation__grab_frame": end_race_stats["instrumentation__grab_frame"],
            "instrumentation__convert_frame": end_race_stats["instrumentation__convert_frame"],
            "instrumentation__grab_floats": end_race_stats["instrumentation__grab_floats"],
            "instrumentation__exploration_policy": end_race_stats["instrumentation__exploration_policy"],
            "instrumentation__request_inputs_and_speed": end_race_stats["instrumentation__request_inputs_and_speed"],
            "tmi_protection_cutoff": end_race_stats["tmi_protection_cutoff"],
            "worker_time_in_rollout_percentage": rollout_results["worker_time_in_rollout_percentage"],
        }
        print("Race time ratio  ", race_stats_to_write[f"race_time_ratio_{map_name}"])

        if not is_explo:
            race_stats_to_write[f"avg_Q_{map_status}_{map_name}"] = np.mean(rollout_results["q_values"])

        if end_race_stats["race_finished"]:
            race_stats_to_write[f"{'explo' if is_explo else 'eval'}_race_time_finished_{map_status}_{map_name}"] = (
                end_race_stats["race_time"] / 1000
            )
            if not is_explo:
                accumulated_stats["rolling_mean_ms"][map_name] = (
                    accumulated_stats["rolling_mean_ms"].get(map_name, config_copy.cutoff_rollout_if_race_not_finished_within_duration_ms)
                    * 0.9
                    + end_race_stats["race_time"] * 0.1
                )
        if (
            (not is_explo)
            and end_race_stats["race_finished"]
            and end_race_stats["race_time"] < 1.02 * accumulated_stats["rolling_mean_ms"][map_name]
        ):
            race_stats_to_write[f"eval_race_time_robust_{map_status}_{map_name}"] = end_race_stats["race_time"] / 1000
            if map_name in reference_times:
                for reference_time_name in ["author", "gold"]:
                    if reference_time_name in reference_times[map_name]:
                        reference_time = reference_times[map_name][reference_time_name]
                        race_stats_to_write[f"eval_ratio_{map_status}_{reference_time_name}_{map_name}"] = (
                            100 * (end_race_stats["race_time"] / 1000) / reference_time
                        )
                        race_stats_to_write[f"eval_agg_ratio_{map_status}_{reference_time_name}"] = (
                            100 * (end_race_stats["race_time"] / 1000) / reference_time
                        )

        if not is_explo:
            for i, split_time in enumerate(
                [
                    (e - s) / 1000
                    for s, e in zip(
                        end_race_stats["cp_time_ms"][:-1],
                        end_race_stats["cp_time_ms"][1:],
                    )
                ]
            ):
                race_stats_to_write[f"split_{map_name}_{i}"] = split_time

        walltime_tb = time.time()
        for tag, value in race_stats_to_write.items():
            tensorboard_writer.add_scalar(
                tag=tag,
                scalar_value=value,
                global_step=accumulated_stats["cumul_number_frames_played"],
                walltime=walltime_tb,
            )

        # ===============================================
        #   SAVE STUFF IF THIS WAS A GOOD RACE
        # ===============================================

        if end_race_stats["race_time"] < accumulated_stats["alltime_min_ms"].get(map_name, 99999999999):
            # This is a new alltime_minimum
            accumulated_stats["alltime_min_ms"][map_name] = end_race_stats["race_time"]
            if accumulated_stats["cumul_number_frames_played"] > config_copy.frames_before_save_best_runs:
                name = f"{map_name}_{end_race_stats['race_time']}"
                utilities.save_run(
                    base_dir,
                    save_dir / "best_runs" / name,
                    rollout_results,
                    f"{name}.inputs",
                    inputs_only=False,
                )
                utilities.save_checkpoint(
                    save_dir / "best_runs",
                    online_network,
                    policy_optimizer,
                    critic_optimizer,
                    alpha_optimizer,
                    policy_scaler,
                    critic_scaler,
                    alpha_scaler,
                )

        if end_race_stats["race_time"] < config_copy.threshold_to_save_all_runs_ms:
            name = f"{map_name}_{end_race_stats['race_time']}_{datetime.now().strftime('%m%d_%H%M%S')}_{accumulated_stats['cumul_number_frames_played']}_{'explo' if is_explo else 'eval'}"
            utilities.save_run(
                base_dir,
                save_dir / "good_runs",
                rollout_results,
                f"{name}.inputs",
                inputs_only=True,
            )

        # ===============================================
        #   FILL BUFFER WITH (S, A, R, S', d) transitions
        # ===============================================
        if fill_buffer:
            (
                buffer,
                buffer_test,
                number_memories_added_train,
                number_memories_added_test,
            ) = buffer_management.fill_buffer_from_rollout_with_n_steps_rule(
                buffer,
                buffer_test,
                rollout_results,
                end_race_stats,
                config_copy.n_steps,
                gamma
            )

            accumulated_stats["cumul_number_memories_generated"] += number_memories_added_train + number_memories_added_test
            shared_steps.value = accumulated_stats["cumul_number_memories_generated"]
            neural_net_reset_counter += number_memories_added_train
            accumulated_stats["cumul_number_single_memories_should_have_been_used"] += (
                config_copy.number_times_single_memory_is_used_before_discard * number_memories_added_train
            )
            print(f" NMG={accumulated_stats['cumul_number_memories_generated']:<8}")

            # ===============================================
            #   LEARN ON BATCH
            # ===============================================

            if not online_network.training:
                online_network.train()

            while (
                len(buffer) >= memory_size_start_learn
                and accumulated_stats["cumul_number_single_memories_used"] + offset_cumul_number_single_memories_used
                <= accumulated_stats["cumul_number_single_memories_should_have_been_used"]
            ):
                if (random.random() < config_copy.buffer_test_ratio and len(buffer_test) > 0) or len(buffer) == 0:
                    test_start_time = time.perf_counter()
                    critic_loss, policy_loss, alpha_loss, entropy = trainer.train_on_batch(buffer_test, do_learn=False)
                    time_testing_since_last_tensorboard_write += time.perf_counter() - test_start_time

                    # Store all test metrics
                    critic_loss_test_history.append(critic_loss)
                    policy_loss_test_history.append(policy_loss)
                    alpha_loss_test_history.append(alpha_loss)
                    entropy_test_history.append(entropy)

                    print(f"BT   {critic_loss=:<8.2e}, {policy_loss=:<8.2e}, {alpha_loss=:<8.2e}, {entropy=:<8.2e}")
                else:
                    train_start_time = time.perf_counter()
                    # Unpack all losses from the trainer
                    critic_loss, policy_loss, alpha_loss, entropy = trainer.train_on_batch(buffer, do_learn=True)
                    train_on_batch_duration_history.append(time.perf_counter() - train_start_time)
                    time_training_since_last_tensorboard_write += train_on_batch_duration_history[-1]
                    accumulated_stats["cumul_number_single_memories_used"] += (
                        4 * config_copy.batch_size
                        if (len(buffer) < buffer._storage.max_size and buffer._storage.max_size > 200_000)
                        else config_copy.batch_size
                    )

                    # Store all metrics in history
                    critic_loss_history.append(critic_loss)
                    policy_loss_history.append(policy_loss)
                    alpha_loss_history.append(alpha_loss)
                    entropy_history.append(entropy)

                    accumulated_stats["cumul_number_batches_done"] += 1
                    print(f"B    {critic_loss=:<8.2e}, {policy_loss=:<8.2e}, {alpha_loss=:<8.2e}, {entropy=:<8.2e}, {train_on_batch_duration_history[-1] * 1000:<8.1f}")

                    if accumulated_stats["cumul_number_batches_done"] % config_copy.send_shared_network_every_n_batches == 0:
                        with shared_network_lock:
                            uncompiled_shared_network.load_state_dict(uncompiled_online_network.state_dict())
            sys.stdout.flush()

        # ===============================================
        #   WRITE AGGREGATED STATISTICS TO TENSORBOARD EVERY 5 MINUTES # 1 minute
        # ===============================================
        save_frequency_s = 1 * 60 # 5 * 60
        if time.perf_counter() - time_last_save >= save_frequency_s:
            accumulated_stats["cumul_training_hours"] += (time.perf_counter() - time_last_save) / 3600
            time_since_last_save = time.perf_counter() - time_last_save
            waited_percentage = time_waited_for_workers_since_last_tensorboard_write / time_since_last_save
            trained_percentage = time_training_since_last_tensorboard_write / time_since_last_save
            tested_percentage = time_testing_since_last_tensorboard_write / time_since_last_save
            time_waited_for_workers_since_last_tensorboard_write = 0
            time_training_since_last_tensorboard_write = 0
            time_testing_since_last_tensorboard_write = 0
            transitions_learned_per_second = (
                accumulated_stats["cumul_number_single_memories_used"] - transitions_learned_last_save
            ) / time_since_last_save
            time_last_save = time.perf_counter()
            transitions_learned_last_save = accumulated_stats["cumul_number_single_memories_used"]

            # ===============================================
            #   COLLECT VARIOUS STATISTICS
            # ===============================================
            step_stats = {
                "gamma": gamma,
                "n_steps": config_copy.n_steps,
                "epsilon": utilities.from_exponential_schedule(config_copy.epsilon_schedule, shared_steps.value),
                "policy_lr": policy_optimizer.param_groups[0]["lr"],
                "critic_lr": critic_optimizer.param_groups[0]["lr"],
                "alpha_lr": alpha_optimizer.param_groups[0]["lr"],
                "alpha_value": torch.exp(log_alpha).item(),
                "discard_non_greedy_actions_in_nsteps": config_copy.discard_non_greedy_actions_in_nsteps,
                "memory_size": len(buffer),
                "number_times_single_memory_is_used_before_discard": config_copy.number_times_single_memory_is_used_before_discard,
                "learner_percentage_waiting_for_workers": waited_percentage,
                "learner_percentage_training": trained_percentage,
                "learner_percentage_testing": tested_percentage,
                "transitions_learned_per_second": transitions_learned_per_second,
            }
            print(len(critic_loss_history), len(critic_loss_test_history))
            if len(critic_loss_history) > 0 and len(critic_loss_test_history) > 0:

                step_stats.update({
                    "critic_loss": np.mean(critic_loss_history),
                    "critic_loss_test": np.mean(critic_loss_test_history),
                    "policy_loss": np.mean(policy_loss_history),
                    "policy_loss_test": np.mean(policy_loss_test_history),
                    "alpha_loss": np.mean(alpha_loss_history),
                    "alpha_loss_test": np.mean(alpha_loss_test_history),
                    "alpha_t": np.mean(entropy_history),
                    "alpha_t_test": np.mean(entropy_test_history),
                    "train_on_batch_duration": np.median(train_on_batch_duration_history),
                })

            if online_network.training:
                online_network.eval()

            critic_loss_history = []
            policy_loss_history = []
            alpha_loss_history = []
            entropy_history = []
            critic_loss_test_history = []
            policy_loss_test_history = []
            alpha_loss_test_history = []
            entropy_test_history = []
            train_on_batch_duration_history = []

            # ===============================================
            #   WRITE TO TENSORBOARD
            # ===============================================

            walltime_tb = time.time()
            for name, param in online_network.named_parameters():
                # Categorize parameters by network type
                if any(x in name for x in ['policy_mean', 'policy_logprob']):
                    prefix = "policy"
                elif any(x in name for x in ['q1_net', 'q2_net']):
                    prefix = "q"
                else:
                    prefix = "shared"
                tensorboard_writer.add_scalar(
                    tag=f"{prefix}/{name}_L2",
                    scalar_value=np.sqrt((param ** 2).mean().detach().cpu().item()),
                    global_step=accumulated_stats["cumul_number_frames_played"],
                    walltime=walltime_tb,
                )

            tensorboard_writer.add_scalar(
                tag="alpha/value",
                scalar_value=torch.exp(trainer.log_alpha).item(),
                global_step=accumulated_stats["cumul_number_frames_played"],
                walltime=walltime_tb,
            )

            for k, v in step_stats.items():
                tensorboard_writer.add_scalar(
                    tag=k,
                    scalar_value=v,
                    global_step=accumulated_stats["cumul_number_frames_played"],
                    walltime=walltime_tb,
                )

            previous_alltime_min = previous_alltime_min or copy.deepcopy(accumulated_stats["alltime_min_ms"])

            tensorboard_writer.add_text(
                "times_summary",
                f"{datetime.now().strftime('%Y/%m/%d, %H:%M:%S')} "
                + " ".join(
                    [
                        f"{'**' if v < previous_alltime_min.get(k, 99999999) else ''}{k}: {v / 1000:.2f}{'**' if v < previous_alltime_min.get(k, 99999999) else ''}"
                        for k, v in accumulated_stats["alltime_min_ms"].items()
                    ]
                ),
                global_step=accumulated_stats["cumul_number_frames_played"],
                walltime=walltime_tb,
            )

            previous_alltime_min = copy.deepcopy(accumulated_stats["alltime_min_ms"])

            # ===============================================
            #   BUFFER STATS
            # ===============================================

            state_floats = np.array([experience.state_float for experience in buffer._storage])
            mean_in_buffer = state_floats.mean(axis=0)
            std_in_buffer = state_floats.std(axis=0)

            print("Raw mean in buffer  :", mean_in_buffer.round(1))
            print("Raw std in buffer   :", std_in_buffer.round(1))
            print("")
            print(
                "Corr mean in buffer :",
                ((mean_in_buffer - config_copy.float_inputs_mean) / config_copy.float_inputs_std).round(1),
            )
            print(
                "Corr std in buffer  :",
                (std_in_buffer / config_copy.float_inputs_std).round(1),
            )
            print("")

            # ===============================================
            #   HIGH PRIORITY TRANSITIONS
            # ===============================================
            if config_copy.make_highest_prio_figures and isinstance(buffer._sampler, PrioritizedSampler):
                highest_prio_transitions(buffer, save_dir)

            # ===============================================
            #   SAVE
            # ===============================================
            utilities.save_checkpoint(save_dir, online_network, policy_optimizer, critic_optimizer, alpha_optimizer, policy_scaler, critic_scaler, alpha_scaler)
            joblib.dump(accumulated_stats, save_dir / "accumulated_stats.joblib")
