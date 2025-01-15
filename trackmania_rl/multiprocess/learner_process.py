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
from datetime import datetime
from multiprocessing.connection import wait
from pathlib import Path

import joblib
import numpy as np
import torch
import torch_optimizer
from torch import multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
from torchrl.data.replay_buffers import PrioritizedSampler

from config_files import config_copy
from trackmania_rl import buffer_management, utilities
from trackmania_rl.agents import iqn as iqn
from trackmania_rl.analysis_metrics import (
    distribution_curves,
    highest_prio_transitions,
    loss_distribution,
    race_time_left_curves,
    tau_curves,
)
from trackmania_rl.buffer_utilities import make_buffers, resize_buffers
from trackmania_rl.map_reference_times import reference_times


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
        "eval_race_time_finished": [
            "Multiline",
            [
                "eval_race_time_finished",
            ],
        ],
        "explo_race_time_finished": [
            "Multiline",
            [
                "explo_race_time_finished",
            ],
        ],
        "loss_Q": ["Multiline", ["loss_Q$", "loss_Q_test$"]],
        "loss_policy": ["Multiline", ["loss_policy$", "loss_policy_test$"]],
        "loss_alpha": ["Multiline", ["loss_alpha$", "loss_alpha_test$"]],
        "values_starting_frame": [
            "Multiline",
            [f"q_value_starting_frame_{i}" for i in range(len(config_copy.inputs))],
        ],
        "policy_starting_frame": [
            "Multiline",
            [f"policy_{i}_starting_frame" for i in range(len(config_copy.inputs))],
        ],
        "single_zone_reached": [
            "Multiline",
            [
                "single_zone_reached",
            ],
        ],
        r"races_finished": ["Multiline", ["explo_race_finished", "eval_race_finished"]],
        "iqn_std": [
            "Multiline",
            [f"std_within_iqn_quantiles_for_action{i}" for i in range(len(config_copy.inputs))],
        ],
        "race_time_ratio": ["Multiline", ["race_time_ratio"]],
        "mean_action_gap": [
            "Multiline",
            [
                "mean_action_gap",
            ],
        ],
        "layer_L2": [
            "Multiline",
            [
                "layer_.*_L2",
            ],
        ],
        "lr_ratio_L2": [
            "Multiline",
            [
                "lr_ratio_.*_L2",
            ],
        ],
        "exp_avg_L2": [
            "Multiline",
            [
                "exp_avg_.*_L2",
            ],
        ],
        "exp_avg_sq_L2": [
            "Multiline",
            [
                "exp_avg_sq_.*_L2",
            ],
        ],
        "eval_race_time": [
            "Multiline",
            [
                "eval_race_time_[^_]*",
            ],
        ],
        "explo_race_time": [
            "Multiline",
            [
                "explo_race_time_[^_]*",
            ],
        ],
        "sac_alpha": [
            "Multiline",
            [
                "sac_alpha",
            ],
        ],
        "policy_entropy": [
            "Multiline",
            [
                "policy_entropy",
                "policy_entropy",
            ],
        ],
    },
        }
    )

    # ========================================================
    # Create new stuff
    # ========================================================

    soft_Q_model1, _ = iqn.make_untrained_SoftIQNQNetwork()
    soft_Q_model2, _ = iqn.make_untrained_SoftIQNQNetwork()
    policy_model, uncompiled_policy_model = iqn.make_untrained_PolicyNetwork()
    logalpha_model, _ = iqn.make_untrained_LogAlphaSingletonNetwork()

    print(soft_Q_model1)
    print(policy_model)
    print(logalpha_model)

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
        soft_Q_model1.load_state_dict(torch.load(save_dir / "soft_Q_weights1.torch"))
        soft_Q_model2.load_state_dict(torch.load(save_dir / "soft_Q_weights2.torch"))
        policy_model.load_state_dict(torch.load(save_dir / "policy_weights.torch"))
        logalpha_model.load_state_dict(torch.load(save_dir / "logalpha_weights.torch"))
        print(" =========================     Weights loaded !     ================================")
    except:
        print(" Could not load weights")

    with shared_network_lock:
        uncompiled_shared_network.load_state_dict(uncompiled_policy_model.state_dict())

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


    soft_Q_optimizer = torch.optim.RAdam(
            soft_Q_model1.parameters(),
            lr=utilities.from_staircase_schedule(config_copy.lr_schedule, accumulated_stats["cumul_number_memories_generated"]),
            eps=config_copy.adam_epsilon,
            betas=(0.9, 0.95),
        )
    soft_Q_optimizer = torch_optimizer.Lookahead(soft_Q_optimizer, k=5, alpha=0.5)

    policy_optimizer = torch.optim.RAdam(
            policy_model.parameters(),
            lr=utilities.from_staircase_schedule(config_copy.lr_schedule, accumulated_stats["cumul_number_memories_generated"]),
            eps=config_copy.adam_epsilon,
            betas=(0.9, 0.95),
        )
    policy_optimizer = torch_optimizer.Lookahead(policy_optimizer, k=5, alpha=0.5)

    logalpha_optimizer = torch.optim.RAdam(
        logalpha_model.parameters(),
        lr=utilities.from_staircase_schedule(config_copy.lr_schedule, accumulated_stats["cumul_number_memories_generated"]),
        eps=config_copy.adam_epsilon,
        betas=(0.9, 0.95),
    )

    soft_Q_scaler = torch.amp.GradScaler("cuda")
    policy_scaler = torch.amp.GradScaler("cuda")
    logalpha_scaler = torch.amp.GradScaler("cuda")

    memory_size, memory_size_start_learn = utilities.from_staircase_schedule(
        config_copy.memory_size_schedule, accumulated_stats["cumul_number_memories_generated"]
    )

    buffer, buffer_test = make_buffers(memory_size)
    offset_cumul_number_single_memories_used = memory_size_start_learn * config_copy.number_times_single_memory_is_used_before_discard

    # noinspection PyBroadException
    try:
        soft_Q_optimizer.load_state_dict(torch.load(save_dir / "soft_Q_optimizer.torch"))
        soft_Q_scaler.load_state_dict(torch.load(save_dir / "soft_Q_scaler.torch"))
        policy_optimizer.load_state_dict(torch.load(save_dir / "policy_optimizer.torch"))
        policy_scaler.load_state_dict(torch.load(save_dir / "policy_scaler.torch"))
        # logalpha_optimizer.load_state_dict(torch.load(save_dir / "logalpha_optimizer.torch"))
        # logalpha_scaler.load_state_dict(torch.load(save_dir / "logalpha_scaler.torch"))
        print(" =========================     Optimizer loaded !     ================================")
    except:
        print(" Could not load optimizer")

    tensorboard_suffix = utilities.from_staircase_schedule(
        config_copy.tensorboard_suffix_schedule,
        accumulated_stats["cumul_number_memories_generated"],
    )
    tensorboard_writer = SummaryWriter(log_dir=str(tensorboard_base_dir / (config_copy.run_name + tensorboard_suffix)))

    loss_Q_history = []
    loss_Q_test_history = []
    loss_policy_history = []
    loss_policy_test_history = []
    loss_alpha_history = []
    loss_alpha_test_history = []
    policy_entropy_history = []
    policy_entropy_test_history = []
    train_on_batch_duration_history = []
    grad_norm_history = []
    layer_grad_norm_history = defaultdict(list)

    # ========================================================
    # Make the trainer
    # ========================================================
    gamma = utilities.from_linear_schedule(config_copy.gamma_schedule,
                                              accumulated_stats["cumul_number_memories_generated"])

    trainer = iqn.Trainer(
        soft_Q_model=soft_Q_model1,
        soft_Q_model2=soft_Q_model2,
        soft_Q_optimizer=soft_Q_optimizer,
        soft_Q_scaler=soft_Q_scaler,
        policy_model=policy_model,
        policy_optimizer=policy_optimizer,
        policy_scaler=policy_scaler,
        logalpha_model=logalpha_model,
        logalpha_optimizer=logalpha_optimizer,
        logalpha_scaler=logalpha_scaler,
        batch_size=config_copy.batch_size,
        iqn_k=config_copy.iqn_k,
        iqn_n=config_copy.iqn_n,
        iqn_kappa=config_copy.iqn_kappa,
        gamma=utilities.from_linear_schedule(config_copy.gamma_schedule,
                                              accumulated_stats["cumul_number_memories_generated"]),
        truncation_amplitude=config_copy.truncation_amplitude,
        target_entropy=config_copy.target_entropy,  # This parameter is typically set to dim(action_space)
        epsilon=utilities.from_staircase_schedule(config_copy.epsilon_schedule,
                                              accumulated_stats["cumul_number_memories_generated"]),
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

        # LR and weight_decay calculation
        learning_rate = utilities.from_exponential_schedule(config_copy.lr_schedule, accumulated_stats["cumul_number_memories_generated"])
        weight_decay = config_copy.weight_decay_lr_ratio * learning_rate
        engineered_speedslide_reward = utilities.from_linear_schedule(
            config_copy.engineered_speedslide_reward_schedule,
            accumulated_stats["cumul_number_memories_generated"],
        )
        engineered_neoslide_reward = utilities.from_linear_schedule(
            config_copy.engineered_neoslide_reward_schedule,
            accumulated_stats["cumul_number_memories_generated"],
        )
        engineered_kamikaze_reward = utilities.from_linear_schedule(
            config_copy.engineered_kamikaze_reward_schedule, accumulated_stats["cumul_number_memories_generated"]
        )
        engineered_close_to_vcp_reward = utilities.from_linear_schedule(
            config_copy.engineered_close_to_vcp_reward_schedule, accumulated_stats["cumul_number_memories_generated"]
        )
        gamma = utilities.from_linear_schedule(config_copy.gamma_schedule, accumulated_stats["cumul_number_memories_generated"])

        # ===============================================
        #   RELOAD
        # ===============================================

        for param_group in soft_Q_optimizer.param_groups:
            param_group["lr"] = learning_rate
        for param_group in policy_optimizer.param_groups:
            param_group["lr"] = learning_rate * config_copy.lr_policy_ratio
        for param_group in logalpha_optimizer.param_groups:
            param_group["lr"] = learning_rate * config_copy.lr_alpha_ratio

        if isinstance(buffer._sampler, PrioritizedSampler):
            buffer._sampler._alpha = config_copy.prio_alpha
            buffer._sampler._beta = config_copy.prio_beta
            buffer._sampler._eps = config_copy.prio_epsilon


        # if config_copy.plot_race_time_left_curves and not is_explo and (loop_number // 5) % 17 == 0:
        #     race_time_left_curves(rollout_results, inferer, save_dir, map_name)
        #    tau_curves(rollout_results, inferer, save_dir, map_name)
        #    distribution_curves(buffer, save_dir, online_network, target_network)
        #    loss_distribution(buffer, save_dir, online_network, target_network)
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
            f"mean_action_gap_{map_name}": -(
                np.array(rollout_results["q_values"]) - np.array(rollout_results["q_values"]).max(axis=1, initial=None).reshape(-1, 1)
            ).mean(),
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

        for i in [0]:
            race_stats_to_write[f"q_value_{i}_starting_frame_{map_name}"] = end_race_stats[f"q_value_{i}_starting_frame"]
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
                    target_network,
                    optimizer1,
                    scaler,
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
        #   FILL BUFFER WITH (S, A, R, S') transitions
        # ===============================================
        if fill_buffer:
            (
                buffer,
                buffer_test,
                number_memories_added,
            ) = buffer_management.fill_buffer_from_rollout_with_n_steps_rule(
                buffer,
                buffer_test,
                rollout_results,
                config_copy.n_steps,
                gamma,
                config_copy.discard_non_greedy_actions_in_nsteps,
                config_copy.n_zone_centers_in_inputs
            )

            accumulated_stats["cumul_number_memories_generated"] += number_memories_added
            accumulated_stats["reset_counter"] += number_memories_added
            accumulated_stats["cumul_number_single_memories_should_have_been_used"] += (
                    config_copy.number_times_single_memory_is_used_before_discard * number_memories_added
            )
            print(f" NMG={accumulated_stats['cumul_number_memories_generated']:<8}")

            # ===============================================
            #   PERIODIC RESET ?
            # ===============================================

            # if neural_net_reset_counter >= config_copy.reset_every_n_frames_generated or single_reset_flag != config_copy.single_reset_flag:
            #     neural_net_reset_counter = 0
            #     single_reset_flag = config_copy.single_reset_flag
            #     accumulated_stats["cumul_number_single_memories_should_have_been_used"] += config_copy.additional_transition_after_reset
            #
            #     _, untrained_iqn_network = make_untrained_iqn_network(config_copy.use_jit, False)
            #     utilities.soft_copy_param(online_network, untrained_iqn_network, config_copy.overall_reset_mul_factor)
            #
            #     with torch.no_grad():
            #         online_network.A_head[2].weight = utilities.linear_combination(
            #             online_network.A_head[2].weight,
            #             untrained_iqn_network.A_head[2].weight,
            #             config_copy.last_layer_reset_factor,
            #         )
            #         online_network.A_head[2].bias = utilities.linear_combination(
            #             online_network.A_head[2].bias,
            #             untrained_iqn_network.A_head[2].bias,
            #             config_copy.last_layer_reset_factor,
            #         )
            #         online_network.V_head[2].weight = utilities.linear_combination(
            #             online_network.V_head[2].weight,
            #             untrained_iqn_network.V_head[2].weight,
            #             config_copy.last_layer_reset_factor,
            #         )
            #         online_network.V_head[2].bias = utilities.linear_combination(
            #             online_network.V_head[2].bias,
            #             untrained_iqn_network.V_head[2].bias,
            #             config_copy.last_layer_reset_factor,
            #         )

            # ===============================================
            #   LEARN ON BATCH
            # ===============================================

            if not online_network.training:
                soft_Q_model1.train()
                policy_model.train()
                logalpha_model.train()

            while (
                len(buffer) >= memory_size_start_learn
                and accumulated_stats["cumul_number_single_memories_used"] + offset_cumul_number_single_memories_used
                <= accumulated_stats["cumul_number_single_memories_should_have_been_used"]
            ):
                if (random.random() < config_copy.buffer_test_ratio and len(buffer_test) > 0) or len(buffer) == 0:
                    loss_Q, loss_policy, loss_alpha, policy_entropy = trainer.train_on_batch(buffer_test, do_learn=False)
                    loss_Q_test_history.append(loss_Q)
                    loss_policy_test_history.append(loss_policy)
                    loss_alpha_test_history.append(loss_alpha)
                    policy_entropy_test_history.append(policy_entropy)
                    print(f"BT   {loss_Q=:<8.2e} {loss_policy=:<8.2e} {loss_alpha=:<8.2e}")
                else:
                    train_start_time = time.time()
                    loss_Q, loss_policy, loss_alpha, policy_entropy = trainer.train_on_batch(buffer, do_learn=True)
                    accumulated_stats["cumul_number_single_memories_used"] += config_copy.batch_size
                    train_on_batch_duration_history.append(time.time() - train_start_time)
                    loss_Q_history.append(loss_Q)
                    loss_policy_history.append(loss_policy)
                    loss_alpha_history.append(loss_alpha)
                    policy_entropy_history.append(policy_entropy)

                    # if not math.isinf(grad_norm):
                    #     grad_norm_history.append(grad_norm)
                    #     # utilities.log_gradient_norms(online_network, layer_grad_norm_history) #~1ms overhead per batch

                    accumulated_stats["cumul_number_batches_done"] += 1
                    print(f"B    {loss_Q=:<8.2e} {loss_policy=:<8.2e} {loss_alpha=:<8.2e}")

                    utilities.custom_weight_decay(soft_Q_model1, 1 - weight_decay)
                    utilities.custom_weight_decay(policy_model, 1 - weight_decay)

                    # ===============================================
                    #   UPDATE TARGET NETWORK
                    # ===============================================
                    if (
                            accumulated_stats["cumul_number_single_memories_used"]
                            >= accumulated_stats["cumul_number_single_memories_used_next_target_network_update"]
                    ):
                        accumulated_stats["cumul_number_target_network_updates"] += 1
                        accumulated_stats[
                            "cumul_number_single_memories_used_next_target_network_update"
                        ] += config_copy.number_memories_trained_on_between_target_network_updates
                        print("UPDATE")
                        utilities.soft_copy_param(soft_Q_model2, soft_Q_model1, config_copy.soft_update_tau)
            sys.stdout.flush()

        # ===============================================
        #   WRITE AGGREGATED STATISTICS TO TENSORBOARD EVERY 5 MINUTES
        # ===============================================
        save_frequency_s = 5 * 60
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
                "epsilon_boltzmann": utilities.from_exponential_schedule(config_copy.epsilon_boltzmann_schedule, shared_steps.value),
                "tau_epsilon_boltzmann": config_copy.tau_epsilon_boltzmann,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "discard_non_greedy_actions_in_nsteps": config_copy.discard_non_greedy_actions_in_nsteps,
                "memory_size": len(buffer),
                "number_times_single_memory_is_used_before_discard": config_copy.number_times_single_memory_is_used_before_discard,
                "learner_percentage_waiting_for_workers": waited_percentage,
                "learner_percentage_training": trained_percentage,
                "learner_percentage_testing": tested_percentage,
                "transitions_learned_per_second": transitions_learned_per_second,
            }
            if len(loss_history) > 0 and len(loss_test_history) > 0:
                step_stats.update(
                    {
                        "loss": np.mean(loss_history),
                        "loss_test": np.mean(loss_test_history),
                        "train_on_batch_duration": np.median(train_on_batch_duration_history),
                        "grad_norm_history_q1": np.quantile(grad_norm_history, 0.25),
                        "grad_norm_history_median": np.quantile(grad_norm_history, 0.5),
                        "grad_norm_history_q3": np.quantile(grad_norm_history, 0.75),
                        "grad_norm_history_d9": np.quantile(grad_norm_history, 0.9),
                        "grad_norm_history_d98": np.quantile(grad_norm_history, 0.98),
                        "grad_norm_history_max": np.max(grad_norm_history),
                    }
                )
                for key, val in layer_grad_norm_history.items():
                    step_stats.update(
                        {
                            f"{key}_median": np.quantile(val, 0.5),
                            f"{key}_q3": np.quantile(val, 0.75),
                            f"{key}_d9": np.quantile(val, 0.9),
                            f"{key}_c98": np.quantile(val, 0.98),
                            f"{key}_max": np.max(val),
                        }
                    )
            if isinstance(buffer._sampler, PrioritizedSampler):
                all_priorities = np.array([buffer._sampler._sum_tree.at(i) for i in range(len(buffer))])
                step_stats.update(
                    {
                        "priorities_min": np.min(all_priorities),
                        "priorities_q1": np.quantile(all_priorities, 0.1),
                        "priorities_mean": np.mean(all_priorities),
                        "priorities_median": np.quantile(all_priorities, 0.5),
                        "priorities_q3": np.quantile(all_priorities, 0.75),
                        "priorities_d9": np.quantile(all_priorities, 0.9),
                        "priorities_c98": np.quantile(all_priorities, 0.98),
                        "priorities_max": np.max(all_priorities),
                    }
                )
            for key, value in accumulated_stats.items():
                if key not in ["alltime_min_ms", "rolling_mean_ms"]:
                    step_stats[key] = value
            for key, value in accumulated_stats["alltime_min_ms"].items():
                step_stats[f"alltime_min_ms_{map_name}"] = value

            loss_history = []
            loss_test_history = []
            train_on_batch_duration_history = []
            grad_norm_history = []
            layer_grad_norm_history = defaultdict(list)

            # ===============================================
            #   COLLECT IQN SPREAD
            # ===============================================

            if online_network.training:
                online_network.eval()
            tau = torch.linspace(0.05, 0.95, config_copy.iqn_k)[:, None].to("cuda")

            # per_quantile_output = inferer.infer_network(rollout_results["frames"][0], rollout_results["state_float"][0], tau)
            # for i, std in enumerate(list(per_quantile_output.std(axis=0))):
            #     step_stats[f"std_within_iqn_quantiles_for_action{i}"] = std

            # ===============================================
            #   WRITE TO TENSORBOARD
            # ===============================================

            walltime_tb = time.time()
            for name, param in online_network.named_parameters():
                tensorboard_writer.add_scalar(
                    tag=f"layer_{name}_L2",
                    scalar_value=np.sqrt((param**2).mean().detach().cpu().item()),
                    global_step=accumulated_stats["cumul_number_frames_played"],
                    walltime=walltime_tb,
                )
            assert len(optimizer1.param_groups) == 1
            try:
                for p, (name, _) in zip(
                    optimizer1.param_groups[0]["params"],
                    online_network.named_parameters(),
                ):
                    state = optimizer1.state[p]
                    exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                    mod_lr = 1 / (exp_avg_sq.sqrt() + 1e-4)
                    tensorboard_writer.add_scalar(
                        tag=f"lr_ratio_{name}_L2",
                        scalar_value=np.sqrt((mod_lr**2).mean().detach().cpu().item()),
                        global_step=accumulated_stats["cumul_number_frames_played"],
                        walltime=walltime_tb,
                    )
                    tensorboard_writer.add_scalar(
                        tag=f"exp_avg_{name}_L2",
                        scalar_value=np.sqrt((exp_avg**2).mean().detach().cpu().item()),
                        global_step=accumulated_stats["cumul_number_frames_played"],
                        walltime=walltime_tb,
                    )
                    tensorboard_writer.add_scalar(
                        tag=f"exp_avg_sq_{name}_L2",
                        scalar_value=np.sqrt((exp_avg_sq**2).mean().detach().cpu().item()),
                        global_step=accumulated_stats["cumul_number_frames_played"],
                        walltime=walltime_tb,
                    )
            except:
                pass

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
            utilities.save_checkpoint(save_dir, online_network, target_network, optimizer1, scaler)
            joblib.dump(accumulated_stats, save_dir / "accumulated_stats.joblib")
