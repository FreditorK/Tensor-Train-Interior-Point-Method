import numpy as np

def print_results_summary(config, args, runtimes, problem_creation_times,
                          num_iters, feasibility_errors, complementary_slackness,
                          ranksX, ranksY, ranksZ, memory=None):
    print("\n" + "=" * 80)
    print(f"{'FINAL RESULTS SUMMARY':^80}")
    print("=" * 80)
    print("Means and standard deviations (Std) are calculated over all seeds.\n")

    for r_i, rank in enumerate(config["max_ranks"]):
        # --- Calculate Means ---
        mean_runtime = np.mean(runtimes[r_i, :])
        mean_creation_time = np.mean(problem_creation_times[r_i, :])
        mean_iters = np.mean(num_iters[r_i, :])
        mean_feasibility = np.mean(feasibility_errors[r_i, :])
        mean_slackness = np.mean(complementary_slackness[r_i, :])

        # --- Calculate Standard Deviations ---
        std_runtime = np.std(runtimes[r_i, :])
        std_creation_time = np.std(problem_creation_times[r_i, :])
        std_iters = np.std(num_iters[r_i, :])
        std_feasibility = np.std(feasibility_errors[r_i, :])
        std_slackness = np.std(complementary_slackness[r_i, :])

        # --- Print Table for the Current Rank ---
        print(f"--- Rank: {rank} ---")
        print(f"  {'Metric':<28} | {'Mean':>12} | {'Std Dev':>12}")
        print(f"  {'-' * 28} | {'-' * 12} | {'-' * 12}")
        print(f"  {'Solution Time (s)':<28} | {mean_runtime:12.3f} | {std_runtime:12.3f}")
        print(f"  {'Problem Creation (s)':<28} | {mean_creation_time:12.3f} | {std_creation_time:12.3f}")
        print(f"  {'Iterations':<28} | {mean_iters:12.1f} | {std_iters:12.1f}")
        print(f"  {'Feasibility Error':<28} | {mean_feasibility:12.4e} | {std_feasibility:12.4e}")
        print(f"  {'Complementary Slackness':<28} | {mean_slackness:12.4e} | {std_slackness:12.4e}")

        if args.track_mem and memory is not None:
            mean_mem = np.mean(memory[r_i, :])
            std_mem = np.std(memory[r_i, :])
            print(f"  {'Peak Memory (MB)':<28} | {mean_mem:12.3f} | {std_mem:12.3f}")

        # --- Calculate and Print Average Rank Arrays ---
        avg_ranks_X = np.mean(ranksX[r_i, :, :], axis=0)
        avg_ranks_Y = np.mean(ranksY[r_i, :, :], axis=0)
        avg_ranks_Z = np.mean(ranksZ[r_i, :, :], axis=0)

        print(f"  {'-' * 28} | {'-' * 12} | {'-' * 12}")
        print(f"  {'Avg Ranks X':<28}: {np.array2string(avg_ranks_X, precision=1, floatmode='fixed', separator=', ')}")
        print(f"  {'Avg Ranks Y':<28}: {np.array2string(avg_ranks_Y, precision=1, floatmode='fixed', separator=', ')}")
        print(f"  {'Avg Ranks Z':<28}: {np.array2string(avg_ranks_Z, precision=1, floatmode='fixed', separator=', ')}")
        print("")

    print("=" * 80)