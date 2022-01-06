import os
import sys
import shutil
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as mtrans
from matplotlib.ticker import FormatStrFormatter, FuncFormatter, SymmetricalLogLocator
from tensorboard.backend.event_processing import event_accumulator


def running_average(nums, horizon=10):
    new_nums = [0] * len(nums)
    new_nums[0] = nums[0]
    for i in range(1, len(nums)):
        new_nums[i] = new_nums[i - 1] + nums[i]
        if i >= horizon:
            new_nums[i] -= nums[i - horizon]

    for i in range(len(nums)):
        new_nums[i] = new_nums[i] / min(i + 1, horizon)

    return new_nums


def error_plot(fig_name, iter_list, tv_errs):
    plt.plot(iter_list, np.mean(tv_errs, axis=0), color='red')
    plt.xlabel("Training Iteration")
    plt.ylabel("Average Total MAE")
    plt.tight_layout()
    plt.savefig(fig_name)
    plt.clf()


def gaps_plot(fig_name, iter_list, ate_gaps, percentiles, zoom_bounds=None, sep_bounds=None, sep_colors=None):
    ate_gap_percentiles = np.percentile(ate_gaps, percentiles, axis=0)

    plt.gca().set_prop_cycle(plt.cycler('color', plt.cm.jet(np.linspace(0, 1, len(percentiles)))))
    if zoom_bounds is not None:
        plt.gca().set_ylim(zoom_bounds)
    plt.plot(iter_list, ate_gap_percentiles.T)
    plt.axhline(y=0.0, color='k', linestyle='-')
    if sep_bounds is not None:
        for i, b in enumerate(sep_bounds):
            plt.axhline(y=b, color=sep_colors[i], linestyle='--')
    plt.xlabel("Training Iteration")
    plt.ylabel("Max ATE - Min ATE")
    plt.legend(percentiles)
    plt.tight_layout()
    plt.savefig(fig_name)
    plt.clf()


def id_acc_plot(fig_name, iter_list, gaps_ucb_list, is_id, boundaries, run_avg=None):
    plt.gca().set_prop_cycle(plt.cycler('color', plt.cm.jet(np.linspace(0, 1, len(boundaries)))))
    plt.gca().set_ylim([0.0, 1.01])
    for b in boundaries:
        gaps_ucb_sep = []
        if isinstance(gaps_ucb_list, dict):
            for exp_type in gaps_ucb_list:
                gaps = gaps_ucb_list[exp_type]
                if exp_type in id_folders:
                    result = (gaps <= b).astype(int)
                else:
                    result = (gaps > b).astype(int)
                gaps_ucb_sep.append(result)
            gaps_ucb_sep = np.concatenate(gaps_ucb_sep, axis=0)
        else:
            if is_id:
                gaps_ucb_sep = (gaps_ucb_list <= b).astype(int)
            else:
                gaps_ucb_sep = (gaps_ucb_list > b).astype(int)
        acc_list = np.mean(gaps_ucb_sep, axis=0)

        if run_avg is not None:
            acc_list = running_average(acc_list, horizon=run_avg)

        plt.plot(iter_list, acc_list)
    plt.xlabel("Training Iteration")
    plt.ylabel("Correct ID %")
    plt.legend(boundaries)
    plt.tight_layout()
    plt.savefig(fig_name)
    plt.clf()


def double_row_plot(fig_name, iter_list, ate_gaps_dict, gaps_ucb_dict, run_avg=None):
    order = ["backdoor", "frontdoor", "m", "napkin", "placeholder", "bow", "extended_bow", "iv", "bad_m_2"]
    fig, axes = plt.subplots(2, len(order), sharex=True, sharey='row',
                             gridspec_kw=dict(width_ratios=[8, 8, 8, 8, 1, 8, 8, 8, 8]))

    for g_ind, graph in enumerate(order):
        if g_ind == 4:
            continue

        ax_gaps = axes[1][g_ind]
        ax_acc = axes[0][g_ind]
        ate_gaps = ate_gaps_dict[graph]
        gaps_ucb_list = gaps_ucb_dict[graph]

        ate_gap_percentiles = np.percentile(ate_gaps, percentiles, axis=0)
        ax_gaps.set_prop_cycle(plt.cycler('color', plt.cm.jet(np.linspace(0, 1, len(percentiles)))))
        ax_gaps.plot(iter_list, ate_gap_percentiles.T)
        ax_gaps.axhline(y=0.0, color='k', linestyle='-')
        ax_gaps.tick_params(axis='both', which='major', labelsize=24)
        ax_gaps.tick_params(axis='both', which='minor', labelsize=18)

        ax_acc.set_prop_cycle(plt.cycler('color', plt.cm.jet(np.linspace(0, 1, len(boundaries)))))
        ax_acc.set_ylim([0.0, 1.01])
        for b in boundaries:
            gaps_ucb_sep = []
            if graph in id_folders:
                gaps_ucb_sep = (gaps_ucb_list <= b).astype(int)
            else:
                gaps_ucb_sep = (gaps_ucb_list > b).astype(int)
            acc_list = np.mean(gaps_ucb_sep, axis=0)

            if run_avg is not None:
                acc_list = running_average(acc_list, horizon=run_avg)

            ax_acc.plot(iter_list, acc_list)
        ax_acc.tick_params(axis='both', which='major', labelsize=24)
        ax_acc.tick_params(axis='both', which='minor', labelsize=18)

    axes[0][4].remove()
    axes[1][4].remove()

    axes[1][0].set_ylabel("Max ATE - Min ATE", fontsize=24)
    axes[0][0].set_ylabel("Correct ID %", fontsize=24)

    trans = mtrans.blended_transform_factory(fig.transFigure,
                                             mtrans.IdentityTransform())
    xlab = fig.text(.5, 15, "Training Iteration", ha='center', fontsize=24)
    xlab.set_transform(trans)

    fig.set_figheight(10)
    fig.set_figwidth(41)
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.05, hspace=0.1, bottom=0.1)
    fig.savefig(fig_name, bbox_inches='tight')
    fig.clf()


parser = argparse.ArgumentParser(description="ID Experiment Results Parser")
parser.add_argument('dir', help="directory of the experiment")
parser.add_argument('--clean', action="store_true",
                    help="delete unfinished experiments")
args = parser.parse_args()

d = args.dir

id_folders = {"ID", "simple", "m", "backdoor", "frontdoor", "napkin"}
nonid_folders = {"nonID", "bow", "iv", "bdm", "bad_m_2", "extended_bow"}

boundaries = [0.01, 0.03, 0.05]
b_colors = ['r', 'g', 'b']
percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]

iters_counted = False
iter_list = []
id_ate_gaps = []
nonid_ate_gaps = []
all_ate_gaps = dict()
all_gap_ucbs = dict()

os.makedirs("{}/figs".format(d), exist_ok=True)
for exp_type in os.listdir(d):
    if exp_type not in id_folders and exp_type not in nonid_folders:
        print("\nSkipping {} directory.".format(exp_type))
        continue
    print("\nScanning {} experiments...".format(exp_type))

    tv_errs = []
    gaps_ucb_list = []
    graph_ate_gaps = []
    for t in os.listdir("{}/{}".format(d, exp_type)):
        if os.path.isdir("{}/{}/{}".format(d, exp_type, t)):
            temp_gap_list = []
            temp_tv_err_list = []
            for r in os.listdir("{}/{}/{}".format(d, exp_type, t)):
                if os.path.isdir("{}/{}/{}/{}".format(d, exp_type, t, r)):
                    if os.path.isdir("{}/{}/{}/{}/logs".format(d, exp_type, t, r)) \
                            and not os.path.exists("{}/{}/{}/{}/results.json".format(d, exp_type, t, r)):
                        if args.clean:
                            print("Trial {}, run {} is incomplete. Deleting contents...".format(t, r))
                            shutil.rmtree("{}/{}/{}/{}".format(d, exp_type, t, r))
                            if os.path.exists("{}/{}/{}/lock".format(d, exp_type, t)):
                                os.remove("{}/{}/{}/lock".format(d, exp_type, t))
                            if os.path.exists("{}/{}/{}/best.th".format(d, exp_type, t)):
                                os.remove("{}/{}/{}/best.th".format(d, exp_type, t))
                        else:
                            print("Trial {}, run {} is incomplete.".format(t, r))

                    else:
                        min_dir = "{}/{}/{}/{}/logs/default/version_0".format(d, exp_type, t, r)
                        max_dir = "{}/{}/{}/{}/logs/default/version_1".format(d, exp_type, t, r)

                        if os.path.isdir(min_dir) and os.path.isdir(max_dir):
                            min_event = None
                            max_event = None
                            for item in os.listdir(min_dir):
                                if min_event is None and "events" in item:
                                    min_event = item
                            for item in os.listdir(max_dir):
                                if max_event is None and "events" in item:
                                    max_event = item
                            ea_min = event_accumulator.EventAccumulator("{}/{}".format(min_dir, min_event))
                            ea_max = event_accumulator.EventAccumulator("{}/{}".format(max_dir, max_event))
                            ea_min.Reload()
                            ea_max.Reload()
                            min_ate_events = ea_min.Scalars('min_ncm_ate')
                            min_tv_err_events = ea_min.Scalars('min_err_dat_tv_ncm_tv')
                            max_ate_events = ea_max.Scalars('max_ncm_ate')
                            max_tv_err_events = ea_max.Scalars('max_err_dat_tv_ncm_tv')

                            try:
                                min_max_gaps = []
                                total_tv_err = []
                                for i in range(len(min_ate_events)):
                                    iter = min_ate_events[i].step
                                    min_ate = min_ate_events[i].value
                                    max_ate = max_ate_events[i].value
                                    min_tv_err = abs(min_tv_err_events[i].value)
                                    max_tv_err = abs(max_tv_err_events[i].value)

                                    if not iters_counted:
                                        iter_list.append(iter + 1)
                                    min_max_gaps.append(max_ate - min_ate)
                                    total_tv_err.append(min_tv_err + max_tv_err)

                                temp_gap_list.append(min_max_gaps)
                                temp_tv_err_list.append(total_tv_err)
                                iters_counted = True
                            except Exception as e:
                                print("Error in trial {}, run {}.".format(t, r))
                                print(e)

            if len(temp_gap_list) > 0:
                temp_gap_list = np.array(temp_gap_list)
                temp_tv_err_list = np.array(temp_tv_err_list)
                tv_errs.append(np.mean(temp_tv_err_list, axis=0))
                gaps_means = np.mean(temp_gap_list, axis=0)
                if exp_type in id_folders:
                    id_ate_gaps.append(gaps_means)
                elif exp_type in nonid_folders:
                    nonid_ate_gaps.append(gaps_means)
                graph_ate_gaps.append(gaps_means)

                if len(temp_gap_list) > 1:
                    gaps_stderr = np.std(temp_gap_list, axis=0) / np.sqrt(len(temp_gap_list))
                    gaps_ucb = gaps_means + 1.65 * gaps_stderr
                    gaps_ucb_list.append(gaps_ucb)
                    if exp_type not in all_gap_ucbs:
                        all_gap_ucbs[exp_type] = []
                    all_gap_ucbs[exp_type].append(gaps_ucb)

    tv_errs = np.array(tv_errs)
    all_ate_gaps[exp_type] = graph_ate_gaps

    # Plot TV error per graph
    error_plot("{}/figs/{}_errors.png".format(d, exp_type), iter_list, tv_errs)

    # Plot gaps per graph
    gaps_plot("{}/figs/{}_gap_percentiles.png".format(d, exp_type), iter_list, graph_ate_gaps, percentiles)

    # Plot accuracy per graph
    if len(gaps_ucb_list) > 0:
        gaps_ucb_list = np.array(gaps_ucb_list)
        id_acc_plot("{}/figs/{}_ID_classification.png".format(d, exp_type), iter_list, gaps_ucb_list,
                    exp_type in id_folders, boundaries, run_avg=None)
        id_acc_plot("{}/figs/{}_ID_classification_10runavg.png".format(d, exp_type), iter_list, gaps_ucb_list,
                    exp_type in id_folders, boundaries, run_avg=10)
        all_gap_ucbs[exp_type] = np.array(all_gap_ucbs[exp_type])

iter_list = np.array(iter_list)
id_ate_gaps = np.array(id_ate_gaps)
nonid_ate_gaps = np.array(nonid_ate_gaps)

# Plot gaps overall
gaps_plot("{}/figs/ID_gap_percentiles.png".format(d), iter_list, id_ate_gaps, percentiles,
          sep_bounds=boundaries, sep_colors=b_colors)
gaps_plot("{}/figs/nonID_gap_percentiles.png".format(d), iter_list, nonid_ate_gaps, percentiles,
          sep_bounds=boundaries, sep_colors=b_colors)
gaps_plot("{}/figs/ID_gap_percentiles_zoomed.png".format(d), iter_list, id_ate_gaps, percentiles,
          zoom_bounds=[-0.01, 0.04], sep_bounds=boundaries, sep_colors=b_colors)
gaps_plot("{}/figs/nonID_gap_percentiles_zoomed.png".format(d), iter_list, nonid_ate_gaps, percentiles,
          zoom_bounds=[-0.01, 0.04], sep_bounds=boundaries, sep_colors=b_colors)

# Plot accuracy overall
id_acc_plot("{}/figs/overall_ID_classification.png".format(d), iter_list, all_gap_ucbs,
            False, boundaries, run_avg=None)
id_acc_plot("{}/figs/overall_ID_classification_10runavg.png".format(d), iter_list, all_gap_ucbs,
            False, boundaries, run_avg=10)

# Plot grid of results
double_row_plot("{}/figs/ID_grid.png".format(d), iter_list, all_ate_gaps, all_gap_ucbs, run_avg=10)
