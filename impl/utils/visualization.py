import logging
import os
import pickle
import time
from bisect import bisect_left
from collections import defaultdict
from itertools import product
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, MultipleLocator
from scipy.spatial.distance import squareform
from scipy.stats import rankdata
from sklearn.manifold import MDS

from impl.config import CONFIG
from impl.mr.mr import Relation
from impl.utils.math_utils import calculate_auc_improvements, area_under_curve, calculate_ds_improvements
from impl.utils.metrics import metrics, pairwise_distance, avg_pw_from_matrix

logger = logging.getLogger(__name__)

verbose_map = {
    "pop": "Population",
    "pop_scen": "Population—Scenario",
    "pop_pert": "Population—Perturbation",
    "solution": "Complete Solutions",
    "archive": "Archive",
    "arc_scen": "Archive—Scenario",
    "arc_pert": "Archive—Perturbation",
    "ccea": "CoCoMEGA",
    "ccea-d": "CoCoMEGA\d",
    "rs": "RS",
    "ga": "SGA",
    "gawa": "SGA with Archives",
    "distinct_solution_num": "Average $DS$",
    "avg_fit": "Average Fitness",
    "avg_pw": "APD",
    "pure_div": "Pure Diversity",
}

style_map = {
    "ccea": {"color": "C1", "marker": "o"},
    "ccea-d": {"color": "C3", "marker": "P"},
    "ga": {"color": "C2", "marker": "*"},
    "rs": {"color": "C0", "marker": "x"}
}

skip_map = {"ccea": 5, "ccea-d": 5, "ga": 2, "rs": 1}


class Visualizer:
    """
        Visualization module with statistics data.
    """

    @classmethod
    def plot_violation_monitor(cls, stats, show, out_dir, title=None, name=None, verbose_name=None):
        pass

    @classmethod
    def visualize_gen_stats(cls, stats, show, out_dir):
        """
            Visualization of generation statistics data.
            :param stats: A dictionary containing lists of statistics per generation.
                              Expected keys are 'gen' for generation numbers,
                              'avg' for average fitness, 'max' for maximum fitness, etc.
            :param show: A boolean to determine whether to show the plots or not.
            :param out_dir: Output directory of plots
        """
        stats = pd.DataFrame(stats)
        if len(stats) == 0:
            logger.warning('No statistics provided. Nothing to visualize.')
            return

        metrics = [col for col in stats.columns if col not in ['pop', 'gen', 'len', 'sim']]
        for metric in metrics:
            stats[metric] = stats[metric].apply(lambda x: x[0])

        gen_stat_figs = cls._plot_generation_statistics(stats, out_dir)
        if show:
            for fig in gen_stat_figs:
                cls._show_plot(fig)

    @classmethod
    def plot_histogram(cls, stats, color, show, out_dir, title=None, name=None, verbose_name=None):
        fig = go.Figure()

        # Adding histogram trace
        fig.add_trace(go.Histogram(
            x=stats,
            histnorm='percent',
            marker=dict(
                color=np.where(color, '#DC143C', '#4682B4')
            ),
        ))

        fig.update_layout(
            title_text=title if title else 'Multicolored Histogram',
            xaxis_title_text=verbose_name if verbose_name else 'Value',
            yaxis_title_text='Percent',
            bargap=0,
        )

        if show:
            cls._show_plot(fig)

        cls._save_fig(fig, out_dir, f'hist_{name}_{int(round(time.time() * 1000))}.png')

    @classmethod
    def plot_pert_boundary_results(cls, meta, out_dir):
        for i, row in meta.iterrows():
            path = os.path.join(CONFIG["workspace"]["solution"], row['statistics_path'])
            cls.visualize_algorithm_Stats(path, name=row['name'], out_dir=out_dir)

    @classmethod
    def visualize_algorithm_Stats(cls, path, name, out_dir):
        with open(path, "rb") as f:
            logbook = pickle.load(f)
        stats = pd.DataFrame(logbook)
        for metric in ["std", "min", "avg", "max"]:
            stats[metric] = stats[metric].apply(lambda x: x[0])
        verbose = {"pop_scen": "Population—Scenario",
                   "pop_pert": "Population—Perturbation",
                   "arc_scen": "Archive—Scenario",
                   "arc_pert": "Archive—Perturbation",
                   "solution": "Complete Solutions"}

        fig, axs = plt.subplots(2, 3, figsize=[20, 8])
        for (pop_name, pop), pos in zip(stats.groupby("pop", sort=False), [(0, 1), (0, 2), (0, 0), (1, 1), (1, 2)]):
            pop = pop.sort_values("gen", ascending=True)
            axs[pos].plot(pop["gen"], pop["std"], "--C0", label="std")
            axs[pos].plot(pop["gen"], pop["min"], ":C1", label="min")
            axs[pos].plot(pop["gen"], pop["avg"], "o-C2", label="avg")
            axs[pos].plot(pop["gen"], pop["max"], ":C3", label="max")
            axs[pos].set_title(verbose[pop_name], fontsize=20)
            axs[pos].set_xlabel("Generations", fontsize=15)
            axs[pos].set_ylabel("Fitness", fontsize=15)
            axs[pos].tick_params(labelsize=13)
            axs[pos].xaxis.set_major_locator(MaxNLocator(integer=True))

            if pop_name == "solution":
                pos = (1, 0)
                axs[pos].plot(pop["gen"], pop["len"], "o-C4", label="#violations")
                # axs[pos].plot(pop["gen"], data[0]["num"], "o-C5", label="#simulations")
                axs[pos].set_title("#Violations", fontsize=20)
                axs[pos].set_xlabel("Generations", fontsize=15)
                axs[pos].set_ylabel("Num", fontsize=15)
                axs[pos].tick_params(labelsize=13)
                axs[pos].xaxis.set_major_locator(MaxNLocator(integer=True))
                axs[pos].legend(fontsize=15)

        handles, labels = axs[pos].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.52, -0.01), ncol=4, fontsize=15)
        fig.tight_layout()
        plt.subplots_adjust(bottom=0.13)
        plt.savefig(os.path.join(out_dir, f"stats_{name}.png"), dpi=300)

    @classmethod
    def _plot_generation_statistics(cls, stats, out_dir):
        """
            Generates line plots for evolutionary algorithm statistics across generations.
        """
        timestamp = int(round(time.time() * 1000))
        figs = []
        for pop_name, pop in stats.groupby('pop'):
            pop = pop.sort_values('gen', ascending=True)
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=pop['gen'],
                y=pop['min'],
                mode='lines+markers',
                name='Min Fitness',
                marker=dict(
                    color='red',
                    opacity=0.6,
                ),
            ))
            fig.add_trace(go.Scatter(
                x=pop['gen'],
                y=pop['max'],
                mode='lines+markers',
                name='Max Fitness',
                marker=dict(
                    color='green',
                    opacity=0.6,
                ),
            ))
            fig.add_trace(go.Scatter(
                x=pop['gen'],
                y=pop['avg'],
                mode='lines+markers',
                name='Average Fitness',
                marker=dict(
                    color='blue',
                    opacity=0.6,
                ),
            ))

            fig.update_layout(
                title=f'Evolutionary Algorithm Generation Statistics for {verbose_map[pop_name]} population',
                xaxis_title='Generation',
                yaxis_title='Fitness',
                legend_title='Metrics',
                xaxis_range=[-0.5, max(pop['gen']) + 0.5],
            )

            cls._save_fig(fig, out_dir=out_dir, name=f'{pop_name}_{int(round(time.time() * 1000))}.png')
            figs.append(fig)
        return figs

    @staticmethod
    def _show_plot(fig):
        """
            Displays the plot.
        """
        fig.show()

    @staticmethod
    def _save_fig(fig, out_dir, name):
        """
            Saves figure as image
        """
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        fig.write_image(os.path.join(out_dir, name))


def visualize_in_one(data, file_name=None, plot_nan=True, verbose=False, show=False):
    """Plot all statistics data in one figure.

    @param data: Statistics data or data file.
    @param file_name: Specify the file name for plots when the data is not a file path.
    @param plot_nan: Plot NaN values.
    @param verbose: Show plots of populations and archives.
    @param show: A boolean to determine whether to show the plots or not.
    """
    if isinstance(data, str):
        file_name = Path(data).stem
        with open(data, "rb") as f:
            data = pickle.load(f)
    stats = pd.DataFrame(data)
    if len(stats) == 0:
        logger.warning('No statistics provided. Nothing to visualize.')
        return
    for metric in ("std", "min", "avg", "max"):
        stats[metric] = stats[metric].apply(lambda x: x[0])
    if plot_nan:
        stats.fillna(0, inplace=True)

    groups = stats.groupby("pop")
    assert "solution" in groups.groups
    # Log statistics.
    if "archive" not in groups.groups:
        logger.warning("Cannot find statistics of archives.")
        return
    pop = groups.get_group("archive").sort_values("gen", ascending=True).fillna(0)
    logger.info(f"#violations of the last archive: {pop['len'].iloc[-1]}.")
    logger.info(f"Max fitness of the last archive: {pop['max'].iloc[-1]}.")
    logger.info(f"Average fitness of the last archive: {pop['avg'].iloc[-1]}.")
    if len(pop["gen"]) > 1:
        logger.info("Growth rate of average fitness over generations: "
                    f"{np.polyfit(pop['gen'], pop['avg'], 1)[0]}.")
        logger.info("Growth rate of max fitness over generations: "
                    f"{np.polyfit(pop['gen'], pop['max'], 1)[0]}.")
        logger.info("Growth rate of average fitness over simulations: " +
                    f"{np.polyfit(pop['sim'], pop['avg'], 1)[0]}.")
        logger.info("Growth rate of max fitness over simulations: " +
                    f"{np.polyfit(pop['sim'], pop['max'], 1)[0]}.")

    core_plots = [name for name in ("solution", "solution", "archive") if name in groups.groups]
    addition_plots = [name for name in ("pop_scen", "arc_scen", "pop_pert", "arc_pert") if name in groups.groups]
    plots = core_plots + addition_plots if verbose else core_plots
    row_num = int(np.ceil(len(plots) / 3))
    column_num = len(plots) if row_num == 1 else 3
    height = 4
    title_size, text_size, tick_size = height * 5, height * 4, height * 3

    def _plot_metrics(_ax, _pop, _name, xlabel=False, ylabel=False, legend=False):
        for _metric, _fmt in [("std", ":C0"), ("min", "--C1"), ("avg", "o-C2"), ("max", "--C3")]:
            _ax.plot(_pop["gen"], _pop[_metric], _fmt, label=_metric)
        _ax.set_title(verbose_map[_name], fontsize=title_size)
        _ax.tick_params(labelsize=tick_size)
        _ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        if xlabel: _ax.set_xlabel("Generation", fontsize=text_size)
        if ylabel: _ax.set_ylabel("Fitness", fontsize=text_size)
        if legend: _ax.legend(fontsize=text_size)

    fig = plt.figure(figsize=(height * 1.5 * column_num, height * row_num))
    axs = []
    for i, name in enumerate(plots):
        ax = fig.add_subplot(row_num, column_num, i + 1,
                             sharex=axs[0] if i > 0 else None,
                             sharey=axs[1] if i > 1 else None)
        axs.append(ax)
        pop = groups.get_group(name).sort_values("gen", ascending=True)
        if i > 0:
            _plot_metrics(ax, pop, name, legend=i == 1, ylabel=(i == 1 or i % 3 == 0))
        else:
            ax.plot(pop["gen"], pop["len"], "o-C4", label="#violations")
            ax.plot(pop["gen"], pop["sim"], "x-C5", label="#simulations")
            ax.set_title("#violations & #simulations", fontsize=title_size)
            ax.set_ylabel("Num", fontsize=text_size)
            ax.tick_params(labelsize=tick_size)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            ax.legend(fontsize=text_size)

    fig.supxlabel("Generation", fontsize=text_size)
    fig.tight_layout()
    fig.savefig(os.path.join(CONFIG["workspace"]["visualization"], f"{file_name}.png"))
    if show: plt.show()


def visualize_comparison(files: Dict[str, List[str]], max_percentile=0.75,
                         box=True, interval=15, avg_line=False,
                         trend_line=True, regression_degree=3, all_lines=False,
                         plot_nan=True, verbose=False, show=False):
    """Plot comparisons among different algorithms.

    @param files: Statistics data files of different algorithms.
    @param max_percentile: Plot the given percentile of max fitness.
    @param box: Show box plots.
    @param interval: Width of intervals for aggregation.
    @param avg_line: Show average lines.
    @param trend_line: Show trend lines.
    @param regression_degree: Degree of regression.
    @param all_lines: Show original lines.
    @param plot_nan: Plot NaN values.
    @param verbose: Show more plots.
    @param show: A boolean to determine whether to show the plots or not.
    """
    data = {}
    for name, file_list in files.items():
        full, merged, agg = defaultdict(list), {}, {}
        low, high = float("inf"), float("-inf")
        for file in file_list:
            with open(file, "rb") as f:
                df = pd.DataFrame(pickle.load(f))
            groups = df.groupby("pop")
            df_solution = groups.get_group("solution").sort_values("gen", ascending=True)
            df_archive = groups.get_group("archive").sort_values("gen", ascending=True)
            for df, pop_name in ((df_solution, "solution"), (df_archive, "archive")):
                for metric in ("std", "min", "avg", "max"):
                    df[metric] = df[metric].apply(lambda x: x[0])
                if plot_nan:
                    df.fillna(0, inplace=True)
                full[pop_name].append(df)
            df_solution["sim"] = df_solution["sim"].cumsum()
            low = min(low, int(np.ceil(df["sim"].iloc[0] / interval)) * interval)
            high = max(high, int(np.floor(df["sim"].iloc[-1] / interval)) * interval)

        helper = pd.DataFrame({"sim": range(low, high + 1, interval)})
        for pop_name, df_list in full.items():
            df = pd.concat(df_list).sort_values("sim", ascending=True)
            merged[pop_name] = df
            agg_list = []
            for df in df_list:
                df = df.groupby("sim", as_index=False).mean()
                agg_df = pd.merge(df, helper, on="sim", how="outer").sort_values("sim", ascending=True)
                agg_df.set_index("sim", inplace=True)
                for metric in ("len", "std", "min", "avg", "max"):
                    agg_df[metric].interpolate("index", inplace=True)
                agg_list.append(agg_df.loc[agg_df.index % interval == 0])
            df = pd.concat(agg_list).groupby("sim").agg(list)
            df["max"] = df["max"].apply(lambda x: [i for i in x if i >= np.quantile(x, max_percentile)])
            agg[pop_name] = df
        data[name] = (full, merged, agg)

    height = 4
    title_size, text_size, tick_size = height * 5, height * 4, height * 3
    row_num = 5 if verbose else 2
    fig = plt.figure(figsize=(height * 4, height * row_num))
    ax1 = fig.add_subplot(row_num, 1, 1)
    ax2 = fig.add_subplot(row_num, 1, 2, sharex=ax1)
    plots = [(ax1, "len", "archive", "#violations"),
             (ax2, "max", "solution",
              f"Max fitness{f' (percentile: {max_percentile})' if max_percentile > 0 else ''}")]
    if verbose:
        ax3 = fig.add_subplot(row_num, 1, 3, sharex=ax1)
        ax4 = fig.add_subplot(row_num, 1, 4, sharex=ax1, sharey=ax2)
        ax5 = fig.add_subplot(row_num, 1, 5, sharex=ax1, sharey=ax3)
        plots += [(ax3, "avg", "solution", "Average fitness"),
                  (ax4, "max", "archive",
                   f"Max fitness of archives{f' (percentile: {max_percentile})' if max_percentile > 0 else ''}"),
                  (ax5, "avg", "archive", "Average fitness of archives")]
    legend_elements = {}

    for ax, metric, pop_name, title in plots:
        for name, (full, merged, agg) in data.items():
            full, merged, agg = full[pop_name], merged[pop_name], agg[pop_name]
            if box:
                ax.boxplot(agg[metric], positions=agg.index.values, widths=2, patch_artist=True, manage_ticks=False,
                           whis=(0, 100), boxprops=dict(facecolor=style_map[name]["color"], alpha=0.4))
            if avg_line: ax.plot(agg.index.values, agg[metric].apply(np.nanmean), **style_map[name])
            if trend_line:
                f = np.poly1d(np.polyfit(merged["sim"], merged[metric], regression_degree))
                ax.plot(merged["sim"], f(merged["sim"]), lw=2, color=style_map[name]["color"])
            if all_lines:
                for line in full:
                    ax.plot(line["sim"], line[metric], "--", lw=1, **style_map[name], alpha=0.6)
            if name not in legend_elements:
                legend_elements[name] = Line2D([0], [0], **style_map[name], label=verbose_map[name])
        ax.set_title(title, fontsize=title_size)
        ax.tick_params(labelsize=tick_size)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_ylabel("#violations" if ax == ax1 else "Fitness", fontsize=text_size)
        ax.grid()

    ax1.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.legend(handles=legend_elements.values(), fontsize=text_size)
    fig.supxlabel("#simulations", fontsize=text_size)
    fig.tight_layout()
    fig.savefig(os.path.join(CONFIG["workspace"]["visualization"], "comparison.png"))
    if show: plt.show()


def visualize_violation(source, follow_up, mr_set, offset=3, verbose=False, show=False):
    """Plot the extent of violation between the source results
    and follow-up results based on the given metamorphic relations.

    @param source: The source results.
    @param follow_up: The follow-up results.
    @param mr_set: The given metamorphic relations.
    @param offset: The offset between the source and follow-up curves.
    @param verbose: Plot the DTW path and matches of positions.
    @param show: A boolean to determine whether to show the plots or not.
    """
    labels = Relation.convert_labels(mr_set.labels)
    matches, origin_df = Relation.pairwise_dataframe(source, follow_up, mr_set.field, labels)
    if CONFIG["violation"]["dtw"]:
        matches, pair_df = Relation.dtw_dataframe(source, follow_up, mr_set.field, labels)
    else:
        pair_df = origin_df.reset_index()

    row_num = 7 if verbose and CONFIG["violation"]["dtw"] else 4
    column_num, height = 6, 3
    title_size, text_size, tick_size = height * 7, height * 5, height * 4
    fig = plt.figure(figsize=(height * column_num, height * row_num))
    gs = GridSpec(row_num, column_num, figure=fig)
    ax1 = fig.add_subplot(gs[0:2, :])
    ax2 = fig.add_subplot(gs[2:4, :], sharey=ax1)

    for ax, df, title in zip((ax1, ax2), (origin_df, pair_df), (f"Matches of {mr_set.field}", "Difference")):
        ax.plot(df.index.values, df[Relation._s], "-C0", label="source")
        ax.plot(df.index.values, df[Relation._f] + (offset if ax is ax1 else 0), "-C1", label="follow-up")
        ax.plot(df.index.values, df.apply(mr_set.relation._extent_func, axis=1, result_type="reduce"),
                "o:C2", label="difference")
        ax.set_title(title, fontsize=title_size)
        ax.tick_params(labelsize=tick_size)
        ax.set_xlabel("Tick" if ax == ax1 else "Match", fontsize=text_size)
        ax.set_ylabel(mr_set.field.capitalize(), fontsize=text_size)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.legend(fontsize=text_size)
    critical_intervals = Relation.critical_intervals(pair_df, labels)
    for i, (x, y) in enumerate(matches):
        color = "gray" if i not in critical_intervals else "crimson"
        ax1.plot((x, y), (source.loc[x, mr_set.field], follow_up.loc[y, mr_set.field] + offset),
                 "--", color=color, zorder=1)

    if len(critical_intervals) > 0:
        for points in np.split(critical_intervals, np.where(np.diff(critical_intervals) != 1)[0] + 1):
            ax2.axvspan(points[0] - 0.5, points[-1] + 0.5, color="red", alpha=0.1)

    if verbose and CONFIG["violation"]["dtw"]:
        ax3 = fig.add_subplot(gs[4:, :3])
        ax3.plot(*list(zip(*matches)), "-C3", label="DTW path")
        ax3.set_title("DTW path", fontsize=title_size)
        ax3.tick_params(labelsize=tick_size)
        ax3.set_xlabel("Tick", fontsize=text_size)
        ax3.set_ylabel("Tick", fontsize=text_size)
        ax3.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax3.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax3.legend(fontsize=text_size)

        if CONFIG["violation"]["strategy"] == "position":
            ax4 = fig.add_subplot(gs[4:, 3:])
            for df, color, shift in zip((source, follow_up), ("C0", "C1"), (0, offset)):
                x, y = (df["position_x"] - shift).to_numpy(), (df["position_y"] - shift).to_numpy()
                ax4.quiver(x[:-1], y[:-1], np.diff(x), np.diff(y), angles="xy", scale_units="xy", scale=1,
                           units="dots", width=3, color=color)
            ax4.set_title("Matches of positions", fontsize=title_size)
            ax4.tick_params(labelsize=tick_size)
            ax4.set_xlabel("x", fontsize=text_size)
            ax4.set_ylabel("y", fontsize=text_size)
            ax4.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax4.yaxis.set_major_locator(MaxNLocator(integer=True))

            for i, (x, y) in enumerate(matches):
                color = "gray" if i not in critical_intervals else "crimson"
                ax4.plot((source.loc[x, "position_x"], follow_up.loc[y, "position_x"] - offset),
                         (source.loc[x, "position_y"], follow_up.loc[y, "position_y"] - offset),
                         "--", color=color, zorder=1)

    fig.tight_layout()
    fig.savefig(os.path.join(CONFIG["workspace"]["visualization"], "violation.png"))
    if show: plt.show()


def visualize_diversity(files: Dict[str, List[str]], show=False):
    """Plot the solution diversity among different algorithms.

    @param files: Solution data files of different algorithms.
    @param show: A boolean to determine whether to show the plots or not.
    """
    data = (defaultdict(list), defaultdict(list), defaultdict(list))
    for name, file_list in files.items():
        for file in file_list:
            with open(file, "rb") as f:
                solutions = pickle.load(f)
            if len(solutions) < 2:
                logger.warning(f"No solutions or only one solution found in {file}.")
                continue

            dist = pairwise_distance(solutions)
            # Average pairwise distance.
            average_pairwise_dist = np.sum(dist) / len(dist)
            data[0][name].append(average_pairwise_dist)
            # Pure diversity.
            dist_matrix = squareform(dist)
            from impl.algorithm.base import BaseAlgorithm
            pd_dist = BaseAlgorithm.population_diversity(dist_matrix)
            data[1][name].append(pd_dist)
            # Average nearest neighbor distance.
            np.fill_diagonal(dist_matrix, np.inf)
            data[2][name].append(np.mean(np.min(dist_matrix, axis=0)))

    height = 4
    title_size, text_size, tick_size = height * 4, height * 4, height * 3
    fig, axs = plt.subplots(1, 3, figsize=(3 * height, height))
    for ax, diversities, title in zip(axs, data, ("Average pairwise distance",
                                                  "Pure diversity",
                                                  "Average nearest\nneighbor distance")):
        bplot = ax.boxplot(diversities.values(), labels=[verbose_map[l] for l in diversities.keys()],
                           patch_artist=True, whis=(0, 100))
        for patch, name in zip(bplot["boxes"], diversities.keys()):
            patch.set_facecolor(style_map[name]["color"])
            patch.set_alpha(0.6)
        ax.set_title(title, fontsize=title_size)
        ax.tick_params(labelsize=tick_size)
        ax.set_ylabel("Distance", fontsize=text_size)
        ax.grid()

    fig.tight_layout()
    fig.savefig(os.path.join(CONFIG["workspace"]["visualization"], f"diversity.png"))
    if show: plt.show()


def visualize_diversity_distribution(file, show=False):
    with open(file, "rb") as f:
        solutions = pickle.load(f)
    if len(solutions) < 2:
        logger.warning("No solutions or only one solution found.")
        return

    dist_matrix = squareform(pairwise_distance(solutions))
    out = MDS(n_components=3, dissimilarity="precomputed").fit(dist_matrix).embedding_
    fig, ax = plt.subplots(figsize=(10, 8))
    ax = plt.axes(projection="3d")
    ax.scatter3D(out[:, 0], out[:, 1], out[:, 2])
    ax.set_box_aspect((np.ptp(out[:, 0]), np.ptp(out[:, 1]), np.ptp(out[:, 2])))
    fig.tight_layout()
    fig.savefig(os.path.join(CONFIG["workspace"]["visualization"], f"{Path(file).stem}-diversity.png"))
    if show: plt.show()


def visualize_archived_distinct_solutions(files: Dict[str, List[str]], fitness_thresholds: List[float],
                                          distance_thresholds: List[float], mr_set, box=False, show=False):
    """Plot the number of distinct solutions from final archived solutions by applying fitness and distance thresholds.

    @param files: Solution data files of different algorithms. Dict[name_of_algorithm, List[solution_file]].
    @param fitness_thresholds: A list of fitness thresholds.
    @param distance_thresholds: A list of distance thresholds.
    @param mr_set: The given MRs.
    @param box: Show box plots.
    @param show: A boolean to determine whether to show the plots or not.
    """
    col_num, height = 3, 4
    title_size, text_size, tick_size = height * 4, height * 4, height * 3
    if len(fitness_thresholds) < col_num: col_num = len(fitness_thresholds)
    row_num = int(np.ceil(len(fitness_thresholds) / col_num))
    fig, axes = plt.subplots(row_num, col_num, figsize=(col_num * height * 1.2, row_num * height))
    ax_map = {gp_name: ax for ax, gp_name in zip(axes.reshape(-1), fitness_thresholds)}

    # vda = defaultdict(lambda: dict())
    data = {}
    for name, file_list in files.items():
        df = pd.DataFrame()
        for file in file_list:
            with open(file, "rb") as f:
                solutions = pickle.load(f)
            solution_df = _filter_by_thresholds(solutions, fitness_thresholds, distance_thresholds, mr_set,
                                                additional_metrics=['distinct_solution_num'])
            df = (pd.concat([df, solution_df], ignore_index=True))

        data[name] = df
        groups = df.groupby("fitness_threshold")
        for gp_name, group in groups:
            group = group.groupby("distance_threshold").agg(list)
            values = group["distinct_solution_num"]
            ax = ax_map[gp_name]
            # vda[gp_name][name] = list(group["distinct_solution_num"])
            mean_val = values.apply(np.mean)
            gp_means = pd.DataFrame({"distance_threshold": distance_thresholds, "mean_ds": mean_val})
            gp_means["fitness_threshold"] = gp_name
            gp_means["alg"] = name
            mean_df = pd.concat([mean_df, gp_means], ignore_index=True)
            ax.errorbar(distance_thresholds, mean_val,
                        yerr=values.apply(lambda row: 0.95 * np.std(row) / np.sqrt(len(row))),
                        **style_map[name], capsize=2, label=verbose_map[name])
            if box: ax.boxplot(group["distinct_solution_num"], positions=group.index.values, widths=0.05,
                               patch_artist=True, manage_ticks=False, whis=(0, 100),
                               boxprops=dict(facecolor=style_map[name]["color"], alpha=0.4))
            ax.set_title(f"Fitness threshold ($\\theta_f={gp_name}$)", fontsize=title_size)
            ax.tick_params(labelsize=tick_size)
            ax.set_xlabel("Distance threshold ($\\theta_d$)", fontsize=text_size)
            ax.set_ylabel("Average $DS$", fontsize=text_size)
            ax.xaxis.set_major_locator(MultipleLocator(0.2))
            ax.yaxis.set_major_locator(MaxNLocator(integer=True, min_n_ticks=1))
            ax.legend()
            ax.grid()
    calculate_ds_improvements(mean_df)
    # for fitness, d in vda.items():
    #     for alg1, alg2 in (("ccea", "ga"), ("ccea", "rs"), ("ga", "rs")):
    #         treatment, control = d[alg1], d[alg2]
    #         for i, (num1, num2) in enumerate(zip(treatment, control)):
    #             estimate, magnitude = Visualizer._vda(num1, num2)
    #             print(f"{fitness}-{distance_thresholds[i]}: {alg1}-{alg2}: {magnitude}-{estimate}.")

    fig.tight_layout()
    fig.savefig(os.path.join(CONFIG["workspace"]["visualization"], f"archived_distinct_solutions.png"))
    if show: plt.show()
    return data


def visualize_distinct_solution_over_simulations(directory: str, files: Dict[str, List[List[str]]],
                                                 fitness_thresholds: List[float], distance_thresholds: List[float],
                                                 max_sim_num: int, interval=10, mrc=False, mr_set=None, show=False):
    """Plot the number of distinct solutions over simulations by applying fitness and distance thresholds.

    @param directory: The directory of checkpoint files.
    @param files: Checkpoint files. Dict[name_of_algorithm, List[Tuple(start_checkpoint, end_checkpoint)]].
    @param fitness_thresholds: A list of fitness thresholds.
    @param distance_thresholds: A list of distance thresholds.
    @param max_sim_num: The maximum number of simulations.
    @param interval: Width of intervals for aggregation (percentage).
    @param mrc: Show the MR coverage.
    @param mr_set: The given MRs.
    @param show: A boolean to determine whether to show the plots or not.
    """
    percent_ranges = np.arange(interval, 101, interval)
    ranges = (percent_ranges / 100 * max_sim_num).round(0).astype(int)
    col_num, height = 3, 4
    title_size, text_size, tick_size = height * 4, height * 4, height * 3
    total_size = len(fitness_thresholds) * len(distance_thresholds)
    if total_size < col_num: col_num = total_size
    row_num = int(np.ceil(total_size / col_num))
    fig, axes = plt.subplots(row_num, col_num, figsize=(col_num * height * 1.2, row_num * height))
    ax_map = {gp_name: ax for ax, gp_name in zip(axes.reshape(-1), product(fitness_thresholds, distance_thresholds))}
    auc_df = pd.DataFrame()

    checkpoint_files = sorted(os.listdir(directory))
    for name, ckp_list in files.items():
        helper = pd.DataFrame({"simulation_num": ranges})
        agg_df_list = defaultdict(list)
        for start, end in ckp_list:
            file_list = [os.path.join(directory, f) for f in checkpoint_files if start <= f <= end]
            df = pd.DataFrame()
            for file in file_list:
                with open(file, "rb") as f:
                    for _ in range(skip_map[name]): pickle.load(f)
                    solutions = pickle.load(f)
                    for _ in range(2): pickle.load(f)
                    budget = pickle.load(f)
                ckp_df = _filter_by_thresholds(solutions, fitness_thresholds, distance_thresholds, mr_set,
                                               additional_metrics=['distinct_solution_num'])
                ckp_df["simulation_num"] = budget.sim_num
                df = (pd.concat([df, ckp_df], ignore_index=True))

            groups = df.groupby(["fitness_threshold", "distance_threshold"])
            for gp_name, group in groups:
                group = group.sort_values("simulation_num", ascending=True)
                agg_df = (pd.merge(group, helper, on="simulation_num", how="outer")
                          .sort_values("simulation_num", ascending=True))
                agg_df.set_index("simulation_num", inplace=True)
                agg_df["distinct_solution_num"].interpolate("index", inplace=True)
                agg_df["violated_mr_num"].interpolate("index", inplace=True)
                temp = agg_df.loc[ranges]
                agg_df_list[gp_name].append(temp)

        for gp_name, agg_dfs in agg_df_list.items():
            agg_df = pd.concat(agg_dfs).groupby("simulation_num").agg(list)
            ax = ax_map[gp_name]
            y = agg_df["violated_mr_num" if mrc else "distinct_solution_num"].apply(np.mean) * 100 / len(
                mr_set.mrs)
            auc_df = pd.concat([auc_df, pd.DataFrame([{
                'alg': name,
                'fitness_threshold': gp_name[0],
                'distance_threshold': gp_name[1],
                'auc': area_under_curve(np.array([0, *percent_ranges]), np.array([0, *y]))
            }])], ignore_index=True)
            ax.plot(percent_ranges,
                    y,
                    **style_map[name], label=verbose_map[name])
            ax.set_title(
                f"Fitness threshold ($\\theta_f={gp_name[0]}$),\nDistance threshold ($\\theta_d={gp_name[1]}$)",
                fontsize=title_size)
            ax.tick_params(labelsize=tick_size)
            ax.xaxis.set_major_locator(MultipleLocator(interval))
            ax.yaxis.set_major_locator(MaxNLocator(integer=True, min_n_ticks=1))
            ax.set_xlabel("Simulation budget (%)", fontsize=text_size)
            ax.set_ylabel("Average $MRC$ (%)" if mrc else "Average $DS$", fontsize=text_size)
            ax.legend()
            ax.grid()

    calculate_auc_improvements(auc_df)

    fig.tight_layout()
    fig.savefig(os.path.join(CONFIG["workspace"]["visualization"], f"distinct_solutions_over_simulations.png"))
    if show: plt.show()


def visualize_archived_solutions_by_gen(directory: str, checkpoints: Dict[str, List[List[str]]], generation_num,
                                        metric_name,
                                        fitness_thresholds: List[float],
                                        distance_thresholds: List[float], mr_set, box=False, show=False,
                                        legend_loc='upper right', padding={'top': 1.1, 'bottom': 0.3}):
    """Plot the number of distinct solutions from final archived solutions by applying fitness and distance thresholds.

    @param directory: The directory of checkpoint files.
    @param checkpoints: Checkpoint files Dict[name_of_algorithm, List[list[checkpoint_file]].
    @param generation_num: Generation Number. If set to K, plot the solutions found after K generations.
    @param metric_name: The metric used for comparison. Options are
        "ds" (Distinct Solutions)
        "avg_pw" (Average Pairwise Distance)
        "pure_div" (Pure Diversity)
        "avg_fitness" (Average Fitness)
    @param fitness_thresholds: A list of fitness thresholds.
    @param distance_thresholds: A list of distance thresholds.
    @param mr_set: The given MRs.
    @param box: Show box plots.
    @param show: A boolean to determine whether to show the plots or not.
    @param legend_loc: Specifies the location of the legend in the plots
    @oaram padding: Paddings between the extreme chart points and the axis range.
    """
    col_num, height = 3, 4
    title_size, text_size, tick_size = height * 4, height * 4, height * 3
    if len(distance_thresholds) < col_num: col_num = len(distance_thresholds)
    row_num = int(np.ceil(len(distance_thresholds) / col_num))
    fig, axes = plt.subplots(row_num, col_num, figsize=(col_num * height * 1.2, row_num * height))
    ax_map = {gp_name: ax for ax, gp_name in zip(axes.reshape(-1), distance_thresholds)}
    checkpoint_files = sorted(os.listdir(directory))
    data = {}
    for alg, runs in checkpoints.items():
        df = pd.DataFrame()
        for i, (start, end) in enumerate(runs):
            run = [os.path.join(directory, f) for f in checkpoint_files if start <= f <= end]
            if len(run) < generation_num:
                cp = run[-1]
            else:
                cp = run[generation_num - 1]

            with open(cp, "rb") as f:
                for _ in range(skip_map[alg]): pickle.load(f)
                solutions = pickle.load(f)

            solution_df = _filter_by_thresholds(solutions, fitness_thresholds, distance_thresholds, mr_set,
                                                additional_metrics=([metric_name]))

            df = (pd.concat([df, solution_df], ignore_index=True))

        data[alg] = df
        groups = df.groupby("distance_threshold")
        y_max, y_min = None, None
        for gp_name, group in groups:
            group = group.groupby("fitness_threshold").agg(list)
            values = group[metric_name]

            chart_y_max = max(
                [np.nanmean(row) + 0.95 * np.nanstd(row) / np.sqrt(np.count_nonzero(~np.isnan(row))) for row in values])
            chart_y_min = min(
                [np.nanmean(row) - 0.95 * np.nanstd(row) / np.sqrt(np.count_nonzero(~np.isnan(row))) for row in values])

            if not y_max or y_max < chart_y_max: y_max = chart_y_max
            if not y_min or y_min > chart_y_min: y_min = chart_y_min

            ax = ax_map[gp_name]
            ax.errorbar(fitness_thresholds, values.apply(np.nanmean),
                        yerr=values.apply(
                            lambda row: 0.95 * np.nanstd(row) / np.sqrt(np.count_nonzero(~np.isnan(row)))),
                        **style_map[alg], capsize=2, label=verbose_map[alg])

            if box: ax.boxplot(group["distinct_solution_num"], positions=group.index.values, widths=0.05,
                               patch_artist=True, manage_ticks=False, whis=(0, 100),
                               boxprops=dict(facecolor=style_map[alg]["color"], alpha=0.4))
            ax.set_title(f"Distance threshold ($\\theta_d={gp_name}$)", fontsize=title_size)
            ax.tick_params(labelsize=tick_size)
            ax.set_xlabel("Fitness threshold ($\\theta_f$)", fontsize=text_size)
            ax.set_ylabel(f'${verbose_map[metric_name]}$', fontsize=text_size)

            curr_y_min, curr_y_max = ax.get_ylim()

            if y_max + padding['top'] > curr_y_max:
                ax.set_ylim(top=y_max + padding['top'])
            if y_min - padding['bottom'] < curr_y_min:
                ax.set_ylim(bottom=y_min - padding['bottom'])
            ax.xaxis.set_major_locator(MultipleLocator(0.2))
            ax.yaxis.set_major_locator(MaxNLocator(integer=True, min_n_ticks=1))
            ax.legend(loc=legend_loc)
            ax.grid()

    fig.tight_layout()
    fig.savefig(
        os.path.join(CONFIG["workspace"]["visualization"], f"archived_{metric_name}_by_gen{generation_num}.png"))
    if show: plt.show()


def visualize_archive_solution_over_generations(directory: str, files: Dict[str, List[List[str]]], metric_name,
                                                fitness_thresholds: List[float], distance_thresholds: List[float],
                                                mr_set, max_gen: int, show=False, legend_loc='upper right',
                                                padding={'top': 0.6, 'bottom': 0.3}):
    """Plot the metrics over generations by applying fitness and distance thresholds.

    @param directory: The directory of checkpoint files.
    @param checkpoints: Checkpoint files Dict[name_of_algorithm, List[list[checkpoint_file]].
    @param metric_name: The metric used for comparison. Options are
        "ds" (Distinct Solutions)
        "avg_pw" (Average Pairwise Distance)
        "pure_div" (Pure Diversity)
        "avg_fitness" (Average Fitness)
    @param fitness_thresholds: A list of fitness thresholds.
    @param distance_thresholds: A list of distance thresholds.
    @param mr_set: The given MRs.
    @param max_gen: The maximum number of generations.
    @param show: A boolean to determine whether to show the plots or not.
    @oaram legend_loc: Location of the legend in the plots.
    @oaram padding: Paddings between the extreme chart points and the axis range.
    """
    col_num, height = 3, 4
    title_size, text_size, tick_size = height * 4, height * 4, height * 3
    total_size = len(fitness_thresholds) * len(distance_thresholds)
    if total_size < col_num: col_num = total_size
    row_num = int(np.ceil(total_size / col_num))
    fig, axes = plt.subplots(row_num, col_num, figsize=(col_num * height * 1.2, row_num * height))
    ax_map = {gp_name: ax for ax, gp_name in zip(axes.reshape(-1), product(fitness_thresholds, distance_thresholds))}

    checkpoint_files = sorted(os.listdir(directory))
    data = {}
    for alg, ckp_list in files.items():
        df = pd.DataFrame()
        for run_counter, (start, end) in enumerate(ckp_list):
            file_list = [os.path.join(directory, f) for f in checkpoint_files if start <= f <= end]
            for i, file in enumerate(file_list):
                if i == max_gen: break
                with open(file, "rb") as f:
                    for _ in range(skip_map[alg]): pickle.load(f)
                    solutions = pickle.load(f)
                ckp_df = _filter_by_thresholds(solutions, fitness_thresholds, distance_thresholds, mr_set,
                                               additional_metrics=[metric_name])
                ckp_df["alg"] = alg
                ckp_df["run"] = run_counter
                ckp_df["gen"] = i + 1
                df = (pd.concat([df, ckp_df], ignore_index=True))
        data[alg] = df
        groups = df.groupby(["fitness_threshold", "distance_threshold"])
        y_max, y_min = None, None

        for gp_name, group in groups:
            group = group.groupby("gen").agg({metric_name: list}).reset_index().sort_values("gen")
            ax = ax_map[gp_name]

            chart_y_max = np.nanmax(group[metric_name].apply(np.nanmean))
            chart_y_min = np.nanmin(group[metric_name].apply(np.nanmean))

            if not y_max or y_max < chart_y_max: y_max = chart_y_max
            if not y_min or y_min > chart_y_min: y_min = chart_y_min

            group[f'{metric_name}_mean'] = group[metric_name].apply(np.nanmean)
            group = group[group[f'{metric_name}_mean'].notna()].sort_values('gen')

            ax.plot(group["gen"],
                    group[f'{metric_name}_mean'],
                    **style_map[alg], label=verbose_map[alg])
            ax.set_title(
                f"Fitness threshold ($\\theta_f={gp_name[0]}$),\nDistance threshold ($\\theta_d={gp_name[1]}$)",
                fontsize=title_size)
            ax.tick_params(labelsize=tick_size)
            # ax.xaxis.set_major_locator(MultipleLocator(interval))
            ax.yaxis.set_major_locator(MaxNLocator(integer=True, min_n_ticks=1))
            ax.set_xlabel("Generation", fontsize=text_size)
            ax.set_ylabel(f'${verbose_map[metric_name]}$', fontsize=text_size)
            curr_y_min, curr_y_max = ax.get_ylim()
            if y_max + padding['top'] > curr_y_max:
                ax.set_ylim(top=y_max + padding['top'])
            if y_min - padding['bottom'] < curr_y_min:
                ax.set_ylim(bottom=y_min - padding['bottom'])

            ax.legend(loc=legend_loc)
            ax.grid()

    fig.tight_layout()
    fig.savefig(os.path.join(CONFIG["workspace"]["visualization"], f"archived_{metric_name}_over_generetations.png"))
    if show: plt.show()


def visualize_computational_efficiency(log_file: str, solution_files: Dict[str, List[str]],
                                       checkpoints: Dict[str, List[List[str]]], show=False):
    """
    Generate a boxplot comparing computational efficiency (duration in hours) across algorithms from a log file.

    @param log_file: Path to the CSV log file containing 'alg', 'start_time', 'end_time' columns.
    @param solution_files: Solution data files of different algorithms. Dict[name_of_algorithm, List[solution_file]].
    @param checkpoints: Checkpoint files Dict[name_of_algorithm, List[list[checkpoint_file]].
    @param show: A boolean to determine whether to show the plots or not.

    """
    df = _generate_execution_time_data(log_file, solution_files, checkpoints)
    algorithms = list(solution_files.keys())
    durations = [df[df['alg'] == alg]['duration_hours'] for alg in algorithms]
    colors = [style_map[alg]['color'] for alg in algorithms]

    fig = plt.figure(figsize=(8, 8))
    box = plt.boxplot(durations, labels=algorithms, patch_artist=True, medianprops=dict(color='black'),
                      showfliers=False)

    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)

    for i, (duration, color) in enumerate(zip(durations, colors), start=1):
        plt.scatter([i] * len(duration), duration, alpha=0.7, color=color)

    plt.title('Comparison of Computational Efficiency (Duration in Hours)')
    plt.xlabel('Algorithm')
    plt.ylabel('Duration (Hours)')
    plt.grid(axis='y', linestyle='--')
    plt.tight_layout()

    fig.tight_layout()
    fig.savefig(os.path.join(CONFIG["workspace"]["visualization"], f"computational_efficiency.png"))
    if show: plt.show()


def _filter_by_thresholds(solutions, fitness_thresholds: List[float], distance_thresholds: List[float], mr_set,
                          additional_metrics: List[str] = []):
    default_columns = ["fitness_threshold", "distance_threshold", "violated_mr_num", "distinct_mr_num", "avg_pw"]
    additional_metrics = [metric for metric in additional_metrics if metric not in default_columns]
    column_names = (*default_columns, *additional_metrics)
    if len(solutions) == 0:
        return pd.DataFrame([[fitness_threshold, distance_threshold, 0, 0, np.nan,
                              *[metrics[metric](solutions) for metric in additional_metrics]]
                             for fitness_threshold in fitness_thresholds
                             for distance_threshold in distance_thresholds], columns=column_names)

    indices_to_remove = [[i for i, solution in enumerate(solutions) if solution.fitness.values[0] < threshold]
                         for threshold in fitness_thresholds]
    indices_of_violated_mrs = np.array(mr_set.violated_mrs([perturbations for _, perturbations in solutions]))
    dist_matrix = squareform(pairwise_distance(solutions)) if len(solutions) > 1 else np.array([[0]])

    results = []
    for idx, indices in enumerate(indices_to_remove):
        dist = np.delete(dist_matrix, indices, axis=0)
        dist = np.delete(dist, indices, axis=1)
        violated_mrs = np.delete(indices_of_violated_mrs, indices, axis=0)
        selected_solutions = [sol for i, sol in enumerate(solutions) if i not in indices]
        assert len(violated_mrs) == len(dist)
        assert len(selected_solutions) == len(dist)
        n = len(dist)

        if n < 2:
            results += [[fitness_thresholds[idx], threshold, n, n, np.nan,
                         *[metrics[metric](selected_solutions) for metric in additional_metrics]] for threshold in
                        distance_thresholds]
            continue

        for threshold in distance_thresholds:
            graph = nx.Graph()
            graph.add_nodes_from(range(n))
            for i in range(n):  # Add edges between points that are closer than the threshold distance.
                for j in range(i + 1, n):
                    if dist[i, j] < threshold:
                        graph.add_edge(i, j)

            # Find a maximal independent set as an approximation to maximum independent set.
            independent_set = nx.algorithms.approximation.maximum_independent_set(graph)
            # The points to keep are in the independent set.
            points_to_keep = set(independent_set)
            # The points to remove are the complement of the independent set.
            # points_to_remove = set(range(n)) - points_to_keep
            mr_indices = list(map(tuple, violated_mrs[list(points_to_keep)]))
            violated_mr_num = len(set(np.hstack(mr_indices)))
            distinct_mr_num = len(set(mr_indices))
            avg_pw_value = avg_pw_from_matrix(dist, list(points_to_keep))
            final_solutions = [sol for i, sol in enumerate(selected_solutions) if i in points_to_keep]
            results.append([fitness_thresholds[idx], threshold, violated_mr_num, distinct_mr_num, avg_pw_value,
                            *[metrics[metric](final_solutions) for metric in additional_metrics]])
    return pd.DataFrame(results, columns=column_names)


def _vda(treatment: List[int], control: List[int]):
    m = len(treatment)
    n = len(control)
    if m != n: raise ValueError("Data d and f must have the same length.")

    r = rankdata(treatment + control)
    r1 = sum(r[0:m])

    # Compute the measure.
    # A = (r1/m - (m+1)/2)/n # Formula (14) in Vargha and Delaney, 2000.
    a = (2 * r1 - m * (m + 1)) / (2 * n * m)  # Equivalent formula to avoid accuracy errors.

    levels = [0.147, 0.33, 0.474]  # Effect sizes from Hess and Kromrey, 2004.
    magnitude = ["negligible", "small", "medium", "large"]
    scaled_a = (a - 0.5) * 2
    return a, magnitude[bisect_left(levels, abs(scaled_a))]


def _generate_execution_time_data(log_path: str, solution_files: Dict[str, List[str]],
                                  checkpoints: Dict[str, List[List[str]]]):
    log_df = _make_log_df(log_path)

    sol_cp_map = pd.DataFrame()
    for alg, cp_ranges in checkpoints.items():
        sol_files = sorted(solution_files[alg], key=lambda f: int(f))
        for i, cp_range in enumerate(cp_ranges):
            start_cp, end_cp = cp_range[0], cp_range[1]
            sol_file = sol_files[i]
            start_cp_time = pd.to_datetime(int(start_cp.split(".")[0]), unit='ms')
            end_cp_time = pd.to_datetime(int(end_cp.split(".")[0]), unit='ms')
            end_time = pd.to_datetime(int(sol_file.split(".")[0].split('-')[-1]), unit='ms')
            algo_logs = log_df[log_df["algorithm"] == alg]
            closest_log = \
                algo_logs[algo_logs["timestamp"] <= start_cp_time].sort_values(by="timestamp", ascending=False).iloc[0]
            sol_cp_map = pd.concat([sol_cp_map, pd.DataFrame([{
                'alg': alg,
                'start_time': closest_log["timestamp"],
                'end_time': end_time,
                'solution': sol_file,
                'start_cp': start_cp,
                'start_cp_time': start_cp_time,
                'end_cp': end_cp,
                'end_cp_time': end_cp_time,
                'process_id': closest_log["process_id"],

            }])], ignore_index=True)

    sol_cp_map['duration_hours'] = (sol_cp_map['end_time'] - sol_cp_map['start_time']).dt.total_seconds() / 3600
    return sol_cp_map


def _make_log_df(log_path: str):
    with open(log_path, "r") as file:
        logs = file.readlines()

    # Convert logs to DataFrame
    log_entries = []
    for log in logs:
        parts = log.strip().split(" INFO ")
        if len(parts) < 2:
            continue

        timestamp = parts[0]
        info = parts[1].split(" --- ")
        if len(info) < 2:
            continue

        process_id = info[0]
        description = info[1]
        algo = None

        if "CCEA started" in description:
            algo = "ccea"
        elif "Genetic" in description:
            algo = "ga"
        elif "Random search" in description:
            algo = "rs"

        if algo:
            log_entries.append([pd.to_datetime(timestamp), algo, process_id])

    log_df = pd.DataFrame(log_entries, columns=["timestamp", "algorithm", "process_id"])
    log_df.sort_values(by="timestamp", inplace=True)
    return log_df
