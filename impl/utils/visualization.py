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
from scipy.spatial.distance import squareform, pdist
from scipy.stats import rankdata
from sklearn.manifold import MDS

from impl.config import CONFIG
from impl.mr.mr import Relation

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
    "rs": "RS",
    "ga": "SGA",
    "gawa": "SGA with Archives",
}

style_map = {
    "ccea": {"color": "C1", "marker": "o"},
    "ga": {"color": "C2", "marker": "*"},
    "rs": {"color": "C0", "marker": "x"}
}


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

            dist = _pairwise_distance(solutions)
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

    dist_matrix = squareform(_pairwise_distance(solutions))
    out = MDS(n_components=3, dissimilarity="precomputed").fit(dist_matrix).embedding_
    fig, ax = plt.subplots(figsize=(10, 8))
    ax = plt.axes(projection="3d")
    ax.scatter3D(out[:, 0], out[:, 1], out[:, 2])
    ax.set_box_aspect((np.ptp(out[:, 0]), np.ptp(out[:, 1]), np.ptp(out[:, 2])))
    fig.tight_layout()
    fig.savefig(os.path.join(CONFIG["workspace"]["visualization"], f"{Path(file).stem}-diversity.png"))
    if show: plt.show()


def visualize_archived_distinct_solutions(files: Dict[str, List[str]], fitness_thresholds: List[float],
                                          distance_thresholds: List[float], box=False, show=False):
    """Plot the number of distinct solutions from final archived solutions by applying fitness and distance thresholds.

    @param files: Solution data files of different algorithms. Dict[name_of_algorithm, List[solution_file]].
    @param fitness_thresholds: A list of fitness thresholds.
    @param distance_thresholds: A list of distance thresholds.
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
            solution_df = _filter_by_thresholds(solutions, fitness_thresholds, distance_thresholds)
            df = (pd.concat([df, solution_df], ignore_index=True))

        data[name] = df
        groups = df.groupby("fitness_threshold")
        for gp_name, group in groups:
            group = group.groupby("distance_threshold").agg(list)
            values = group["distinct_solution_num"]
            ax = ax_map[gp_name]
            # vda[gp_name][name] = list(group["distinct_solution_num"])
            ax.errorbar(distance_thresholds, values.apply(np.mean),
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
                                                 max_sim_num: int, interval=10, mrc=False, show=False):
    """Plot the number of distinct solutions over simulations by applying fitness and distance thresholds.

    @param directory: The directory of checkpoint files.
    @param files: Checkpoint files. Dict[name_of_algorithm, List[Tuple(start_checkpoint, end_checkpoint)]].
    @param fitness_thresholds: A list of fitness thresholds.
    @param distance_thresholds: A list of distance thresholds.
    @param max_sim_num: The maximum number of simulations.
    @param interval: Width of intervals for aggregation (percentage).
    @param mrc: Show the MR coverage.
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

    checkpoint_files = sorted(os.listdir(directory))
    skip = {"ccea": 5, "ga": 2, "rs": 1}
    for name, ckp_list in files.items():
        helper = pd.DataFrame({"simulation_num": ranges})
        agg_df_list = defaultdict(list)
        for start, end in ckp_list:
            file_list = [os.path.join(directory, f) for f in checkpoint_files if start <= f <= end]
            df = pd.DataFrame()
            for file in file_list:
                with open(file, "rb") as f:
                    for _ in range(skip[name]): pickle.load(f)
                    solutions = pickle.load(f)
                    for _ in range(2): pickle.load(f)
                    budget = pickle.load(f)
                ckp_df = _filter_by_thresholds(solutions, fitness_thresholds, distance_thresholds)
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

        from impl.problem import mr_set
        for gp_name, agg_dfs in agg_df_list.items():
            agg_df = pd.concat(agg_dfs).groupby("simulation_num").agg(list)
            ax = ax_map[gp_name]
            ax.plot(percent_ranges,
                    agg_df["violated_mr_num" if mrc else "distinct_solution_num"].apply(np.mean) * 100 / len(mr_set.mrs),
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

    fig.tight_layout()
    fig.savefig(os.path.join(CONFIG["workspace"]["visualization"], f"distinct_solutions_over_simulations.png"))
    if show: plt.show()


def _filter_by_thresholds(solutions, fitness_thresholds: List[float], distance_thresholds: List[float]):
    column_names = ("fitness_threshold", "distance_threshold",
                    "distinct_solution_num", "violated_mr_num", "distinct_mr_num")
    if len(solutions) == 0:
        return pd.DataFrame([[fitness_threshold, distance_threshold, 0, 0, 0]
                             for fitness_threshold in fitness_thresholds
                             for distance_threshold in distance_thresholds], columns=column_names)

    indices_to_remove = [[i for i, solution in enumerate(solutions) if solution.fitness.values[0] < threshold]
                         for threshold in fitness_thresholds]
    from impl.problem import mr_set
    indices_of_violated_mrs = np.array(mr_set.violated_mrs([perturbations for _, perturbations in solutions]))
    dist_matrix = squareform(_pairwise_distance(solutions))

    results = []
    for idx, indices in enumerate(indices_to_remove):
        dist = np.delete(dist_matrix, indices, axis=0)
        dist = np.delete(dist, indices, axis=1)
        violated_mrs = np.delete(indices_of_violated_mrs, indices, axis=0)
        assert len(violated_mrs) == len(dist)
        n = len(dist)
        if n < 2:
            results += [[fitness_thresholds[idx], threshold, n, n, n] for threshold in distance_thresholds]
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
            results.append([fitness_thresholds[idx], threshold, len(points_to_keep),
                            violated_mr_num, distinct_mr_num])
    return pd.DataFrame(results, columns=column_names)


def _pairwise_distance(solutions):
    follow_ups = []
    for scenario, perturbation in solutions:
        perturbation.perturb(scenario)
        follow_ups.append(scenario)
    return pdist(np.array(follow_ups, dtype=object).reshape((len(follow_ups), -1)), lambda x, y: x[0].dist(y[0]))


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
