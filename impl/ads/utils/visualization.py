import logging
import os
import pickle
import time
from bisect import bisect_left
from collections import defaultdict
from itertools import product
from typing import Dict, List

import carla
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

from impl import config as cfg
from impl.ads.mr.mr import Relation
from impl.ads.utils.math_utils import calculate_auc_improvements, area_under_curve, calculate_ds_improvements
from impl.ads.utils.math_utils import polar_to_cartesian
from impl.ads.utils.metrics import metrics, pairwise_distance, avg_pw_from_matrix
from impl.config import init_project_directory

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
    "ccea-c": "CoCoMEGA\c",
    "ccea-d": "CoCoMEGA\d",
    "ccea+ri": "CoCoMEGA with RI",
    "ccea-c+ri": "CoCoMEGA\c with RI",
    "rs": "RS",
    "rs+ri": "RS with RI",
    "ga": "SGA",
    "ga+ri": "SGA with RI",
    "gawa": "SGA with Archives",
    "distinct_solution_num": "Average $DS$",
    "ds": "$DS$",
    "avg_fit": "Average Fitness",
    "avg_pw": "$APD$",
    "pure_div": "Pure Diversity",
}

style_map = {
    "ccea": {"color": "C1", "marker": "o"},
    "ccea-c": {"color": "C3", "marker": "P"},
    "ccea-d": {"color": "C3", "marker": "P"},
    "ccea+ri": {"color": "C4", "marker": "P"},
    "ccea-c+ri": {"color": "C5", "marker": "P"},
    "ga": {"color": "C2", "marker": "*"},
    "ga+ri": {"color": "C6", "marker": "*"},
    "rs": {"color": "C0", "marker": "x"},
    "rs+ri": {"color": "C7", "marker": "x"},
}
default_style = {"color": "C8", "marker": "o"}

skip_map = {
    "ccea": 5,
    "ga": 2,
    "rs": 1,
}


def get_skip(name):
    for alg in skip_map:
        if alg in name:
            return skip_map[alg]
    return None


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
            Expected keys are :data:`gen` for generation numbers,
            :data:`avg` for average fitness, :data:`max` for maximum fitness, etc.
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
        """
        Plot a histogram of the provided statistics.

        :param stats: Data to be plotted in the histogram.
        :param color: Whether the color is for a specific group or not.
        :param show: Whether to display the plot.
        :param out_dir: The directory to save the plot.
        :param title: The title of the histogram.
        :param name: The name for the histogram.
        :param verbose_name: More descriptive name for the histogram.
        """
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
            path = os.path.join(cfg.CONFIG["workspace"]["solution"], row['statistics_path'])
            cls.visualize_algorithm_Stats(path, name=row['name'], out_dir=out_dir)

    @classmethod
    def visualize_algorithm_Stats(cls, path, name, out_dir):
        """
        Visualize algorithm statistics from a logbook file.

        :param path: The file path of the logbook containing the algorithm statistics.
        :param name: The name of the algorithm being visualized.
        :param out_dir: The directory to save the plot.
        """
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
        Generate line plots for evolutionary algorithm statistics across generations.

        :param stats: Data containing statistics over generations.
        :param out_dir: The directory to save the plots.
        :return: List of figures created during the plotting.
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
                title=f'Evolutionary Algorithm Generation Statistics for {verbose_map.get(pop_name, pop_name)} population',
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
        Display the plot.

        :param fig: The plot figure to display.
        """
        fig.show()

    @staticmethod
    def _save_fig(fig, out_dir, name):
        """
        Save the figure as an image file.

        :param fig: The figure to save.
        :param out_dir: The directory where the image should be saved.
        :param name: The name of the saved file.
        """
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        fig.write_image(os.path.join(out_dir, name))


def visualize_in_one(project, plot_nan=True, verbose=False, save_path=None, show=False):
    """Plot all statistics data in one figure.

    :param project: Project name.
    :param plot_nan: Plot :data:`NaN` values.
    :param verbose: Show plots of populations and archives.
    :param save_path: The path to save the figure.
    :param show: A boolean to determine whether to show the plots or not.
    """
    init_project_directory(project, resume=True)
    stat_file = next(cfg.CONFIG["workspace"]["solution"].rglob("statistics*"), None)
    if stat_file is None:
        logger.warning(f"No statistics file found for {project}.")
        return

    data = pickle.loads(stat_file.read_bytes())
    stats = pd.DataFrame(data)
    if len(stats) == 0:
        logger.warning("No statistics provided. Nothing to visualize.")
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
        _ax.set_title(verbose_map.get(_name, _name), fontsize=title_size)
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
    fig.savefig(save_path if save_path else (cfg.CONFIG["workspace"]["visual"] / f"{stat_file.stem}.png"))
    if show: plt.show()


def visualize_comparison(projects: Dict[str, List[str]], max_percentile=0.75,
                         box=True, interval=15, avg_line=False,
                         trend_line=True, regression_degree=3, all_lines=False,
                         plot_nan=True, verbose=False, save_path=None, show=False):
    """Plot comparisons among different algorithms.

    :param projects: Project names of different algorithms.
    :param max_percentile: Plot the given percentile of max fitness.
    :param box: Show box plots.
    :param interval: Width of intervals for aggregation.
    :param avg_line: Show average lines.
    :param trend_line: Show trend lines.
    :param regression_degree: Degree of regression.
    :param all_lines: Show original lines.
    :param plot_nan: Plot NaN values.
    :param verbose: Show more plots.
    :param save_path: The path to save the figure.
    :param show: A boolean to determine whether to show the plots or not.
    """
    data = {}
    for name, project_list in projects.items():
        full, merged, agg = defaultdict(list), {}, {}
        low, high = float("inf"), float("-inf")
        for project in project_list:
            stat_file = next((cfg.CONFIG["workspace"]["result"] / project / "solutions").rglob("statistics*"), None)
            if stat_file is None:
                logger.warning(f"No statistics file found for {project}.")
                continue
            df = pd.DataFrame(pickle.loads(stat_file.read_bytes()))
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
                           whis=(0, 100),
                           boxprops=dict(facecolor=style_map.get(name, default_style)["color"], alpha=0.4))
            if avg_line: ax.plot(agg.index.values, agg[metric].apply(np.nanmean), **style_map.get(name, default_style))
            if trend_line:
                f = np.poly1d(np.polyfit(merged["sim"], merged[metric], regression_degree))
                ax.plot(merged["sim"], f(merged["sim"]), lw=2, color=style_map.get(name, default_style)["color"])
            if all_lines:
                for line in full:
                    ax.plot(line["sim"], line[metric], "--", lw=1, **style_map.get(name, default_style), alpha=0.6)
            if name not in legend_elements:
                legend_elements[name] = Line2D([0], [0], **style_map.get(name, default_style),
                                               label=verbose_map.get(name, name))
        ax.set_title(title, fontsize=title_size)
        ax.tick_params(labelsize=tick_size)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_ylabel("#violations" if ax == ax1 else "Fitness", fontsize=text_size)
        ax.grid()

    ax1.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.legend(handles=legend_elements.values(), fontsize=text_size)
    fig.supxlabel("#simulations", fontsize=text_size)
    fig.tight_layout()
    fig.savefig(save_path if save_path else (cfg.CONFIG["workspace"]["visualization"] / "comparison.png"))
    if show: plt.show()


def visualize_violation(project, source, follow_up, mr_set, offset=3, verbose=False, save_path=None, show=False):
    """Plot the extent of violation between the source results
    and follow-up results based on the given metamorphic relations.

    :param project: Project name.
    :param source: The file name of source results.
    :param follow_up: The file name of follow-up results.
    :param mr_set: The given metamorphic relations.
    :param offset: The offset between the source and follow-up curves.
    :param verbose: Plot the DTW path and matches of positions.
    :param save_path: The path to save the figure.
    :param show: A boolean to determine whether to show the plots or not.
    """
    init_project_directory(project, resume=True)
    source = pd.read_csv((cfg.CONFIG["workspace"]["sim_result"] / source))
    follow_up = pd.read_csv((cfg.CONFIG["workspace"]["sim_result"] / follow_up))
    source.set_index(source.columns[0], inplace=True)
    follow_up.set_index(follow_up.columns[0], inplace=True)

    labels = Relation.convert_labels(mr_set.labels)
    matches, origin_df = Relation.pairwise_dataframe(source, follow_up, mr_set.relation.field, labels)
    if cfg.CONFIG["violation"]["dtw"]:
        matches, pair_df = Relation.dtw_dataframe(source, follow_up, mr_set.relation.field, labels)
    else:
        pair_df = origin_df.reset_index()

    row_num = 7 if verbose and cfg.CONFIG["violation"]["dtw"] else 4
    column_num, height = 6, 3
    title_size, text_size, tick_size = height * 7, height * 5, height * 4
    fig = plt.figure(figsize=(height * column_num, height * row_num))
    gs = GridSpec(row_num, column_num, figure=fig)
    ax1 = fig.add_subplot(gs[0:2, :])
    ax2 = fig.add_subplot(gs[2:4, :], sharey=ax1)

    for ax, df, title in zip((ax1, ax2), (origin_df, pair_df), (f"Matches of {mr_set.relation.field}", "Difference")):
        ax.plot(df.index.values, df[Relation._s], "-C0", label="source")
        ax.plot(df.index.values, df[Relation._f] + (offset if ax is ax1 else 0), "-C1", label="follow-up")
        ax.plot(df.index.values, df.apply(mr_set.relation._extent_func, axis=1, result_type="reduce"),
                "o:C2", label="difference")
        ax.set_title(title, fontsize=title_size)
        ax.tick_params(labelsize=tick_size)
        ax.set_xlabel("Tick" if ax == ax1 else "Match", fontsize=text_size)
        ax.set_ylabel(mr_set.relation.field.capitalize(), fontsize=text_size)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.legend(fontsize=text_size)
    critical_intervals = Relation.critical_intervals(pair_df, labels)
    for i, (x, y) in enumerate(matches):
        color = "gray" if i not in critical_intervals else "crimson"
        ax1.plot((x, y), (source.loc[x, mr_set.relation.field], follow_up.loc[y, mr_set.field] + offset),
                 "--", color=color, zorder=1)

    if len(critical_intervals) > 0:
        for points in np.split(critical_intervals, np.where(np.diff(critical_intervals) != 1)[0] + 1):
            ax2.axvspan(points[0] - 0.5, points[-1] + 0.5, color="red", alpha=0.1)

    if verbose and cfg.CONFIG["violation"]["dtw"]:
        ax3 = fig.add_subplot(gs[4:, :3])
        ax3.plot(*list(zip(*matches)), "-C3", label="DTW path")
        ax3.set_title("DTW path", fontsize=title_size)
        ax3.tick_params(labelsize=tick_size)
        ax3.set_xlabel("Tick", fontsize=text_size)
        ax3.set_ylabel("Tick", fontsize=text_size)
        ax3.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax3.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax3.legend(fontsize=text_size)

        if cfg.CONFIG["violation"]["strategy"] == "position":
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
    fig.savefig(save_path if save_path else (cfg.CONFIG["workspace"]["visual"] / "violation.png"))
    if show: plt.show()


def visualize_diversity(projects: Dict[str, List[str]], save_path=None, show=False):
    """Plot the solution diversity among different algorithms.

    :param projects: Project names of different algorithms.
    :param save_path: The path to save the figure.
    :param show: A boolean to determine whether to show the plots or not.
    """
    data = (defaultdict(list), defaultdict(list), defaultdict(list))
    for name, project_list in projects.items():
        for project in project_list:
            solution_file = next((cfg.CONFIG["workspace"]["result"] / project / "solutions").rglob("solutions*"), None)
            if solution_file is None:
                logger.warning(f"No solution file found for {project}.")
                continue
            solutions = pickle.loads(solution_file.read_bytes())
            if len(solutions) < 2:
                logger.warning(f"No solutions or only one solution found in {project}.")
                continue

            dist = pairwise_distance(solutions)
            # Average pairwise distance.
            average_pairwise_dist = np.sum(dist) / len(dist)
            data[0][name].append(average_pairwise_dist)
            # Pure diversity.
            dist_matrix = squareform(dist)
            from impl.core.algorithm.base import BaseAlgorithm
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
        bplot = ax.boxplot(diversities.values(), labels=[verbose_map.get(l, l) for l in diversities.keys()],
                           patch_artist=True, whis=(0, 100))
        for patch, name in zip(bplot["boxes"], diversities.keys()):
            patch.set_facecolor(style_map.get(name, default_style)["color"])
            patch.set_alpha(0.6)
        ax.set_title(title, fontsize=title_size)
        ax.tick_params(labelsize=tick_size)
        ax.set_ylabel("Distance", fontsize=text_size)
        ax.grid()

    fig.tight_layout()
    fig.savefig(save_path if save_path else (cfg.CONFIG["workspace"]["visualization"] / "diversity.png"))
    if show: plt.show()


def visualize_diversity_distribution(project, save_path=None, show=False):
    """Plot the solution diversity into a 3D space.

    :param project: Project Name.
    :param save_path: The path to save the figure.
    :param show: A boolean to determine whether to show the plots or not.
    """
    init_project_directory(project, resume=True)
    solution_file = next(cfg.CONFIG["workspace"]["solution"].rglob("solutions*"), None)
    if solution_file is None:
        logger.warning(f"No solution file found for {project}.")
        return

    solutions = pickle.loads(solution_file.read_bytes())
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
    fig.savefig(save_path if save_path else (cfg.CONFIG["workspace"]["visual"] / f"{solution_file.stem}-diversity.png"))
    if show: plt.show()


def visualize_archived_distinct_solutions(projects: Dict[str, List[str]], fitness_thresholds: List[float],
                                          distance_thresholds: List[float], mr_set, tick_rate=0.2,
                                          box=False, save_path=None, show=False):
    """Plot the number of distinct solutions from final archived solutions by applying fitness and distance thresholds.

    :param projects: Project names of different algorithms. ``Dict[name_of_algorithm, List[project_name]]``.
    :param fitness_thresholds: A list of fitness thresholds.
    :param distance_thresholds: A list of distance thresholds.
    :param mr_set: The given MRs.
    :param tick_rate: Tick interval for x-axis.
    :param box: Show box plots.
    :param save_path: The path to save the figure.
    :param show: A boolean to determine whether to show the plots or not.
    """
    col_num, height = 3, 4
    title_size, text_size, tick_size = height * 4, height * 4, height * 3
    if len(fitness_thresholds) < col_num: col_num = len(fitness_thresholds)
    row_num = int(np.ceil(len(fitness_thresholds) / col_num))
    fig, axes = plt.subplots(row_num, col_num, figsize=(col_num * height * 1.2, row_num * height))
    ax_map = {gp_name: ax for ax, gp_name in zip(axes.reshape(-1), fitness_thresholds)}

    # vda = defaultdict(lambda: dict())
    data = {}
    mean_df = pd.DataFrame()
    for name, project_file in projects.items():
        df = pd.DataFrame()
        for project in project_file:
            solution_file = next((cfg.CONFIG["workspace"]["result"] / project / "solutions").rglob("solutions*"), None)
            if solution_file is None:
                logger.warning(f"No solution file found for {project}.")
                continue
            solutions = pickle.loads(solution_file.read_bytes())
            solution_df = _filter_by_thresholds(solutions, fitness_thresholds, distance_thresholds, mr_set,
                                                additional_metrics=["distinct_solution_num"])
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
                        yerr=values.apply(lambda row: 1.96 * np.std(row) / np.sqrt(len(row))),
                        **style_map.get(name, default_style), capsize=2, label=verbose_map.get(name, name), alpha=0.7)
            if box: ax.boxplot(group["distinct_solution_num"], positions=group.index.values, widths=0.05,
                               patch_artist=True, manage_ticks=False, whis=(0, 100),
                               boxprops=dict(facecolor=style_map.get(name, default_style)["color"], alpha=0.4))
            ax.set_title(f"Fitness threshold ($\\theta_f={gp_name}$)", fontsize=title_size)
            ax.tick_params(labelsize=tick_size)
            ax.set_xlabel("Distance threshold ($\\theta_d$)", fontsize=text_size)
            ax.set_ylabel("Average $DS$", fontsize=text_size)
            ax.xaxis.set_major_locator(MultipleLocator(tick_rate))
            ax.yaxis.set_major_locator(MaxNLocator(integer=True, min_n_ticks=1))
            ax.legend()
            ax.grid(True, which="both")
    calculate_ds_improvements(mean_df)
    # for fitness, d in vda.items():
    #     for alg1, alg2 in (("ccea", "ga"), ("ccea", "rs"), ("ga", "rs")):
    #         treatment, control = d[alg1], d[alg2]
    #         for i, (num1, num2) in enumerate(zip(treatment, control)):
    #             estimate, magnitude = Visualizer._vda(num1, num2)
    #             print(f"{fitness}-{distance_thresholds[i]}: {alg1}-{alg2}: {magnitude}-{estimate}.")

    fig.tight_layout()
    fig.savefig(save_path if save_path else
                (cfg.CONFIG["workspace"]["visualization"] / "archived_distinct_solutions.png"))
    if show: plt.show()
    return data


def visualize_distinct_solution_over_simulations(projects: Dict[str, List[str]],
                                                 fitness_thresholds: List[float], distance_thresholds: List[float],
                                                 max_sim_num: int, interval=10, mrc=False, mr_set=None,
                                                 save_path=None, show=False):
    """Plot the number of distinct solutions over simulations by applying fitness and distance thresholds.

    :param projects: Project names of different algorithms. ``Dict[name_of_algorithm, List[project_name]]``.
    :param fitness_thresholds: A list of fitness thresholds.
    :param distance_thresholds: A list of distance thresholds.
    :param max_sim_num: The maximum number of simulations.
    :param interval: Width of intervals for aggregation (percentage).
    :param mrc: Show the MR coverage.
    :param mr_set: The given MRs.
    :param save_path: The path to save the figure.
    :param show: A boolean to determine whether to show the plots or not.
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

    for name, project_list in projects.items():
        helper = pd.DataFrame({"simulation_num": ranges})
        agg_df_list = defaultdict(list)
        for project in project_list:
            checkpoints = sorted((cfg.CONFIG["workspace"]["result"] / project / "checkpoints").iterdir())
            df = pd.DataFrame()
            for checkpoint in checkpoints:
                with checkpoint.open("rb") as f:
                    for _ in range(get_skip(name)): pickle.load(f)
                    solutions = pickle.load(f)
                    for _ in range(2): pickle.load(f)
                    budget = pickle.load(f)
                ckp_df = _filter_by_thresholds(solutions, fitness_thresholds, distance_thresholds, mr_set,
                                               additional_metrics=["distinct_solution_num"])
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
            if mrc:
                y = agg_df["violated_mr_num"].apply(np.mean) * 100 / len(mr_set.mrs)
            else:
                y = agg_df["distinct_solution_num"].apply(np.mean)
            auc_df = pd.concat([auc_df, pd.DataFrame([{
                "alg": name,
                "fitness_threshold": gp_name[0],
                "distance_threshold": gp_name[1],
                "auc": area_under_curve(np.array([0, *percent_ranges]), np.array([0, *y]))
            }])], ignore_index=True)
            ax.plot(percent_ranges, y, **style_map.get(name, default_style), label=verbose_map.get(name, name))
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
    fig.savefig(save_path if save_path else
                (cfg.CONFIG["workspace"]["visualization"] / "distinct_solutions_over_simulations.png"))
    if show: plt.show()


def visualize_archived_solutions_by_gen(projects: Dict[str, List[str]], generation_num, metric_name,
                                        fitness_thresholds: List[float], distance_thresholds: List[float],
                                        mr_set, box=False, save_path=None, show=False,
                                        legend_loc="upper right", padding=None):
    """Plot the number of distinct solutions from final archived solutions by applying fitness and distance thresholds.

    :param projects: Project names of different algorithms. ``Dict[name_of_algorithm, List[project_name]]``.
    :param generation_num: Generation Number. If set to ``K``, plot the solutions found after ``K`` generations.
    :param metric_name: The metric used for comparison. Options are
        :data:`ds` (Distinct Solutions)
        :data:`avg_pw` (Average Pairwise Distance)
        :data:`pure_div` (Pure Diversity)
        :data:`avg_fit` (Average Fitness)
    :param fitness_thresholds: A list of fitness thresholds.
    :param distance_thresholds: A list of distance thresholds.
    :param mr_set: The given MRs.
    :param box: Show box plots.
    :param save_path: The path to save the figure.
    :param show: A boolean to determine whether to show the plots or not.
    :param legend_loc: Specifies the location of the legend in the plots.
    :param padding: Paddings between the extreme chart points and the axis range.
    """
    col_num, height = 3, 4
    title_size, text_size, tick_size = height * 4, height * 4, height * 3
    if len(distance_thresholds) < col_num: col_num = len(distance_thresholds)
    row_num = int(np.ceil(len(distance_thresholds) / col_num))
    fig, axes = plt.subplots(row_num, col_num, figsize=(col_num * height * 1.2, row_num * height))
    ax_map = {gp_name: ax for ax, gp_name in zip(axes.reshape(-1), distance_thresholds)}
    if padding is None: padding = {"top": 1.1, "bottom": 0.3}

    data = {}
    for alg, project_list in projects.items():
        df = pd.DataFrame()
        for i, project in enumerate(project_list):
            run = sorted((cfg.CONFIG["workspace"]["result"] / project / "checkpoints").iterdir())
            if len(run) < generation_num:
                cp = run[-1]
            else:
                cp = run[generation_num - 1]

            with cp.open("rb") as f:
                for _ in range(get_skip(alg)): pickle.load(f)
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
                [np.nanmean(row) + 1.96 * np.nanstd(row) / np.sqrt(np.count_nonzero(~np.isnan(row))) for row in values])
            chart_y_min = min(
                [np.nanmean(row) - 1.96 * np.nanstd(row) / np.sqrt(np.count_nonzero(~np.isnan(row))) for row in values])

            if not y_max or y_max < chart_y_max: y_max = chart_y_max
            if not y_min or y_min > chart_y_min: y_min = chart_y_min

            ax = ax_map[gp_name]
            ax.errorbar(fitness_thresholds, values.apply(np.nanmean),
                        yerr=values.apply(
                            lambda row: 1.96 * np.nanstd(row) / np.sqrt(np.count_nonzero(~np.isnan(row)))),
                        **style_map.get(alg, default_style), capsize=2, label=verbose_map.get(alg, alg))

            if box: ax.boxplot(group["distinct_solution_num"], positions=group.index.values, widths=0.05,
                               patch_artist=True, manage_ticks=False, whis=(0, 100),
                               boxprops=dict(facecolor=style_map.get(alg, default_style)["color"], alpha=0.4))
            ax.set_title(f"Distance threshold ($\\theta_d={gp_name}$)", fontsize=title_size)
            ax.tick_params(labelsize=tick_size)
            ax.set_xlabel("Fitness threshold ($\\theta_f$)", fontsize=text_size)
            ax.set_ylabel(verbose_map.get(metric_name, metric_name), fontsize=text_size)

            curr_y_min, curr_y_max = ax.get_ylim()

            if y_max + padding["top"] > curr_y_max:
                ax.set_ylim(top=y_max + padding["top"])
            if y_min - padding["bottom"] < curr_y_min:
                ax.set_ylim(bottom=y_min - padding["bottom"])
            ax.xaxis.set_major_locator(MultipleLocator(0.2))
            ax.yaxis.set_major_locator(MaxNLocator(integer=True, min_n_ticks=1))
            ax.legend(loc=legend_loc)
            ax.grid()

    fig.tight_layout()
    fig.savefig(save_path if save_path else
                (cfg.CONFIG["workspace"]["visualization"] / f"archived_{metric_name}_by_gen{generation_num}.png"))
    if show: plt.show()


def visualize_archive_solution_over_generations(projects: Dict[str, List[str]], metric_name,
                                                fitness_thresholds: List[float], distance_thresholds: List[float],
                                                mr_set, max_gen: int, save_path=None, show=False,
                                                legend_loc="upper right", padding=None):
    """Plot the metrics over generations by applying fitness and distance thresholds.

    :param projects: Project names of different algorithms. ``Dict[name_of_algorithm, List[project_name]]``.
    :param metric_name: The metric used for comparison. Options are
        :data:`ds` (Distinct Solutions)
        :data:`avg_pw` (Average Pairwise Distance)
        :data:`pure_div` (Pure Diversity)
        :data:`avg_fit` (Average Fitness)
    :param fitness_thresholds: A list of fitness thresholds.
    :param distance_thresholds: A list of distance thresholds.
    :param mr_set: The given MRs.
    :param max_gen: The maximum number of generations.
    :param save_path: The path to save the figure.
    :param show: A boolean to determine whether to show the plots or not.
    :param legend_loc: Location of the legend in the plots.
    :param padding: Paddings between the extreme chart points and the axis range.
    """
    col_num, height = 3, 4
    title_size, text_size, tick_size = height * 4, height * 4, height * 3
    total_size = len(fitness_thresholds) * len(distance_thresholds)
    if total_size < col_num: col_num = total_size
    row_num = int(np.ceil(total_size / col_num))
    fig, axes = plt.subplots(row_num, col_num, figsize=(col_num * height * 1.2, row_num * height))
    ax_map = {gp_name: ax for ax, gp_name in zip(axes.reshape(-1), product(fitness_thresholds, distance_thresholds))}
    if padding is None: padding = {"top": 0.6, "bottom": 0.3}

    data = {}
    for alg, project_list in projects.items():
        df = pd.DataFrame()
        for run_counter, project in enumerate(project_list):
            file_list = sorted((cfg.CONFIG["workspace"]["result"] / project / "checkpoints").iterdir())
            for i, file in enumerate(file_list):
                if i == max_gen: break
                with file.open("rb") as f:
                    for _ in range(get_skip(alg)): pickle.load(f)
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

            group[f"{metric_name}_mean"] = group[metric_name].apply(np.nanmean)
            group = group[group[f"{metric_name}_mean"].notna()].sort_values("gen")

            ax.plot(group["gen"],
                    group[f'{metric_name}_mean'],
                    **style_map.get(alg, default_style), label=verbose_map.get(alg, alg))
            ax.set_title(
                f"Fitness threshold ($\\theta_f={gp_name[0]}$),\nDistance threshold ($\\theta_d={gp_name[1]}$)",
                fontsize=title_size)
            ax.tick_params(labelsize=tick_size)
            # ax.xaxis.set_major_locator(MultipleLocator(interval))
            ax.yaxis.set_major_locator(MaxNLocator(integer=True, min_n_ticks=1))
            ax.set_xlabel("Generation", fontsize=text_size)
            ax.set_ylabel(verbose_map.get(metric_name, metric_name), fontsize=text_size)
            curr_y_min, curr_y_max = ax.get_ylim()
            if y_max + padding["top"] > curr_y_max:
                ax.set_ylim(top=y_max + padding["top"])
            if y_min - padding["bottom"] < curr_y_min:
                ax.set_ylim(bottom=y_min - padding["bottom"])

            ax.legend(loc=legend_loc)
            ax.grid()

    fig.tight_layout()
    fig.savefig(save_path if save_path else
                (cfg.CONFIG["workspace"]["visualization"] / f"archived_{metric_name}_over_generetations.png"))
    if show: plt.show()


def visualize_computational_efficiency(log_file: str, projects: Dict[str, List[str]], save_path=None, show=False):
    """
    Generate a boxplot comparing computational efficiency (duration in hours) across algorithms from a log file.

    :param log_file: Path to the CSV log file containing ``alg``, ``start_time``, ``end_time`` columns.
    :param projects: Project names of different algorithms. ``Dict[name_of_algorithm, List[project_name]]``.
    :param save_path: The path to save the figure.
    :param show: A boolean to determine whether to show the plots or not.
    """
    df = _generate_execution_time_data(log_file, projects)
    algorithms = list(projects.keys())
    durations = [df[df['alg'] == alg]['duration_hours'] for alg in algorithms]
    colors = [style_map.get(alg, default_style)['color'] for alg in algorithms]

    height = 4
    title_size, text_size, tick_size = height * 5, height * 4, height * 3
    fig = plt.figure(figsize=(height * 2, height * 2))
    box = plt.boxplot(durations, labels=[verbose_map.get(alg, alg) for alg in algorithms], patch_artist=True,
                      showmeans=True, medianprops=dict(color='black'),
                      meanprops=dict(marker='D', markerfacecolor='black', markeredgecolor='black', alpha=0.7),
                      showfliers=False)

    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)

    for i, (duration, color) in enumerate(zip(durations, colors), start=1):
        plt.scatter([i] * len(duration), duration, alpha=0.7, color=color)

    # plt.title('Comparison of Computational Efficiency (Duration in Hours)', fontsize=title_size)
    plt.xlabel('Algorithm', fontsize=text_size)
    plt.ylabel('Duration (Hours)', fontsize=text_size)
    plt.tick_params(labelsize=tick_size)
    plt.grid(axis='y', linestyle='--')
    plt.legend(handles=[Line2D([0], [0], marker='D', markerfacecolor='black', markeredgecolor='black',
                               color='w', alpha=0.7, linestyle='None', label='Mean value')], fontsize=text_size)
    plt.tight_layout()

    fig.tight_layout()
    fig.savefig(save_path if save_path else (cfg.CONFIG["workspace"]["visualization"] / "computational_efficiency.png"))
    if show: plt.show()


def visualize_scenario(scenario):
    carla_instance = cfg.CONFIG["simulation"]["instances"][0]
    client = carla.Client(carla_instance["host"], carla_instance["port"])
    world = client.load_world(scenario.town)
    blueprints = world.get_blueprint_library()
    world.set_weather(carla.WeatherParameters(
        **cfg.CONFIG["blueprint"]["scenario"]["weather"][scenario.weather],
        sun_altitude_angle=cfg.CONFIG["blueprint"]["scenario"]["brightness"][scenario.brightness]
    ))

    # Ego
    ego_bp = blueprints.find(cfg.CONFIG["blueprint"]["vehicle"]["model"][scenario.ego_vehicle.model])
    ego_start = carla.Transform(
        carla.Location(
            x=scenario.trajectory["start"]["x"],
            y=scenario.trajectory["start"]["y"],
            z=scenario.trajectory["start"]["z"]),
        carla.Rotation(yaw=scenario.trajectory["start"]["yaw"])
    )
    world.spawn_actor(ego_bp, ego_start)
    world.get_spectator().set_transform(carla.Transform(
        ego_start.location + carla.Location(z=50),
        carla.Rotation(pitch=-90)
    ))

    # Actors
    for typ in ("vehicle", "walker", "static"):
        actors = getattr(scenario, f"{typ}s")
        for actor in actors:
            actor_bp = blueprints.find(cfg.CONFIG["blueprint"][typ]["model"][actor.model])
            actor_start = carla.Transform(
                ego_start.transform(carla.Location(*polar_to_cartesian(actor.radius, actor.angle), z=0.2)),
                carla.Rotation(yaw=actor.yaw)
            )
            world.spawn_actor(actor_bp, actor_start)

    # Trajectory
    waypoints = [carla.Location(x=wp["x"], y=wp["y"], z=wp["z"]) for wp, _ in scenario.trajectory["route"]]
    lifetime = 100
    offset = 30
    arrow_interval = (len(waypoints) - offset) // 5

    for i in range(len(waypoints) - 1):
        start_point = waypoints[i]
        end_point = waypoints[i + 1]

        # Add some height to make it visible
        start_point.z += 0.15
        end_point.z += 0.15

        if (i + offset) % arrow_interval == 0:
            world.debug.draw_arrow(
                start_point,
                end_point,
                thickness=0.2,
                arrow_size=1.0,
                color=carla.Color(255, 0, 0),
                life_time=lifetime
            )
        else:
            # Draw line segment
            world.debug.draw_line(
                start_point,
                end_point,
                thickness=0.2,
                color=carla.Color(0, 255, 0),
                life_time=lifetime
            )
    for point, label in (
            (waypoints[0], "Start"),
            (waypoints[int((len(waypoints) - 1) * 0.5)], "Middle"),
            (waypoints[int((len(waypoints) - 1) * 0.75)], "3rd Quantile"),
            (waypoints[-1], "End"),
    ):
        world.debug.draw_point(
            carla.Location(point.x, point.y, point.z + 0.2),
            size=0.1,
            color=carla.Color(255, 0, 0),
            life_time=lifetime
        )
        world.debug.draw_string(
            carla.Location(point.x, point.y, point.z + 2.0),
            label,
            color=carla.Color(255, 0, 0),
            life_time=lifetime
        )


def _filter_by_thresholds(solutions, fitness_thresholds: List[float], distance_thresholds: List[float],
                          mr_set, additional_metrics: List[str] = None):
    default_columns = ["fitness_threshold", "distance_threshold", "violated_mr_num", "distinct_mr_num", "avg_pw"]
    if additional_metrics is None: additional_metrics = []
    additional_metrics = [metric for metric in additional_metrics if metric not in default_columns]
    column_names = (*default_columns, *additional_metrics)
    if len(solutions) == 0:
        return pd.DataFrame([[fitness_threshold, distance_threshold, 0, 0, np.nan,
                              *[metrics[metric](solutions) for metric in additional_metrics]]
                             for fitness_threshold in fitness_thresholds
                             for distance_threshold in distance_thresholds], columns=column_names)

    indices_to_remove = [[i for i, solution in enumerate(solutions) if solution.fitness.values[0] < threshold]
                         for threshold in fitness_thresholds]
    indices_of_violated_mrs = np.array(mr_set.violated_mrs([perturbations for _, perturbations in solutions]),
                                       dtype=object)
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


def _generate_execution_time_data(log_path: str, projects: Dict[str, List[str]]):
    log_df = _make_log_df(log_path)

    sol_cp_map = pd.DataFrame()
    for alg, project_list in projects.items():
        for project in project_list:
            sol_file = next((cfg.CONFIG["workspace"]["result"] / project / "solutions").rglob("solutions*"), None)
            checkpoints = sorted((cfg.CONFIG["workspace"]["result"] / project / "checkpoints").iterdir())
            start_cp, end_cp = checkpoints[0], checkpoints[-1]
            start_cp_time = pd.to_datetime(int(start_cp.stem), unit='ms')
            end_cp_time = pd.to_datetime(int(end_cp.stem), unit='ms')
            end_time = pd.to_datetime(int(sol_file.stem.split('-')[-1]), unit='ms')
            algo_logs = log_df[log_df["algorithm"] == alg]
            closest_log = (
                algo_logs[algo_logs["timestamp"] <= start_cp_time]
                .sort_values(by="timestamp", ascending=False)
                .iloc[0]
            )
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
