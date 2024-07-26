import logging
import os
import pickle
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from scipy.spatial.distance import squareform, pdist
from sklearn.manifold import MDS

from impl.config import CONFIG
from impl.mr.mr import Relation

logger = logging.getLogger(__name__)

verbose_map = {
    'pop': 'Population',
    'pop_scen': 'Population—Scenario',
    'pop_pert': 'Population—Perturbation',
    'solution': 'Complete Solutions',
    'archive': 'Archive',
    'arc_scen': 'Archive—Scenario',
    'arc_pert': 'Archive—Perturbation',
    'ccea': 'CCEA',
    'rs': 'Random Search',
    'ga': 'Standard Genetic Algorithm',
    'gawa': 'SGA with Archives',
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

    @staticmethod
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
        Visualizer._log_statistics(groups)
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

    @staticmethod
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
                    first_row, last_row = df.iloc[0].copy(), df.iloc[-1].copy()
                    first_row["sim"], last_row["sim"] = low, high
                    df = pd.concat([pd.DataFrame([first_row]), df, pd.DataFrame([last_row])], ignore_index=True)
                    agg_df = pd.merge(df, helper, on="sim", how="outer").sort_values("sim", ascending=True)
                    for metric in ("len", "std", "min", "avg", "max"):
                        agg_df[metric].interpolate("linear", inplace=True)
                    agg_df = agg_df[agg_df["sim"] % interval == 0]
                    agg_list.append(agg_df)
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
            for (name, (full, merged, agg)), color in zip(data.items(), ("C1", "C2", "C0", "C4")):
                full, merged, agg = full[pop_name], merged[pop_name], agg[pop_name]
                if box:
                    ax.boxplot(agg[metric], positions=agg.index.values, widths=2, patch_artist=True, manage_ticks=False,
                               whis=(0, 100), boxprops=dict(facecolor=color, alpha=0.4))
                if avg_line:
                    ax.plot(agg.index.values, agg[metric].apply(np.nanmean), "o-", color=color)
                if trend_line:
                    f = np.poly1d(np.polyfit(merged["sim"], merged[metric], regression_degree))
                    ax.plot(merged["sim"], f(merged["sim"]), "-", lw=2, color=color)
                if all_lines:
                    for line in full:
                        ax.plot(line["sim"], line[metric], "x--", lw=1, color=color, alpha=0.6)
                if name not in legend_elements:
                    legend_elements[name] = Line2D([0], [0], linestyle="-", marker="o",
                                                   color=color, lw=2, label=verbose_map[name])
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

    @staticmethod
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

        for x, y in matches:
            ax1.plot((x, y), (source.loc[x, mr_set.field], follow_up.loc[y, mr_set.field] + offset),
                     "--", color="gray", zorder=1)

        critical_intervals = Relation.critical_intervals(pair_df, labels)
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

                for x, y in matches:
                    ax4.plot((source.loc[x, "position_x"], follow_up.loc[y, "position_x"] - offset),
                             (source.loc[x, "position_y"], follow_up.loc[y, "position_y"] - offset),
                             "--", color="gray", zorder=1)

        fig.tight_layout()
        fig.savefig(os.path.join(CONFIG["workspace"]["visualization"], "violation.png"))
        if show: plt.show()

    @staticmethod
    def visualize_diversity(file, show=False):
        with open(file, "rb") as f:
            solutions = pickle.load(f)
        if len(solutions) < 2:
            logger.warning("No solutions or only one solution found.")
            return

        dist = lambda x, y: np.sqrt((x[0].dist(y[0])) ** 2 + (x[1].dist(y[1])) ** 2)
        from impl.algorithm.base import BaseAlgorithm
        diversity = BaseAlgorithm.population_diversity(solutions, dist)
        logger.info(f"Population diversity: {diversity}, size: {len(solutions)}.")

        dist_matrix = squareform(pdist(np.array(solutions, dtype=object).reshape((len(solutions), -1)), dist))
        out = MDS(n_components=3, dissimilarity="precomputed").fit(dist_matrix).embedding_
        fig, ax = plt.subplots(figsize=(10, 8))
        ax = plt.axes(projection="3d")
        ax.scatter3D(out[:, 0], out[:, 1], out[:, 2])
        ax.view_init(azim=-5, elev=145)
        ax.set_box_aspect((np.ptp(out[:, 0]), np.ptp(out[:, 1]), np.ptp(out[:, 2])))
        fig.tight_layout()
        fig.savefig(os.path.join(CONFIG["workspace"]["visualization"], f"{Path(file).stem}-diversity.png"))
        if show: plt.show()

    @staticmethod
    def _log_statistics(groups):
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
