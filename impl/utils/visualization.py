import logging
import os
import pickle
import time
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

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
    def visualize_in_one(data, verbose=False, plot_nan=True, show=False, file_name=None):
        """Plot all statistics data in one figure.

        @param data: Statistics data or data file.
        @param verbose: Show plots of populations and archives.
        @param plot_nan: Plot NaN values.
        @param show: A boolean to determine whether to show the plots or not.
        @param file_name: Specify the file name for plots when the data is not a file path.
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

        def _plot_metrics(_ax, _pop, _name, xlabel=False, ylabel=False, legend=False):
            for _metric, _fmt in [("std", ":C0"), ("min", "--C1"), ("avg", "o-C2"), ("max", "--C3")]:
                _ax.plot(_pop["gen"], _pop[_metric], _fmt, label=_metric)
            _ax.set_title(verbose_map[_name], fontsize=20)
            _ax.tick_params(labelsize=13)
            _ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            if xlabel: _ax.set_xlabel("Generation", fontsize=15)
            if ylabel: _ax.set_ylabel("Fitness", fontsize=15)
            if legend: _ax.legend(fontsize=15)

        fig = plt.figure(figsize=(6 * column_num, 4 * row_num))
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
                ax.set_title("#violations & #simulations", fontsize=20)
                ax.set_ylabel("Num", fontsize=15)
                ax.tick_params(labelsize=13)
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                ax.yaxis.set_major_locator(MaxNLocator(integer=True))
                ax.legend(fontsize=15)

        fig.supxlabel("Generation", fontsize=15)
        fig.tight_layout()
        fig.savefig(os.path.join(CONFIG["workspace"]["visualization"], f"{file_name}.png"))
        if show: plt.show()

    @staticmethod
    def visualize_comparison(files: Dict[str, List[str]], pop_name="archive",
                             box=True, interval=10, avg_line=False,
                             trend_line=True, regression_degree=3, all_lines=False,
                             plot_nan=True, show=False):
        """Plot comparisons among different algorithms.

        @param files: Statistics data files of different algorithms.
        @param pop_name: The name of the population for comparison. Options are "archive" and "solution".
        @param box: Show box plots.
        @param interval: Width of intervals for aggregation.
        @param avg_line: Show average lines.
        @param trend_line: Show trend lines.
        @param regression_degree: Degree of regression.
        @param all_lines: Show original lines.
        @param plot_nan: Plot NaN values.
        @param show: A boolean to determine whether to show the plots or not.
        """
        data = {}
        for name, file_list in files.items():
            df_list = []
            for file in file_list:
                with open(file, "rb") as f:
                    df = pd.DataFrame(pickle.load(f))
                df = df.groupby("pop").get_group(pop_name).sort_values("gen", ascending=True)
                for metric in ("std", "min", "avg", "max"):
                    df[metric] = df[metric].apply(lambda x: x[0])
                if plot_nan:
                    df.fillna(0, inplace=True)
                if pop_name == "solution":
                    df["sim"] = df["sim"].cumsum()
                df_list.append(df)
            df = pd.concat(df_list).sort_values("sim", ascending=True)
            lower = int(np.floor(df["sim"].iloc[0] / interval)) * interval
            upper = int(np.ceil(df["sim"].iloc[-1] / interval)) * interval
            agg = df.groupby(pd.cut(df["sim"], range(lower, upper + 1, interval))).agg(list).drop(columns="pop")
            agg = agg[agg["gen"].str.len() > 0]
            agg["sim"] = agg["sim"].apply(np.nanmean)
            data[name] = (df_list, df, agg)

        fig = plt.figure(figsize=(20, 12))
        ax1 = fig.add_subplot(3, 1, 1)
        ax2 = fig.add_subplot(3, 1, 2, sharex=ax1)
        ax3 = fig.add_subplot(3, 1, 3, sharex=ax1)
        legend_elements = {}
        for ax, metric, title in ((ax1, "len", "#violations"),
                                  (ax2, "max", "Max fitness"),
                                  (ax3, "avg", "Average fitness")):
            for (name, (df_list, df, agg)), color in zip(data.items(), ("C1", "C2", "C0", "C4")):
                if box:
                    ax.boxplot(agg[metric], positions=agg["sim"], widths=2, patch_artist=True, manage_ticks=False,
                               showfliers=False, boxprops=dict(facecolor=color, alpha=0.4))
                if avg_line:
                    ax.plot(agg["sim"], agg[metric].apply(np.nanmean), "o-", color=color)
                if trend_line:
                    f = np.poly1d(np.polyfit(df["sim"], df[metric], regression_degree))
                    ax.plot(df["sim"], f(df["sim"]), "-", color=color)
                if all_lines:
                    for line in df_list:
                        ax.plot(line["sim"], line[metric], "x--", color=color, alpha=0.6)
                if name not in legend_elements:
                    legend_elements[name] = Line2D([0], [0], linestyle="-", marker="o",
                                                   color=color, lw=2, label=verbose_map[name])
            ax.set_title(title, fontsize=20)
            ax.tick_params(labelsize=13)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.set_ylabel("#violations" if ax == ax1 else "Fitness", fontsize=15)
            ax.grid()

        ax1.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax1.legend(handles=legend_elements.values(), fontsize=15)
        fig.supxlabel("#simulations", fontsize=15)
        fig.tight_layout()
        fig.savefig(os.path.join(CONFIG["workspace"]["visualization"], "comparison.png"))
        if show: plt.show()

    @staticmethod
    def visualize_violation(source, follow_up, mr_set, offset=3, show=False):
        """Plot the extent of violation between the source results
        and follow-up results based on the given metamorphic relations.

        @param source: The source results.
        @param follow_up: The follow-up results.
        @param mr_set: The given metamorphic relations.
        @param offset: The offset between the source and follow-up curves.
        @param show: A boolean to determine whether to show the plots or not.
        """
        origin_index = source.index.union(follow_up.index)
        origin_df = pd.DataFrame([(
            source[mr_set.field].get(i, np.nan),
            follow_up[mr_set.field].get(i, np.nan)
        ) for i in origin_index], columns=(Relation._s, Relation._f))
        origin_df.set_index(origin_index, inplace=True)

        labels = Relation.convert_labels(mr_set.labels)
        matches, dtw_df = Relation.dtw_dataframe(source, follow_up, mr_set.field, labels)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8), sharey="all")
        for ax, df, title in zip((ax1, ax2), (origin_df, dtw_df), ("Original difference", "DTW difference")):
            ax.plot(df.index.values, df[Relation._s], "-C0", label="source")
            ax.plot(df.index.values, df[Relation._f] + (offset if ax is ax1 else 0), "-C1", label="follow-up")
            ax.plot(df.index.values, df.apply(mr_set.relation._extent_func, axis=1, result_type="reduce"),
                    "o:C2", label="difference")
            ax1.set_title(title, fontsize=20)
            ax.tick_params(labelsize=13)
            ax.set_xlabel("Tick" if ax == ax1 else "Match", fontsize=15)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax1.legend(fontsize=15)

        for x, y in matches:
            ax1.plot((x, y), (source.loc[x, mr_set.field], follow_up.loc[y, mr_set.field] + offset), "--", color="gray")

        critical_intervals = Relation.critical_intervals(dtw_df, labels)
        for points in np.split(critical_intervals, np.where(np.diff(critical_intervals) != 1)[0] + 1):
            ax2.axvspan(points[0] - 0.5, points[-1] + 0.5, color="red", alpha=0.1)

        fig.supylabel(mr_set.field.capitalize(), fontsize=15)
        fig.tight_layout()
        fig.savefig(os.path.join(CONFIG["workspace"]["visualization"], "violation.png"))
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
