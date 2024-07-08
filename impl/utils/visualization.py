import logging
import os
import pickle
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from matplotlib.ticker import MaxNLocator
from tslearn.metrics import dtw_path

from impl.config import CONFIG
from impl.mr.mr import Relation

logger = logging.getLogger(__name__)

verbose_map = {
    'pop': 'Population',
    'pop_scen': 'Scenario',
    'pop_pert': 'Perturbation',
    'solution': 'Solution',
    'arc_scen': 'Scenario Archive',
    'arc_pert': 'Perturbation Archive',
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
    def visualize_in_one(file: str, verbose=False, plot_nan=True, show=False):
        """Plot all statistics data in one figure.

        @param file: Statistics data file.
        @param verbose: Show plots of populations and archives.
        @param plot_nan: Plot NaN values.
        @param show: A boolean to determine whether to show the plots or not.
        """
        with open(file, "rb") as f:
            logbook = pickle.load(f)
        stats = pd.DataFrame(logbook)
        if len(stats) == 0:
            logger.warning('No statistics provided. Nothing to visualize.')
            return

        for metric in ["std", "min", "avg", "max"]:
            stats[metric] = stats[metric].apply(lambda x: x[0])

        fig = plt.figure(figsize=(20, 8))
        if verbose:
            axs = [fig.add_subplot(2, 3, 5), fig.add_subplot(2, 3, 6)]
            for i in [3, 4, 1, 2]:
                axs.append(fig.add_subplot(2, 3, i, sharex=axs[1], sharey=axs[1]))
        else:
            axs = [fig.add_subplot(1, 2, 2), fig.add_subplot(1, 2, 1)]

        i = 1
        for pop_name, pop in stats.groupby("pop", sort=True):
            pop = pop.sort_values("gen", ascending=True)
            if pop_name == "solution" or verbose:
                for metric, fmt in [("std", ":C0"), ("min", "--C1"), ("avg", "o-C2"), ("max", "--C3")]:
                    if plot_nan:
                        masked = np.ma.masked_invalid(pop[metric])
                        np.ma.set_fill_value(masked, 0.0)
                        data = masked.filled()
                    else:
                        data = pop[metric]
                    axs[i].plot(pop["gen"], data, fmt, label=metric)
                axs[i].set_title(verbose_map[pop_name], fontsize=20)
                axs[i].tick_params(labelsize=13)
                axs[i].xaxis.set_major_locator(MaxNLocator(integer=True))
                if not verbose: axs[i].legend(fontsize=15)
                i += 1

            if pop_name == "solution":
                logger.info(f"Average number of violations: {np.mean(pop['len'])}.")
                logger.info(f"Average fitness of the last solution archive: {pop['avg'].iloc[-1]}.")
                logger.info(f"Growth rate of fitness: {np.polyfit(pop['gen'], pop['avg'], 1)[0]}.")
                axs[0].plot(pop["gen"], pop["len"], "o-C4", label="#violations")
                axs[0].plot(pop["gen"], pop["sim"], "x-C5", label="#simulations")
                axs[0].set_title("#Violations & #Simulations", fontsize=20)
                axs[0].set_ylabel("Num", fontsize=15)
                axs[0].tick_params(labelsize=13)
                axs[0].xaxis.set_major_locator(MaxNLocator(integer=True))
                axs[0].yaxis.set_major_locator(MaxNLocator(integer=True))
                axs[0].legend(fontsize=15)

        if verbose:
            fig.supxlabel("Generations", fontsize=15)
            fig.supylabel("Fitness", fontsize=15)
            handles, labels = axs[-1].get_legend_handles_labels()
            fig.legend(handles, labels, loc="lower right", ncol=4, fontsize=15)

        fig.tight_layout()
        fig.savefig(os.path.join(CONFIG["workspace"]["visualization"], f"{Path(file).stem}.png"))
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

        matches = [(source.index.values[i], follow_up.index.values[j])
                   for i, j in dtw_path(source[mr_set.field], follow_up[mr_set.field])[0]]
        labels = [f"{Relation._d}-{label}" for label in mr_set.labels or set()] \
            if CONFIG["violation"]["strategy"] == "simulation" else []
        dtw_df = pd.DataFrame([(
            source.loc[i, mr_set.field],
            follow_up.loc[j, mr_set.field],
            *[np.nanmin([source.loc[i].get(label, np.nan), follow_up.loc[j].get(label, np.nan)])
              for label in [Relation._d] + labels],
        ) for i, j in matches], columns=(Relation._s, Relation._f, *([Relation._d] + labels)))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8), sharey="all")
        for ax, df in zip((ax1, ax2), (origin_df, dtw_df)):
            ax.plot(df.index.values, df[Relation._s], "-C0", label="source")
            ax.plot(df.index.values, df[Relation._f] + (offset if ax is ax1 else 0), "-C1", label="follow up")
            ax.plot(df.index.values, df.apply(mr_set.relation._extent_func, axis=1, result_type="reduce"), "o:C2",
                    label="diff")
            ax.tick_params(labelsize=13)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        for x, y in matches:
            ax1.plot((x, y), (source.loc[x, mr_set.field], follow_up.loc[y, mr_set.field] + offset), "--", color="gray")

        critical_points = dtw_df.index[(dtw_df[labels].min(axis=1) if len(labels) > 0 else dtw_df[Relation._d])
                                       < CONFIG["violation"]["threshold"]["max_ego_distance"]]

        for points in np.split(critical_points, np.where(np.diff(critical_points) != 1)[0] + 1):
            ax2.axvspan(points[0] - 0.5, points[-1] + 0.5, color="red", alpha=0.1)

        ax1.set_title("Original difference", fontsize=20)
        ax2.set_title("DTW difference", fontsize=20)
        ax2.legend(fontsize=15)
        fig.supxlabel("Ticks", fontsize=15)
        fig.supylabel(mr_set.field.capitalize(), fontsize=15)
        fig.tight_layout()

        fig.savefig(os.path.join(CONFIG["workspace"]["visualization"], f"violation.png"))
        if show: plt.show()
