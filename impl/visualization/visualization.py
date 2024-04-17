import plotly.graph_objects as go
import logging
import os
import pandas as pd
from impl.config import CONFIG
import time

logger = logging.getLogger(__name__)

verbose_map = {
    'pop_scen': 'Scenario',
    'pop_pert': 'Perturbation',
    'solution': 'Solution',
    'arc_scen': 'Scenario Archive',
    'arc_pert': 'Perturbation Archive',
}


class EvolutionVisualization:
    def __init__(self, stats):
        """
        Initialize the visualization with statistics data.
        :param stats: A dictionary containing lists of statistics per generation.
                      Expected keys are 'gen' for generation numbers,
                      'avg' for average fitness, 'max' for maximum fitness, etc.
        """
        self.stats = pd.DataFrame(stats)
        self.out_dir = os.path.join(CONFIG["workspace"], CONFIG["simulation"]["visualization"])

        if not os.path.exists(self.out_dir):
            os.mkdir(self.out_dir)

        self.metrics = [col for col in self.stats.columns if col not in ['pop', 'gen', 'len']]

        for metric in self.metrics:
            self.stats[metric] = self.stats[metric].apply(lambda x: x[0])

    def visualize(self, show):
        if len(self.stats) == 0:
            logger.warning('No statistics provided. Nothing to visualize.')
            return

        gen_stat_figs = self.plot_generation_statistics()
        if show:
            for fig in gen_stat_figs:
                self.show_plot(fig)

    def plot_generation_statistics(self):
        """
        Generates line plots for evolutionary algorithm statistics across generations.
        """
        figs = []
        for pop_name, pop in self.stats.groupby('pop'):
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
            self.save_fig(fig, name=f'{pop_name}_{int(round(time.time() * 1000))}.png')
            figs.append(fig)
        return figs

    def show_plot(self, fig):
        """
        Displays the plot.
        """
        fig.show()

    def save_fig(self, fig, name):
        """
        Save figure as image
        """
        fig.write_image(os.path.join(self.out_dir, name))

        # Example usage:
        # stats = {
        #     'gen': [1, 2, 3, ...],
        #     'avg': [0.5, 0.7, 0.9, ...],
        #     'max': [0.8, 0.9, 1.0, ...],
        # }
        # evol_vis = EvolutionVisualization(stats)
        # fig = evol_vis.plot_generation_statistics()
        # evol_vis.show_plot(fig)
