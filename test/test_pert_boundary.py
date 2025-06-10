import logging
import os
import sys

import yaml
import time

import subprocess

import pandas as pd
import test

from impl.ads.utils.visualization import Visualizer

config = test.CONFIG
logger = logging.getLogger("impl")


class TestPertBoundary:
    def __init__(self, resume):
        self.resume = resume
        self.base_dir = config["workspace"]["test_result"]
        if not os.path.exists(self.base_dir):
            os.mkdir(self.base_dir)
        if self.resume:
            test_dir = sorted([l for l in os.listdir(config["workspace"]["test_result"]) if l.startswith('pert_boundary')], reverse=True)[0]
            self.out_dir = os.path.join(self.base_dir, test_dir)
            logger.info(f"Resuming test {test_dir}")
        else:
            self.out_dir = os.path.join(self.base_dir,
                                        f'pert_boundary_{int(round(time.time() * 1000))}')
            if not os.path.exists(self.out_dir):
                os.mkdir(self.out_dir)

    def run(self):
        try:
            meta = pd.read_pickle(os.path.join(self.out_dir, "meta.pkl"))
            covered_angles = list(meta["name"])
        except:
            meta = pd.DataFrame()
            covered_angles = []

        angles = [20, 30, 40, 50]
        for angle in angles:
            if str(angle) in covered_angles:
                logger.info(f"Angle {angle} is already covered!")
                continue

            with open('config.yaml', 'r') as file:
                data = yaml.safe_load(file)

                data["boundary"]["region"] = {
                    "focus": {
                        "angle": [-float(angle), float(angle)],
                        "radius": [5, 30]
                    },
                    "left": {
                        "angle": [-float(angle), -float(angle)],
                        "radius": [5, 30]
                    },
                    "right": {
                        "angle": [float(angle), float(angle)],
                        "radius": [5, 30]
                    },
                }

            with open('config.yaml', 'w') as file:
                yaml.safe_dump(data, file)

            process = subprocess.run([
                f".venv/bin/python -m impl srch"
            ],
                shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if process.returncode == 0:
                logger.info("Boundary test completed.")
            else:
                raise Exception(
                    f"There was an error running boundary test: {process.stdout}, {process.stderr}"
                )

            files = sorted(
                [item for item in os.listdir(config["workspace"]["solution"]) if 'statistics' in item],
                reverse=True)
            logger.info(f"Statistics path: {files[0]}.")

            meta = pd.concat([meta, pd.DataFrame([{
                'name': str(angle),
                'statistics_path': files[0],
            }])])

            meta.to_pickle(os.path.join(self.out_dir, 'meta.pkl'))

        Visualizer.plot_pert_boundary_results(meta, out_dir=self.out_dir)

        return meta


if __name__ == '__main__':
    resume = '-r' in sys.argv
    meta = TestPertBoundary(resume).run()
