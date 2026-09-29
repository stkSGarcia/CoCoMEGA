# CoCoMEGA and CoCoMagic

This repository contains the implementations of **CoCoMEGA** and **CoCoMagic**, two related methods for
automated test case generation for autonomous systems.

**CoCoMEGA** combines Cooperative Co-Evolutionary Algorithms (CCEAs) with Metamorphic Testing (MT) to
find test cases that violate specified Metamorphic Relations (MRs). **CoCoMagic** extends CoCoMEGA by
adding Differential Testing (DT), enabling comparisons between versions of a system under test to
identify differences in their metamorphic relation violations.

The shared framework includes an autonomous driving implementation using CARLA and InterFuser, along with
core abstractions and a domain template for extending it to other systems.

## Key Features

- **Co-evolutionary search:** Evolve scenarios and perturbations together to discover MR violations.
- **Differential testing (CoCoMagic):** Extend CoCoMEGA to compare MR violations across system versions.
- **Interpretability:** Analyze generated test cases to help diagnose observed violations.
- **Search baselines:** Compare co-evolutionary search with genetic algorithms and random search.
- **Extensible domains:** Define custom scenarios, metamorphic relations, and evaluators using the domain template.

## Documentation

The documentation covers installation, configuration, usage, framework architecture, and domain extension.
Build it locally, then follow the **Installation** and **Usage** guides to get started.

The build requires Sphinx, the Read the Docs theme, and Graphviz with its `dot` executable on `PATH`.
Run the following commands from the repository root:

```shell
python -m pip install sphinx sphinx-rtd-theme
cd docs
make html
```

Then open `docs/_build/html/index.html` (relative to the repository root) in your browser.
Generating the complete API reference also requires the project's runtime dependencies, since Sphinx imports
project modules during the build.

## Repository Structure

```text
conf/                   General configuration and logging defaults
docs/                   Sphinx documentation sources
impl/__main__.py        Command-line entry point
impl/core/              Search algorithms and shared abstractions
impl/ads/               CARLA and InterFuser integration
impl/sorting/           Sorting domain example
impl/domain_template/   Template for adding a domain
test/                   Test modules
```

## Related Papers

1. **CoCoMEGA:** Hossein Yousefizadeh, Shenghui Gu, Lionel C. Briand, and Ali Nasr, "Using Cooperative
   Co-Evolutionary Search to Generate Metamorphic Test Cases for Autonomous Driving Systems," in
   *IEEE Transactions on Software Engineering*, vol. 51, no. 6, pp. 1882–1911, June 2025,
   doi: [10.1109/TSE.2025.3570897](https://doi.org/10.1109/TSE.2025.3570897).
2. **CoCoMagic:** Hossein Yousefizadeh, Shenghui Gu, Lionel C. Briand, and Ali Nasr, "Constrained Co-evolutionary
   Differential Metamorphic Testing for Autonomous Systems with an Interpretability Approach," in
   *ACM Transactions on Software Engineering and Methodology*, June 2026,
   doi: [10.1145/3821438](https://doi.org/10.1145/3821438).

## Citation

If you use CoCoMEGA or CoCoMagic in your research, please cite the corresponding paper below.

### CoCoMEGA

```bibtex
@article{yousefizadeh2025using,
  author    = {Hossein Yousefizadeh and Shenghui Gu and Lionel C. Briand and Ali Nasr},
  title     = {Using Cooperative Co-Evolutionary Search to Generate Metamorphic Test Cases for Autonomous Driving Systems},
  journal   = {{IEEE} Transactions on Software Engineering},
  year      = 2025,
  month     = jun,
  volume    = {51},
  number    = {6},
  pages     = {1882--1911},
  publisher = {Institute of Electrical and Electronics Engineers ({IEEE})},
  address   = {Manhattan, NY, USA},
  doi       = {10.1109/tse.2025.3570897},
  url       = {http://doi.org/10.1109/TSE.2025.3570897}
}
```

### CoCoMagic

```bibtex
@article{yousefizadeh2026constrained,
  author    = {Hossein Yousefizadeh and Shenghui Gu and Lionel C. Briand and Ali Nasr},
  title     = {Constrained Co-evolutionary Differential Metamorphic Testing for Autonomous Systems with an Interpretability Approach},
  journal   = {{ACM} Transactions on Software Engineering and Methodology},
  year      = 2026,
  month     = jun,
  publisher = {Association for Computing Machinery ({ACM})},
  address   = {New York, NY, USA},
  doi       = {10.1145/3821438},
  url       = {https://doi.org/10.1145/3821438}
}
```

## License

This repository is licensed under the [GNU General Public License v3.0](LICENSE).
