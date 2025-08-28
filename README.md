# CoCoMagic

CoCoMagic is an automated test case generation framework for evaluating autonomous systems.
It combines Cooperative Co-Evolutionary Algorithms (CCEAs) with Metamorphic Testing (MT) to discover test cases that
violate specified Metamorphic Relations (MRs).
The framework also supports Differential Testing to compare the behavior of different versions of an autonomous system.
In addition, it integrates an interpretability module to help diagnose the root causes of violations observed in the
generated test cases.
For autonomous driving system testing, CoCoMagic is compatible with the CARLA simulator and external driving agents,
such as InterFuser.

For more details about the design, please refer to our papers:

1. Hossein Yousefizadeh, Shenghui Gu, Lionel C. Briand, and Ali Nasr, "Constrained Co-evolutionary Metamorphic
Differential Testing for Autonomous Systems with an Interpretability Framework".
2. Hossein Yousefizadeh, Shenghui Gu, Lionel C. Briand and Ali Nasr, "Using Cooperative Co-Evolutionary Search to
Generate Metamorphic Test Cases for Autonomous Driving Systems," in *IEEE Transactions on Software Engineering*, vol.
51, no. 6, pp. 1882-1911, June 2025, doi: 10.1109/TSE.2025.3570897.


## Documentation

Build the documentation using the following commands (requires `Graphviz`):

```shell
cd docs
make clean
make html
```

The generated documentation is available at `docs/_build/html/index.html`.

## Citation

If you find our repo or paper useful, please cite us as:

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

## License

This project is licensed under the GPL-3.0 License - see the [LICENSE](LICENSE) file for details.
