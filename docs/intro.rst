Introduction
============

CoCoMagic is an automated test case generation framework for evaluating autonomous systems.
It combines Cooperative Co-Evolutionary Algorithms (CCEAs) with Metamorphic Testing (MT) to discover test cases that
violate specified Metamorphic Relations (MRs).
The framework also supports Differential Testing to compare the behavior of different versions of an autonomous system.
In addition, it integrates an interpretability module to help diagnose the root causes of violations observed in the
generated test cases.
For autonomous driving system testing, CoCoMagic is compatible with the CARLA simulator and external driving agents,
such as InterFuser.

For more details about the design, please refer to our papers:

#. Hossein Yousefizadeh, Shenghui Gu, Lionel C. Briand, and Ali Nasr, "Constrained Co-evolutionary Metamorphic
   Differential Testing for Autonomous Systems with an Interpretability Framework".

#. Hossein Yousefizadeh, Shenghui Gu, Lionel C. Briand and Ali Nasr, "Using Cooperative Co-Evolutionary Search to
   Generate Metamorphic Test Cases for Autonomous Driving Systems," in *IEEE Transactions on Software Engineering*, vol.
   51, no. 6, pp. 1882-1911, June 2025, doi: 10.1109/TSE.2025.3570897.
