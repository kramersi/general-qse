# general-qse

Generalized Qualitative State Estimation (QSE)

It's still under development.

This Script is based on the QSE algorithme, which Kris Villez implemented in Matlab. It produces a qualitative probabilistic estimation of a singal, which tells with which probabilty it follows a certain shape (e.g. goes down, up, stay constant, etc).  This method, based on the earlier developed Qualtiative Path estimation is further generalized by allowing all 39 different combinations of signs between
signal, first and second derivative and implements the possibility to choose from different kernels. Furthermore it is capable of handling different polynom order bigger than 1. This version is three times faster than the matlab version, because it's not using a loop over each moving window, but applying matrix calculations of all moving windows at same time. A estimation of the best bandwidth is uncluded by using generalized cross validation.

The standard method is well described in the following two papers:

    - Villez, K., & Rengaswamy, R. (2013, July). A generative approach to qualitative trend analysis for batch process
      fault diagnosis. In Control Conference (ECC), 2013 European (pp. 1958-1963). IEEE.
    - Thürlimann, C. M., Dürrenmatt, D. J., & Villez, K. (2015). Evaluation of qualitative trend analysis as a tool for
      automation. In Computer Aided Chemical Engineering (Vol. 37, pp. 2531-2536). Elsevier.


Copyright (C) 2018 Kris Villez, Simon Kramer

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program.
If not, see <http://www.gnu.org/licenses/>.
