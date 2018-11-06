# general-qse

Generalized Qualitative State Estimation (QSE)

This Script is based on the QSE algorithme, which Kris Villez implemented in Matlab. It produces a qualitative probabilistic estimation of a singal, which tells with which probabilty it follows a certain shape (e.g. goes down, up, stay constant, etc).  This method, based on the earlier developed Qualtiative Path estimation is further generalized by allowing all 39 different combinations of signs between
signal, first and second derivative and implements the possibility to choose from different kernels. Furthermore it is capable of handling different polynom order bigger than 1. This version is three times faster than the matlab version, because it's not using a loop over each moving window, but applying matrix calculations of all moving windows at same time. A estimation of the best bandwidth is uncluded by using generalized cross validation.

## References

The standard method is well described in the following two papers:

- Villez, K., & Rengaswamy, R. (2013, July). A generative approach to qualitative trend analysis for batch process fault diagnosis. In Control Conference (ECC), 2013 European (pp. 1958-1963). IEEE.
- Thürlimann, C. M., Dürrenmatt, D. J., & Villez, K. (2015). Evaluation of qualitative trend analysis as a tool for automation. In Computer Aided Chemical Engineering (Vol. 37, pp. 2531-2536). Elsevier.
      
The used ICI method is well decribed in:

- Sucic, V., Lerga, J., & Vrankic, M. (2013). Adaptive filter support selection for signal denoising based on the improved ICI rule.        Digital Signal Processing, 23(1), 65-74.

The used iteratively reweighted least square mehtod is presented in:

- Chan, S. C., & Zhang, Z. (2004). Robust local polynomial regression using M-estimator with adaptive bandwidth. In Proceedings-IEEE International Symposium on Circuits and Systems. IEEE..

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

- qse_algorithme.py: holds the main class generalQSE with its main methods run and plot.
- evaluation_script.py: evaluate a qualitative trend of the prediction data and ground_truth data and compares performance (classification accuracy and cross entropy) of different general QSE scenarios.
- diff_metric.py: auxilary file, which holds some difference metrics for comparing two qualitative trends.


### Prerequisites

- Python 3.6  (If not installed, go to  https://www.python.org/downloads/windows/,download Python 3.6.7 (64 bit) version and run installer)
- Git (If git is not installed, install it from https://git-scm.com/download/win, leave all install options to default)
- Pipenv (if not installed go to the Script directory, where your installed python is located (in my case C:\APPS\Python\Python36\Scripts), execute in cmd: pip install pipenv

### Installing

A step by step series of examples that tell you how to get a development env running

    1.	Clone Code from Github (you can delete the other code we cloned)
        a.	In the cmd go to the directory you want the code to be cloned in.
        b.	In cmd: git clone https://github.com/kramersi/general-qse.git 
    2.	Create virtualenv for that project with pipenv
        a.	In cmd: cd general-qse  (to go into code directory)
        b.	In cmd: pipenv install (creating virtualenv for that project folder)
    3. Run script qse_algorithme in a console or common IDE like pyCharm


### Running the script

      pipenv run qse_algorithm.py
      
or

      pipenv shell
      python qse_algorithm.py

## Authors

* **Simon Kramer** - *Initial work* - [kramersi](https://github.com/kramersi)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License
Copyright (C) 2018 Kris Villez, Simon Kramer

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program.
If not, see <http://www.gnu.org/licenses/>.

## Acknowledgments

* Kris Villez for helping implementing it
* Matthew Moy de Vitry for good discussions about how to apply it


