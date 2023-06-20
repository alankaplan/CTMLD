Continuous-Time Models for Longitudinal Data (CTMLD) is coded in Python and estimates models that are useful for longitidinal data, such as medical health records. The model is a probability density between two measures and the time interval between them. An estimated model can be used to make future predictions, compute odds ratios of outcomes, or estimate risk.

See the following paper for more information on the methodology [1]: http://dx.doi.org/10.1016/j.jbi.2022.104084.

LLNL-CODE-848447

Data Format
----------------
Data should be formatted as a text file and readable by numpy.loadtxt. The first row contains the time intervals, the second row contains the first measure, and the third row contains the second measure.
This is an example data file containing four measurements of each variable.
```
1.4 5.3 0.1 0.5
0.7 0.2 2.1 3.5
4 3 4 2
```

Example Usage
----------------
To train a model on the sample data `0.0_ex_dat.txt` (data corresponding to Section 3.1.1 and Figure 2 of [1]), run:
```
python train_model.py 0.0_ex_dat.txt log0 model0
```
As the model is training, Bayesian Optimization (BO) is used to select the model dimension. The JSON log file shows every (BO) iteration performed. Within each BO iteration, Expectation Maximization (EM) is used to estimate parameters for a given model size.

To train a model on the sample data `0.1_ex_dat.txt` (data corresponding to Section 3.1.2 and Figure 3 of [1]), run:
```
python train_model.py 0.1_ex_dat.txt log1 model1
```

Citation
----------------
[1] Kaplan, Alan D., Uttara Tipnis, Jean C. Beckham, Nathan A. Kimbrel, David W. Oslin, and Benjamin H. McMahon. 2022. “Continuous-Time Probabilistic Models for Longitudinal Electronic Health Records.” Journal of Biomedical Informatics.
