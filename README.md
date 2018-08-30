# Bayesian Change Point Detector

It works as follows:
1) Given the dataset, estimate some number number of changepoints using PyMC3.
This is very helpful [resource](http://www.claudiobellei.com/2017/01/25/changepoint-bayesian/) as well as
PyMC3 documentation and [this repo](https://github.com/hildensia/bayesian_changepoint_detection).
2) Given the locations of the changepoints, build Bayesian regression between each changepoints.
3) Calculate fit of the models in terms of RMSE.
4) Compare the fits of the models and choose the ones with lowest RMSE and take the number of changepoints
5) Plot the result and check if the number of changepoints meets the expectactions.

So far this method still needs some more tuning/adjusting as it is highly dependant on the fits of the regressions.
If the priors are really off, then it may not converge or converge to unsatisfactory results which may result in
too many/too few changepoints.

In this case we consider something to be a changepoint when the mean substantially changes.

One way of expanding this code would be to have some kind of function to choose _optimal_ number of changepoints
that takes into account both number of changepoints and fits (RMSE) of the regressions (maybe BIC or AIC type of thing).

This is the initial version that still needs a few adjustments and changes.