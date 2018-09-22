import numpy as np
import pymc3 as pm
import pandas as pd
import operator
from collections import Counter
from theano import tensor as T
from functools import reduce
from sklearn.preprocessing import MinMaxScaler


class ChangePointDetector(object):
    """Purpose: given data,
    return most likely number of changepoints,
    their locations and model instances fitted onto these locations
    """

    def __init__(self, data: 'list | array', num_changepoints: int):
        """Purpose: class constructor that
        instatiates data
        data - data we want to check for changepoints
        num_changepoints - maximum number of changepoints
        """
        
        if not isinstance(data, np.ndarray):
            try:
                data = np.asarray(data)
            except:
                raise TypeError("Input data must be convertible to numpy array!")
        
        
        # scale data to be in the range [0, 1]
        #self.scaler = MinMaxScaler(feature_range=(0, 1))
        #data = self.scaler.fit_transform(data.reshape(-1, 1))
        
        self.max_data = np.max(data)
        self.data = data / self.max_data
        self.num_changepoints = num_changepoints
        self.n_iter = 2000

        return None

    def get_changepoints_location(self, n_changepoints:int) -> 'changepoint locations, list':
        """Purpose: get most likely changepoint locations given
        data and number of changepoints.
        Heavily inspired by:
        http://www.claudiobellei.com/2017/01/25/changepoint-bayesian/
        """

        if len(self.data) < 10:
            raise ValueError("Received less than 10 data points!")

        if n_changepoints < 1:
            raise ValueError("Number of changepoints must be positive integer!")

        # dictionary of means
        self.mu_d = {}

        # stochastic means
        self.mus_d = {}

        # dictionary of changepoints
        self.tau_d = {}

        # define time range
        t = np.arange(len(self.data))

        with pm.Model() as model:
            # variance
            #sigma = pm.Uniform("sigma",1.e-3,20)
            sigma = pm.HalfNormal('sigma', sd=1)
            # define means
            for i in range(n_changepoints+1):
                self.mu_d['mu_{0}'.format(i)] = pm.Uniform('mu_{0}'.format(i), lower=self.data.min(),
                                                           upper=self.data.max(),)# testval=int(len(self.data)/(n_changepoints+1)*(i+1)))

            # define changepoint locations and stochastic variable(s) _mu
            for i in (range(n_changepoints)):
                if i == 0:
                    self.tau_d['tau_{0}'.format(i)] = pm.DiscreteUniform('tau_{0}'.format(i),
                                                                    t.min(), t.max())
                    self.mus_d['mus_d_{0}'.format(i)] = T.switch(self.tau_d['tau_{0}'.format(i)] >= t,
                                                                 self.mu_d['mu_{0}'.format(i)],
                                                                 self.mu_d['mu_{0}'.format(i+1)])
                else:
                    self.tau_d['tau_{0}'.format(i)] = pm.DiscreteUniform('tau_{0}'.format(i),
                                                                         self.tau_d['tau_{0}'.format(i-1)], t.max())
                    self.mus_d['mus_d_{0}'.format(i)] = T.switch(self.tau_d['tau_{0}'.format(i)] >= t,
                                                                 self.mus_d['mus_d_{0}'.format(i-1)],
                                                                 self.mu_d['mu_{0}'.format(i+1)])

            def logp_func(data):
                """Function to be provided to PyMC3
                """
                return logp.sum()

            # define log-likelihood for the parameters given data
            logp = - T.log(sigma * T.sqr(2.0 * np.pi)) \
               - T.sqr(self.data - self.mus_d['mus_d_{0}'.format(i)]) / (2.0 * sigma * sigma)

            # define density dist
            L_obs = pm.DensityDist('L_obs', logp_func, observed=self.data)

            # start MCMC algorithm
            start = pm.find_MAP()
            step = pm.Metropolis()
            # iterate MCMC
            trace = pm.sample(self.n_iter, step, start=start, random_seed=123, progressbar=False)

            # calculate changepoints
            locations = []
            for k in range(n_changepoints):
                changepoint = Counter(trace.get_values('tau_{0}'.format(k))).most_common(1)[0][0]
                locations.append(changepoint)

        return sorted(set(locations)), trace

    @staticmethod
    def build_bayesian_regression(data):
        """Purpose: given the data, estimate
        linear model and return model fit
        in terms of RMSE.
        This function is taken from PyMC3 homepage
        """

        # number of draws
        n_iter = 2000

        with pm.Model() as model:
            # Define priors
            x = np.arange(len(data))
            sigma = pm.HalfNormal('sigma', sd=1)
            intercept = pm.Normal('intercept', mu=0, sd=1)
            x_coeff = pm.Normal('x_coef', mu=0, sd=1)

            # Define likelihood
            likelihood = pm.Normal('y', mu=intercept + x_coeff * x,
                                sd=sigma, observed=data)

            # Inference!
            step = pm.Metropolis()
            trace = pm.sample(n_iter, step, random_seed=123,
                              progressbar=False)

            # evaluate model fit
            df = pd.DataFrame(pm.summary(trace))
            prediction = df['mean']['intercept'] + df['mean']['x_coef'] * x

        return prediction, trace

    def get_models_fit_for_changepoints(self, changepoints_locations):
        """Purpose: given the changepoint locations
        fit linear models in each of the interval
        """

        if changepoints_locations == []:
            raise ValueError('changepoint location is empty!')

        # cut the data based on changepoint locations
        args = (0,) + tuple(c+1 for c in changepoints_locations) + (len(self.data)+1,)
        data_lists = [self.data[s:e] for s, e in zip(args, args[1:])]

        # get predictions and traces for all the intervals
        models_predictions = []
        traces_list = []
        for i in data_lists:
            pred, trace = ChangePointDetector.build_bayesian_regression(i)
            models_predictions.append(pred)
            traces_list.append(trace)

        # now calculate rmse for the given number of changepoints
        self.models_predictions = models_predictions
        total_preds = reduce(operator.concat, [list(i) for i in models_predictions])

        assert len(self.data) == len(total_preds), "Predictions are shorter than actual data!"

        # calculate rmse
        rmse = np.sqrt(np.mean((self.data - total_preds)**2))

        return rmse, traces_list, total_preds

    def calculate_changepoint_locations(self):
        """Purpose: given the changepoints, estimate regressions
        and compare the results
        """

        # define changepoints dictionary
        changepoints_ = {}
        changepoint_traces_ = {}
        # define rmse dictionary for given number of changepoints
        rmse_models_fit = []
        traces_ = []
        model_preds_list = []

        # for maximum number of changepoints
        for c in range(1, self.num_changepoints+1):
            print("CHECKING CHANGEPOINT {0}".format(c))
            # location of changepoints
            cp_locs, cp_trace = self.get_changepoints_location(n_changepoints=c)
            changepoints_[c] = cp_locs 
            changepoint_traces_[c] = cp_trace

            # for each changepoint in changepoints_ fit linear models and return fit
            # in terms of rmse
            rm, traces_list, model_preds = self.get_models_fit_for_changepoints(changepoints_[c])
            rmse_models_fit.append(rm)
            traces_.append(traces_list)
            model_preds_list.append(model_preds)

        # now given the rmse and changepoint locations, find the smallest rmse
        # and correspoding changepoint locations
        changepoints_rmse = dict(zip(rmse_models_fit, changepoints_.values()))

        # for debugging purposes
        self.rmse_models_fit = rmse_models_fit
        self.changepoints_ = changepoints_
        self.traces_ = traces_
        self.changepoint_traces_ = changepoint_traces_
        self.model_preds_list = model_preds_list

        min_rmse_cp_locations = changepoints_rmse[min(rmse_models_fit)]

        # get predictions for given rmse
        preds_out = model_preds_list[rmse_models_fit.index(min(rmse_models_fit))]
        
        # return transformed mean predictions to make sure they are on the same scale
        #preds_out = self.scaler.inverse_transform(np.asarray(preds_out).reshape(-1, 1))
        preds_out = [i * self.max_data for i in preds_out]

        return min_rmse_cp_locations, preds_out