"""Finite mixture of Student's t-distributions fit using EM.

Author: Jonathan Parkinson <jlparkinson1@gmail.com>
License: MIT
"""
import numpy as np
from scipy.special import logsumexp, digamma, polygamma
from sklearn.cluster import KMeans
from .utilities import sq_maha_distance, scale_update_calcs
from .mixture_base_class import MixtureBaseClass
from ._backend import get_array_module, to_device, to_numpy



class EMStudentMixture(MixtureBaseClass):
    """A finite student's t mixture model fitted using the EM algorithm.

    Attributes:
        start_df_ (float): The starting value for degrees of freedom.
        fixed_df (bool): If True, degrees of freedom are fixed and are not optimized.
        n_components (int): The number of mixture components.
        tol (float): The threshold at which the fitting process is determined to have
            converged.
        reg_covar (float): A small constant added to the diagonal of scale matrices
            for numerical stability; provides regularization.
        max_iter (int): The maximum number of iterations per restart during fitting.
        init_type (str): The procedure for initializing the cluster centers; one of
            either 'kmeans' or 'k++'.
        n_init (int): The number of restarts (since fitting can converge on a local
            maximum).
        random_state (int): The random seed for random number generator initialization.
        verbose (bool): If True, print updates throughout fitting.
        covariance_type (str): The type of covariance parameters to use. One of
            'full' (each component has its own full covariance matrix, shape M x M x K)
            or 'diag' (each component has its own diagonal covariance, stored as M x K).
        mix_weights_ (np.ndarray): The mixture weights; a 1d numpy array of shape K
            for K components.
        location_ (np.ndarray): The cluster centers; corresponds to the mean values
            in a Gaussian mixture model. A 2d numpy array of shape K x M (for K
            components, M input dimensions.)
        scale_ (np.ndarray): The component scale matrices; corresponds to the covariance
            matrices of a Gaussian mixture model. A 3d numpy array of shape M x M x K
            for M input dimensions, K components. For 'diag', shape is M x K.
        scale_cholesky_ (np.ndarray): The cholesky decompositions of the scale matrices;
            same shape as the scale_ attribute. For 'diag', contains the square roots
            of the diagonal elements, shape M x K.
        converged_ (bool): Indicates whether the model converged during fitting.
        n_iter_ (int): The number of iterations completed during fitting.
        df_ (np.ndarray): The degrees of freedom for each mixture component; a 1d
            array of shape K for K components.
    """

    def __init__(self, n_components = 2, tol=1e-5,
            reg_covar=1e-06, max_iter=1000, n_init=1,
            df = 4.0, fixed_df = True, random_state=123, verbose=False,
            init_type = "kmeans", covariance_type="full", n_jobs=1,
            device="cpu"):
        """Constructor for EMStudentMixture.

        Args:
            n_components (int): The number of mixture components. Defaults to 2.
            tol (float): A threshold below which fitting is determined to have
                converged. Defaults to 1e-5.
            reg_covar (float): A regularization parameter added to the diagonal
                of the scale matrices for numerical stability. Defaults to 1e-6,
                which is a good value in general.
            max_iter (int): The maximum number of iterations per restart. Defaults
                to 1000.
            n_init (int): The number of restarts (since a restart may converge on
                a local maximum). Defaults to 1.
            df (float): The starting value for degrees of freedom for all mixture
                components. Defaults to 4.0.
            fixed_df (bool): If True, df_ remains fixed at the starting value and
                is not optimized. Defaults to True.
            random_state (int): The seed for the random number generator for
                initializing the model.
            verbose (bool): If True, print updates throughout fitting. Defaults to
                False.
            init_type (str): One of 'kmeans', 'k++'. Determines how cluster centers
                are initialized. 'kmeans' provides better performance and is the
                default; 'k++' may be slightly faster.
            covariance_type (str): One of 'full', 'diag', 'tied', 'spherical'.
                'full' uses unrestricted covariance matrices (M x M x K).
                'diag' uses diagonal covariance matrices (stored as M x K).
                'tied' and 'spherical' are not yet implemented.
                Defaults to 'full'.
            n_jobs (int): Number of parallel jobs for n_init restarts. 1 means
                sequential (default). -1 uses all available cores. Requires
                joblib (installed with scikit-learn). Defaults to 1.
            device (str): Compute device. 'cpu' uses NumPy/SciPy (default).
                'gpu' uses CuPy for GPU acceleration (requires CuPy installed).
                Defaults to 'cpu'.
        """
        super().__init__()
        self._validate_covariance_type(covariance_type)
        self.check_user_params(n_components, tol, reg_covar, max_iter, n_init, df, random_state,
                init_type)
        self.start_df_ = float(df)
        self.fixed_df = fixed_df
        self.n_components = n_components
        self.tol = tol
        self.reg_covar = reg_covar
        self.max_iter = max_iter
        self.init_type = init_type
        self.n_init = n_init
        self.random_state = random_state
        self.verbose = verbose
        self.covariance_type = covariance_type
        self.n_jobs = n_jobs
        self.device = device
        self._use_gpu = (device == 'gpu')
        self.mix_weights_ = None
        self.location_ = None
        self.scale_ = None
        self.scale_cholesky_ = None
        self.converged_ = False
        self.n_iter_ = 0
        self.df_ = None


    def _validate_covariance_type(self, covariance_type):
        """Validate the covariance_type parameter.

        Args:
            covariance_type (str): The covariance type to validate.

        Raises:
            ValueError: If covariance_type is not a recognized string.
            NotImplementedError: If covariance_type is 'tied' or 'spherical'.
        """
        if covariance_type in ('tied', 'spherical'):
            raise NotImplementedError(
                f"covariance_type='{covariance_type}' is not yet implemented. "
                f"Use 'full' or 'diag'."
            )
        if covariance_type not in ('full', 'diag'):
            raise ValueError(
                f"Invalid covariance_type '{covariance_type}'. "
                f"Must be one of: 'full', 'diag', 'tied', 'spherical'."
            )




    def fit(self, X):
        """Fit model using the parameters the user selected when creating the model object.
        Creates multiple restarts by calling the fitting_restart function n_init times.
        When n_jobs != 1, restarts run in parallel using joblib.

        Args:
            X (np.ndarray): The raw data for fitting. This must be either a 1d array, in which case
                self.check_fitting_data will reshape it to a 2d 1-column array, or
                a 2d array where each column is a feature and each row a datapoint.
        """
        x = self.check_fitting_data(X)
        best_lower_bound = -np.inf

        if self.n_jobs == 1 or self.n_init == 1:
            # Sequential path
            for i in range(self.n_init):
                lower_bound, convergence, loc_, scale_, mix_weights_,\
                        df_, scale_cholesky_ = self.fitting_restart(x, self.random_state + i)
                if self.verbose:
                    print(f"Restart {i} now complete")
                if not convergence:
                    print(f"Restart {i+1} did not converge!")
                elif lower_bound > best_lower_bound:
                    best_lower_bound = lower_bound
                    self.df_, self.location_, self.scale_ = df_, loc_, scale_
                    self.scale_cholesky_ = scale_cholesky_
                    self.mix_weights_ = mix_weights_
                    self.converged_ = True
        else:
            from joblib import Parallel, delayed
            results = Parallel(n_jobs=self.n_jobs)(
                delayed(self.fitting_restart)(x, self.random_state + i)
                for i in range(self.n_init)
            )
            for i, (lower_bound, convergence, loc_, scale_,
                    mix_weights_, df_, scale_cholesky_) in enumerate(results):
                if self.verbose:
                    print(f"Restart {i} now complete")
                if not convergence:
                    print(f"Restart {i+1} did not converge!")
                elif lower_bound > best_lower_bound:
                    best_lower_bound = lower_bound
                    self.df_, self.location_, self.scale_ = df_, loc_, scale_
                    self.scale_cholesky_ = scale_cholesky_
                    self.mix_weights_ = mix_weights_
                    self.converged_ = True

        if not self.converged_:
            print("The model did not converge on any of the restarts! Try increasing max_iter or "
                        "tol or check data for possible issues.")


    def fitting_restart(self, X, random_state):
        """A single fitting restart.

        Args:
            X (np.ndarray): The raw data. Must be a 2d array where each column is a feature and
                each row is a datapoint. The caller (self.fit) ensures this is true.
            random_state (int): The seed for the random number generator.

        Returns:
            current_bound (float): The lower bound for the current fitting iteration.
                The caller (self.fit) keeps the set of parameters that have the best
                associated lower bound.
            convergence (bool): A boolean indicating convergence or lack thereof.
            loc_ (np.ndarray): The locations (analogous to means for a Gaussian) of the components.
                Shape is K x M for K components, M dimensions.
            scale_ (np.ndarray): The scale matrices (analogous to covariance for a Gaussian).
                Shape is M x M x K for M dimensions, K components.
            mix_weights_ (np.ndarray): The mixture weights. Shape is K for K components.
            df_ (np.ndarray): The degrees of freedom. Shape is K for K components.
            scale_cholesky_ (np.ndarray): The cholesky decomposition of the scale matrices.
                Shape is M x M x K for M dimensions, K components.
        """
        xp, xsp = get_array_module(self._use_gpu)

        # Initialization stays on CPU, then move to device
        df_ = np.full((self.n_components), self.start_df_, dtype=np.float64)
        loc_, scale_, mix_weights_, scale_cholesky_ = \
                self.initialize_params(X, random_state, self.init_type)

        if self._use_gpu:
            X_dev = to_device(X, xp)
            df_ = to_device(df_, xp)
            loc_ = to_device(loc_, xp)
            scale_ = to_device(scale_, xp)
            mix_weights_ = to_device(mix_weights_, xp)
            scale_cholesky_ = to_device(scale_cholesky_, xp)
        else:
            X_dev = X

        lower_bound, convergence = -np.inf, False

        for _ in range(self.max_iter):
            resp, E_gamma, current_bound = self.Estep(X_dev, df_, loc_,
                                scale_cholesky_, mix_weights_, None,
                                xp=xp, xsp=xsp)

            mix_weights_, loc_, scale_, scale_cholesky_, \
                            df_ = self.Mstep(X_dev, resp, E_gamma, scale_,
                                scale_cholesky_, df_, xp=xp)
            change = current_bound - lower_bound
            if abs(change) / max(abs(current_bound), 1.0) < self.tol:
                convergence = True
                break
            lower_bound = current_bound
            if self.verbose:
                print(f"Change in lower bound: {change}")
                print(f"Actual lower bound: {current_bound}")

        # Move results back to CPU numpy arrays
        if self._use_gpu:
            loc_ = to_numpy(loc_)
            scale_ = to_numpy(scale_)
            mix_weights_ = to_numpy(mix_weights_)
            df_ = to_numpy(df_)
            scale_cholesky_ = to_numpy(scale_cholesky_)

        return current_bound, convergence, loc_, scale_, \
                mix_weights_, df_, scale_cholesky_



    def Estep(self, X, df_, loc_, scale_cholesky_, mix_weights_,
            sq_maha_dist, xp=None, xsp=None):
        """Update the "hidden variables" in the mixture description.

        Args:
            X: The input data. Shape N x M.
            df_: The degrees of freedom. Shape K.
            loc_: The locations of the components. Shape K x D.
            scale_cholesky_: The cholesky decomposition of scale matrices.
            mix_weights_: The mixture weights. Shape K.
            sq_maha_dist: Unused (kept for API compat).
            xp: Array module (numpy or cupy).
            xsp: Scipy module (scipy or cupyx.scipy).

        Returns:
            resp: Responsibilities, shape N x K.
            E_gamma: Expected gamma values, shape N x K.
            lower_bound: Mean log-likelihood (float).
        """
        if xp is None:
            xp = np
        xsp_special = xsp.special if xsp is not None and hasattr(xsp, 'special') else None

        sq_maha_dist = sq_maha_distance(X, loc_, scale_cholesky_,
                covariance_type=self.covariance_type, xp=xp)

        loglik = self.get_loglikelihood(X, sq_maha_dist, df_,
                scale_cholesky_, mix_weights_, xp=xp, xsp_special=xsp_special)

        weighted_log_prob = loglik + xp.log(xp.clip(mix_weights_,
                                        a_min=1e-12, a_max=None))[xp.newaxis,:]
        if xsp is not None and hasattr(xsp, 'special'):
            _logsumexp = xsp.special.logsumexp
        else:
            _logsumexp = logsumexp
        log_prob_norm = _logsumexp(weighted_log_prob, axis=1)
        resp = xp.exp(weighted_log_prob - log_prob_norm[:, xp.newaxis])
        E_gamma = (df_[xp.newaxis,:] + X.shape[1]) / (df_[xp.newaxis,:] + sq_maha_dist)
        lower_bound = float(xp.mean(log_prob_norm))
        return resp, E_gamma, lower_bound



    def Mstep(self, X, resp, E_gamma, scale_, scale_cholesky_, df_, xp=None):
        """The M-step in mixture fitting. Updates the component parameters
        using the "hidden variable" values calculated in the E-step.

        Args:
            X: The input data. Shape N x M.
            resp: Responsibilities. Shape N x K.
            E_gamma: Expected gamma values. Shape N x K.
            scale_: Scale matrices. Shape M x M x K (full) or M x K (diag).
            scale_cholesky_: Cholesky decomposition of scale matrices.
            df_: Degrees of freedom. Shape K.
            xp: Array module (numpy or cupy).

        Returns:
            mix_weights_, loc_, scale_, scale_cholesky_, df_
        """
        if xp is None:
            xp = np
        mix_weights_ = xp.mean(resp, axis=0)
        ru = resp * E_gamma
        loc_ = xp.dot(ru.T, X)
        resp_sum = xp.sum(ru, axis=0) + 10 * np.finfo(np.float64).eps
        loc_ = loc_ / resp_sum[:,xp.newaxis]

        scale_, scale_cholesky_ = scale_update_calcs(X, ru, loc_,
                resp_sum, self.reg_covar, covariance_type=self.covariance_type,
                xp=xp)

        # Rescue empty components: reinitialize to the worst-fit datapoint.
        for k in range(self.n_components):
            if float(mix_weights_[k]) < 1e-8:
                worst_idx = int(xp.argmin(xp.max(resp, axis=1)))
                loc_[k] = X[worst_idx]
                if self.covariance_type == 'diag':
                    scale_[:,k] = xp.var(X, axis=0) + self.reg_covar
                    scale_cholesky_[:,k] = scale_[:,k]
                else:
                    # Covariance computation — use numpy on CPU for np.cov compat
                    X_cpu = to_numpy(X) if self._use_gpu else X
                    cov_matrix = xp.asarray(np.cov(X_cpu, rowvar=False))
                    scale_[:,:,k] = cov_matrix
                    scale_[:,:,k].flat[::X.shape[1]+1] += self.reg_covar
                    scale_cholesky_[:,:,k] = xp.linalg.cholesky(scale_[:,:,k])
                mix_weights_[k] = 1.0 / self.n_components
                mix_weights_ /= mix_weights_.sum()

        if not self.fixed_df:
            # DF optimization uses scipy special functions — run on CPU
            if self._use_gpu:
                df_cpu = self.optimize_df(to_numpy(X), to_numpy(resp),
                                         to_numpy(E_gamma), to_numpy(df_))
                df_ = to_device(df_cpu, xp)
            else:
                df_ = self.optimize_df(X, resp, E_gamma, df_)
        return mix_weights_, loc_, scale_, scale_cholesky_, df_



    def optimize_df(self, X, resp, E_gamma, df_):
        """Optimizes the df parameter using a vectorized Newton-Raphson over
        all K components simultaneously.

        Args:
            X (np.ndarray): The input data, a numpy array of shape N x M
                for N datapoints, M features.
            resp (np.ndarray): The responsibility of each cluster for each
                datapoint. An N x K numpy array for K components.
            E_gamma (np.ndarray): The ML estimate of the "hidden variable" described by
                a gamma distribution in the formulation of the student's t-distribution
                as a scale mixture of normals.
            df_ (np.ndarray): The current estimate of degrees of freedom.

        Returns:
            df_ (np.ndarray): The updated estimate for degrees of freedom.
        """
        df_x_dim = 0.5 * (df_ + X.shape[1])
        resp_sum = np.sum(resp, axis=0)
        ru_sum = np.sum(resp * (np.log(E_gamma) - E_gamma), axis=0)
        constant_term = 1.0 + (ru_sum / resp_sum) + digamma(df_x_dim) - \
                    np.log(df_x_dim)

        current_df = df_.copy()
        for _ in range(min(self.max_iter, 50)):
            clipped_df = np.clip(current_df, a_min=1e-9, a_max=None)
            half_df = 0.5 * clipped_df
            # First derivative (f) and second derivative (fp) for Newton step
            f = constant_term - digamma(half_df) + np.log(half_df)
            fp = -0.5 * polygamma(1, half_df) + 1.0 / clipped_df
            step = f / fp
            current_df = current_df - step
            if np.all(np.abs(step) < 1e-3):
                break

        # Keep old values where Newton diverged (NaN)
        mask = np.isnan(current_df)
        current_df[mask] = df_[mask]
        # DF should never be less than 1 but can go arbitrarily high.
        current_df = np.clip(current_df, 1.0, None)
        return current_df


    def dof_first_deriv(self, dof, constant_term):
        """First derivative of the complete data log likelihood w/r/t df. This is used to
        optimize the input value (dof) via the Newton-Raphson algorithm using the scipy.optimize.
        newton function (see self.optimize_df).
        """
        clipped_dof = np.clip(dof, a_min=1e-9, a_max=None)
        return constant_term - digamma(dof * 0.5) + np.log(0.5 * clipped_dof)

    def dof_second_deriv(self, dof, constant_term):
        """Second derivative of the complete data log likelihood w/r/t df. This is used to
        optimize the input value (dof) via the Newton-Raphson algorithm using the
        scipy.optimize.newton function (see self.optimize_df).
        """
        return -0.5 * polygamma(1, 0.5 * dof) + 1 / dof

    def dof_third_deriv(self, dof, constant_term):
        """Third derivative of the complete data log likelihood w/r/t df. This is used to optimize
        the input value (dof) via the Newton-Raphson algorithm using the scipy.optimize.newton
        function (see self.optimize_df).
        """
        return -0.25 * polygamma(2, 0.5 * dof) - 1 / (dof**2)




    def initialize_params(self, X, random_seed, init_type):
        """Initializes the model parameters. Two approaches are available for initializing
        the location (analogous to mean of a Gaussian): k++ and kmeans. THe scale is
        always initialized using a broad value (the covariance of the full dataset + reg_covar)
        for all components, while mixture_weights are always initialized to be equal for all
        components, and df is set to the starting value. All initialized parameters are 
        returned to caller.

        Args:
            X (np.ndarray): The input data; a 2d numpy array of shape N x M
                for N datapoints, M features.
            random_seed (int): The random seed for the random number generator.
            init_type (str): One of 'kmeans', 'k++'. 'kmeans' is better but slightly
                slower.

        Returns:
            loc_ (np.ndarray): The initial cluster centers which will be optimized
                during fitting. This is a numpy array of shape K x M for K components,
                M features.
            scale_ (np.ndarray): Initial scale matrices (analogous to covariance of
                a Gaussian). This is a numpy array of shape M x M x K for M features,
                K components. These are initially set to a very broad default.
            mix_weights_ (np.ndarray): The mixture weights. This is a numpy array
                of shape K for K components.
            scale_cholesky_ (np.ndarray): The cholesky decomposition of the scale
                matrices; same shape as scale_.
        """
        labels = None
        if init_type == "kmeans":
            loc_, labels = self.kmeans_initialization(X, random_seed)
        else:
            loc_ = self.kplusplus_initialization(X, random_seed)

        mix_weights_ = np.empty(self.n_components)
        mix_weights_.fill(1/self.n_components)

        #Set all scale matrices to a broad default -- the covariance of the data.
        default_scale_matrix = np.cov(X, rowvar=False)
        default_scale_matrix.flat[::X.shape[1] + 1] += self.reg_covar

        #For 1-d data, ensure default scale matrix has correct shape.
        if len(default_scale_matrix.shape) < 2:
            default_scale_matrix = default_scale_matrix.reshape(-1,1)

        if self.covariance_type == 'diag':
            # For diagonal covariance, use per-cluster variance from kmeans
            # assignments when available, falling back to global covariance.
            default_diag = np.diag(default_scale_matrix).copy()
            scale_ = np.empty((X.shape[1], self.n_components))
            if labels is not None:
                for k in range(self.n_components):
                    mask = labels == k
                    if np.sum(mask) > 1:
                        scale_[:,k] = np.var(X[mask], axis=0) + self.reg_covar
                    else:
                        scale_[:,k] = default_diag
            else:
                for k in range(self.n_components):
                    scale_[:,k] = default_diag
            # this is not really a cholesky decomposition but M x K (diagonal variances)
            # named like this for simplicity of return
            scale_cholesky_ = scale_
        else:
            scale_ = np.stack([default_scale_matrix for i in range(self.n_components)],
                            axis=-1)
            scale_cholesky_ = [np.linalg.cholesky(scale_[:,:,i]) for i in range(self.n_components)]
            scale_cholesky_ = np.stack(scale_cholesky_, axis=-1)
        return loc_, scale_, mix_weights_, scale_cholesky_



    def kplusplus_initialization(self, X, random_seed):
        """The first option for initializing loc_ is k++, a modified version of the
        kmeans++ algorithm.
        On each iteration until we have reached n_components,
        a new cluster center is chosen randomly with a probability for each datapoint
        inversely proportional to its smallest distance to any existing cluster center.
        This tends to ensure that starting cluster centers are widely spread out.

        Args:
            X (np.ndarray): The input training data.
            random_seed (int): The random seed for the random number generator.

        Returns:
            loc_ (np.ndarray): The selected cluster centers.
        """
        np.random.seed(random_seed)
        loc_ = [X[np.random.randint(0, X.shape[0]-1), :]]
        dist_arr_list = []
        for i in range(1, self.n_components):
            dist_arr = np.sum((X - loc_[i-1])**2, axis=1)
            dist_arr_list.append(dist_arr)
            distmat = np.stack(dist_arr_list, axis=-1)
            min_dist = np.min(distmat, axis=1)
            min_dist = min_dist / np.sum(min_dist)
            next_center_id = np.random.choice(distmat.shape[0], size=1, p=min_dist)
            loc_.append(X[next_center_id[0],:])
        return np.stack(loc_)



    def kmeans_initialization(self, X, random_state):
        """The second alternative for initializing loc_is kmeans clustering. It takes as
        input the data X (NxD for N datapoints, D dimensions) and as a starting point
        loc_ returned by self.kplusplus_initialization (KxD for K components, D dimensions).
        Starting with version 0.0.2.2, we are using sklearn's KMeans for clustering --
        it's simpler than rolling our own and theirs performs very well.

        Args:
            X (np.ndarray): The input data.
            random_state (int): The random seed for the random number generator.

        Returns:
            km.cluster_centers_ (np.ndarray): A numpy array of shape (n_components,)
                with the center of each cluster; these will be refined during fitting.
            km.labels_ (np.ndarray): A numpy array of shape (N,) with the cluster
                assignment for each datapoint.
        """
        km = KMeans(n_clusters = self.n_components, n_init=3,
                random_state = random_state).fit(X)
        return km.cluster_centers_, km.labels_


    #The remaining functions are called only for trained models. They are kept separate from the base
    #class because the variational approach does not require BIC or AIC calculations; 
    #these are EM-specific.


    def aic(self, X):
        """Returns the Akaike information criterion (AIC) for the input dataset.
        Useful in selecting the number of components. AIC places heavier
        weight on model performance than BIC and less weight on penalizing
        a large number of parameters.

        Args:
            X (np.ndarray): The input data, a 2d numpy array with N datapoints, 
                M dimensions.

        Returns:
            aic (float): The Akaike information criterion for the data. Lower is
                better.
        """
        self.check_model()
        x = self.check_inputs(X)
        n_params = self.get_num_parameters()
        score = self.score(x, perform_model_checks = False)
        return 2 * n_params - 2 * score * X.shape[0]


    def bic(self, X):
        """Returns the Bayes information criterion (BIC) for the input dataset.
        Useful in selecting the number of components, more heavily penalizes
        n_components than AIC.

        Args:
            X (np.ndarray): The input data, a 2d numpy array with N datapoints, 
                M dimensions.

        Returns:
            bic (float): The Bayes information criterion for the data. Lower is
                better.
        """
        self.check_model()
        x = self.check_inputs(X)
        score = self.score(x, perform_model_checks = False)
        n_params = self.get_num_parameters()
        return n_params * np.log(x.shape[0]) - 2 * score * x.shape[0]
