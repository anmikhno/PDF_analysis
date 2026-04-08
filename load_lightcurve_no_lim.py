from cProfile import label
import os

import numpy as np
from scipy.stats import norm, lognorm
from scipy.integrate import trapezoid
from astropy.table import Table
import matplotlib
matplotlib.use('Agg')   # non-interactive, much faster
import matplotlib.pyplot as plt
from iminuit import Minuit
from scipy.optimize._numdiff import approx_derivative
from scipy.interpolate import UnivariateSpline
from scipy.stats import gamma
from scipy.stats import kstest
from matplotlib.backends.backend_pdf import PdfPages
import argparse
from scipy.stats import levy_stable
from multiprocessing import Pool,  cpu_count
from astropy.io import ascii
import pandas as pd

parser = argparse.ArgumentParser(
    prog = "load_lightcurve.py",
    description = "compares PDF models based on the fit and MC simulations, per selected source and cadence"
)

parser.add_argument(
    "-p", help = "path to the simulation", type = str,
)

parser.add_argument(
    "-s", help = "name of the source", type = str,
)

parser.add_argument(
    "-n_sim", help = "number of simulations for MC", type = int,
)

parser.add_argument(
    "-fig_path", help = "path to where the directory with the results is created", type = str,
)

args = parser.parse_args()

def load_lightcurve(filename):
    data = Table.read(filename, format="ascii.ecsv")
    t = data["tstart"]+data["lvtm"]*0.5
    phi = data["integrated_flux"]
    ephi = data["integrated_flux_error"]

    return t, phi, ephi


def plot_lightcurve(title, t, phi, ephi, path):
    exposure = phi / (ephi ** 2)

    # Transform for visualization: log10 of absolute values
    exposure_log = np.log10(np.abs(exposure))

    # --- Compute 5-sigma outliers using quantiles ---
    upper_sigma = np.mean(exposure_log)+5*np.std(exposure_log)
    outliers_mask = (exposure_log > upper_sigma)

    # --- Lightcurve ---
    fig = plt.figure(figsize=(15, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[4, 2], wspace=0.3)
    ax = fig.add_subplot(gs[0])
    ax_hist = fig.add_subplot(gs[1])

    ax.set_title(title)
    ax.set_xlabel(r"Time [h]", labelpad=12, fontsize=12)
    ax.set_ylabel(r"Energy flux [TeV cm$^{-2}$ s$^{-1}$]", labelpad=12, fontsize=12)
    ax.set_yscale('log')

    # Color points: red if outlier, black otherwise
    normal_mask = ~outliers_mask
    ax.errorbar(t[normal_mask], phi[normal_mask], yerr=ephi[normal_mask], fmt='o', mfc='none', color='k', alpha=0.7)
    ax.errorbar(t[outliers_mask], phi[outliers_mask], yerr=ephi[outliers_mask], fmt='o', mfc='none', color='red',
                alpha=0.9)

    # Histogram with proper bins
    bins = np.linspace(np.min(exposure_log), np.max(exposure_log), 30)
    counts, bin_edges = np.histogram(exposure_log, bins=bins)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Color bins red if they contain any outliers
    bin_colors = []
    for right in zip( bin_edges[1:]):
        mask =  (exposure_log < right)
        if np.any(outliers_mask[mask]):
            bin_colors.append('red')
        else:
            bin_colors.append('gray')

    ax_hist.bar(bin_centers, counts, width=bin_edges[1] - bin_edges[0], color=bin_colors, alpha=0.7)
    ax_hist.set_xlabel("log10(|Exposure|)")
    ax_hist.set_ylabel("Counts")
    ax_hist.legend(handles=[
        plt.Line2D([0], [0], color='gray', lw=6, label='Normal'),
        plt.Line2D([0], [0], color='red', lw=6, label='Outliers')
    ], fontsize=10)


    plt.savefig(f"{path}/Lightcurve.pdf", dpi=150)
    plt.close(fig)
    return outliers_mask

def hist_flux(phi, ephi, rebin):
    #load the histogram
    phi_min, phi_max = np.min(phi-ephi), np.max(phi+ephi)
    phi_min = max(phi_min, 0.)
    bin_size_log10 = np.median( ephi/(phi*np.log(10)) )*rebin
    phi_min = np.min(phi[phi > 0])
    log10_phi_min, log10_phi_max = np.log10(phi_min), np.log10(phi_max) 
    bins = np.arange(log10_phi_min, log10_phi_max+bin_size_log10, bin_size_log10)
    hist, bin_edges = np.histogram(np.log10(phi), bins=bins)

    return hist, bin_edges

def plot_histo(title, phi, ephi, pdf, rebin=3):
    hist, bins = hist_flux(phi, ephi, rebin)

    #plot
    fig, ax = plt.subplots(figsize=(8, 6), nrows=1, ncols = 1)
    plt.subplots_adjust(bottom = 0.1, top = 0.95, left=0.1, right=0.96, hspace=0.05)
    ax.set_title(title)
    ax.set_xlabel(r"log$_{10}$(energy flux [TeV cm$^{-2}$ s$^{-1}$])", ha="center", va = 'center', labelpad=12, fontsize=12)
    ax.set_ylabel(r"# entries", ha="center", va = 'center', labelpad=12, fontsize=12)
    #ax.set_yscale('log')
    log_phi = np.log10(phi)
    ax.stairs(hist,bins)
    bin_centers = 0.5*(bins[1:]+bins[:-1])
    sel = hist>=1
    ax.errorbar(bin_centers[sel], hist[sel], yerr = np.sqrt(hist[sel]), fmt='',  mfc='none', color='tab:blue', ls='none')

    bin_size_log10 = bins[1]-bins[0]
    log10phi_plot = np.linspace(np.min(log_phi), np.max(log_phi))
    phi_plot = np.power(10, log10phi_plot)
    normalization_pdf = phi.size*bin_size_log10*np.log(10)
    ax.plot(log10phi_plot, phi_plot*pdf(phi_plot)*normalization_pdf)
    plt.show()

def pdf_Gauss(x, mu, sigma):
    return norm(loc=mu, scale=sigma).pdf(x)

def pdf_LogNorm(x, mu, sigma):
    rv = lognorm(s=sigma, scale=np.exp(mu))
    return rv.pdf(x)

def pdf_gammaf(x, a, b):
    return gamma.pdf(x, a, scale=1/b)


def pdf_alpha_stable(x, loc, scale, alpha, beta=1.0):
    return levy_stable.pdf(x, alpha, beta, loc=loc, scale=scale)

MODELS = {
    "lognorm": {
        "pdf": lambda x, m, s: pdf_LogNorm(x, m, s),
        "names": ["mu", "sigma"]
    },
    "gaussian": {
        "pdf": lambda x, m, s: pdf_Gauss(x, m, s),
        "names": ["mean", "std"]
    },
    "alpha": {
    "pdf": lambda x, alpha, loc, scale: pdf_alpha_stable(x, loc, scale, alpha),
    "names": ["alpha", "loc", "scale"]
}

}


def summarize_fit(minuit):
    fmin = minuit.fmin

    print("\n=== FIT SUMMARY ===")
    print(f"Converged (is_valid)   : {fmin.is_valid}")
    print(f"Has covariance         : {minuit.covariance is not None}")
    print(f"EDM                    : {fmin.edm}")
    print(f"EDM goal               : {minuit.tol}")
    print(f"FCN                    : {fmin.fval}")
    print(f"Nfcn                   : {fmin.nfcn}")

    print(f"Has reached call limit : {fmin.has_reached_call_limit}")
    print(f"Hesse ok               : {fmin.has_accurate_covar}")
    print(f"Posdef covariance      : {fmin.has_posdef_covar}")

def parameter_table(minuit):
    data = []

    for name in minuit.parameters:
        val = minuit.values[name]
        err = minuit.errors[name]

        data.append({
            "Parameter": name,
            "Value": val,
            "Error": err
        })

    df = pd.DataFrame(data)
    print("\n=== PARAMETERS ===")
    print(df)

    return df


def minos_table(minuit):
    minuit.minos()

    data = []
    for name in minuit.parameters:
        m = minuit.merrors.get(name, None)
        if m is None:
            continue

        data.append({
            "Parameter": name,
            "Lower": m.lower,
            "Upper": m.upper,
        })

    df = pd.DataFrame(data)
    print("\n=== MINOS ERRORS ===")
    print(df)

    return df



def likelihood_scan(minuit, pdf=None):
    """Draws the MN matrix; saves to pdf if provided, otherwise shows."""
    minuit.draw_mnmatrix()
    fig = plt.gcf()
    if pdf is not None:
        pdf.savefig(fig)
        plt.close(fig)
    else:
        plt.show()


def fitting(phi_i, ephi_i, init_mean, init_std,
            model="lognorm", pdf=None,
            do_plot=False, collect_stats=False, return_full=False,
            skip_fit=False, fixed_params=None, init_alpha=None):
    """
        Fits selected model on the data, giving best fit parameters and corresponding errors.
        Can be improved as the Gaussian fit is slow.
    """
    nsigma = 5
    if model not in MODELS:
        raise ValueError(f"Model '{model}' not supported. Choose from: {list(MODELS.keys())}")

    pdf_model = MODELS[model]["pdf"]
    param_names = MODELS[model]["names"]

    #COME BACK HERE
    if model == "alpha" and init_alpha is None:
        init_alpha = estimate_alpha_tail(phi_i)

    if model == "gaussian":
        def minimize_func(init_mean, init_std):
            var = init_std ** 2 + ephi_i ** 2
            logL_i = -0.5 * (np.log(2 * np.pi * var) + (phi_i - init_mean) ** 2 / var)

            return -np.sum(logL_i)

    if model == "alpha":
        # Compute a scale factor from the data once
        flux_scale = np.median(phi_i)  # e.g. ~1e-10

        def minimize_func(alpha, loc, scale):
            if alpha <= 1.05 or alpha > 2.0:
                return 1e10
            if scale <= 0 or loc < 0:
                return 1e10
            try:
                logpdf_vals = levy_stable.logpdf(phi_i, alpha, 1.0, loc=loc, scale=scale)
                logpdf_vals = np.clip(logpdf_vals, -1e300, None)
                if not np.isfinite(logpdf_vals).all():
                    return 1e10
                return -np.sum(logpdf_vals)
            except Exception:
                return 1e10

    else:
        def minimize_func(init_mean, init_std):
            # define the function to be integrated
            def integ(pdf_test, phi_0, ephi_0):
                # range for integration over the data uncertainties
                phi_int_min = np.min(phi_0 - nsigma * ephi_0)
                phi_int_max = np.max(phi_0 + nsigma * ephi_0)
                phi_integrand = np.linspace(phi_int_min, phi_int_max, 500)

                # integrand
                rv = norm(loc=phi_0, scale=ephi_0)
                integrand = rv.pdf(phi_integrand) *  pdf_test(phi_integrand, init_mean, init_std)
                integral_val = trapezoid(integrand, x=phi_integrand)
                integral_val = max(integral_val, 1e-300)
                return integral_val


            likelihood = np.array([integ(pdf_model, phi_i[i], ephi_i[i]) for i in range(phi_i.size)])


            return -np.sum(np.log(likelihood))

    if model == "alpha":
        loc0 = init_mean
        scale0 = init_std
        alpha0 = init_alpha  # from estimator


        minuit_obj = Minuit(minimize_func, alpha0, loc0, scale0, name=param_names)
       # minuit_obj.limits["alpha"] = (1.1, 2.0)
        #minuit_obj.limits["scale"] = (0, 100000 * np.max(phi_i))
        # loc must be <= min(phi), since beta=1 means support is [loc, inf)
        #minuit_obj.limits["loc"] = (1e-2 * np.min(phi_i), 10 * np.max(phi_i))
        minuit_obj.values["alpha"] = init_alpha
        minuit_obj.fixed["alpha"] = True
        # minuit_obj.limits["scale"] = (1e-12, None)

        max_phi, min_phi = np.max(phi_i), np.min(phi_i)
        amp = np.abs(max_phi - min_phi)
        #minuit_obj.limits["loc"] = (-10000 * amp, 10000 * amp)
        #minuit_obj.limits["loc"] = (0, np.min(phi_i))


    else:
        minuit_obj = Minuit(minimize_func, init_mean, init_std, name=param_names)

    if not skip_fit:
        minuit_obj.simplex(ncall=2000)  # gradient-free, finds basin
        minuit_obj.migrad(ncall=5000)

        # retry with simplex seed if migrad failed
        if not minuit_obj.fmin.is_valid:
            minuit_obj.simplex(ncall=3000)
            minuit_obj.migrad(ncall=5000)
        if model == "alpha" and minuit_obj.fmin.is_valid:
            minuit_obj.hesse()
    else:
        if fixed_params is None:
            raise ValueError("fixed_params must be provided when skip_fit=True")

        # assign fitted values directly
        for name, val in zip(minuit_obj.parameters, fixed_params):
            minuit_obj.values[name] = val

        # important: fake a "converged" state
        converged = True

    if not skip_fit:
        converged = minuit_obj.fmin.is_valid
    else:
        converged = True

    for name in minuit_obj.parameters:
        val = minuit_obj.values[name]
        low, high = minuit_obj.limits[name]

        if low is not None and np.isclose(val, low, rtol=1e-3):
            print(f"Parameter {name} is at LOWER limit!")

        if high is not None and np.isclose(val, high, rtol=1e-3):
            print(f"Parameter {name} is at UPPER limit!")

    if skip_fit:
        edm = np.nan
        nfcn = 0
        converged = 0
        fval =0
        ngrad = np.nan
    else:
        edm = minuit_obj.fmin.edm
        nfcn = minuit_obj.fmin.nfcn
        converged = minuit_obj.fmin.is_valid
        fval =  minuit_obj.fmin.fval
        ngrad =  minuit_obj.fmin.ngrad

    stats = {
        "converged": converged,
        "nfcn": nfcn,  # number of function calls
        "niter":ngrad,  # gradient evaluations
        "edm": edm,  # estimated distance to minimum
        "fval": fval,  # function value at minimum
        **{f"val_{n}": minuit_obj.values[n] for n in minuit_obj.parameters},
        **{f"err_{n}": minuit_obj.errors[n] for n in minuit_obj.parameters},
    }

    if collect_stats:
        print("Collecting stats")
        print(stats)
        #return stats

    if not converged:
        print("Fit did not converge!")
    if not skip_fit:
        summarize_fit(minuit_obj)

    param_df = parameter_table(minuit_obj)
    minos_df = minos_table(minuit_obj) if (not skip_fit and converged) else None
    #  scan
    if do_plot:
        n_params = len(minuit_obj.parameters)
        fig, axes = plt.subplots(1, n_params, figsize=(5 * n_params, 4))
        if n_params == 1:
            axes = [axes]

        for ax, p in zip(axes, minuit_obj.parameters):
            # size=30 instead of default 100 — 3× faster, still informative
            x, y, ok = minuit_obj.mnprofile(p, size=30)
            ax.plot(x, y, lw=1.8)
            ax.axvline(minuit_obj.values[p], color="red",
                       linestyle="--", lw=1.2, label="best fit")
            ax.set_xlabel(p)
            ax.set_ylabel("−2 ln L")
            ax.set_title(f"Profile: {p}")
            ax.legend(fontsize=8)

        if skip_fit:
            plt.suptitle(f"model={model}  |  skip_fit=True  |  converged={converged}", fontsize=9)
        else:
            plt.suptitle(
                f"model={model}  |  converged={converged}"
                f"  |  nfcn={minuit_obj.fmin.nfcn}"
                f"  |  EDM={minuit_obj.fmin.edm:.2e}"
                f"  |  niter={minuit_obj.fmin.ngrad}"
                f"  |  fval={minuit_obj.fmin.fval}",
                fontsize=9
            )
        plt.tight_layout()
        if pdf is not None:
            pdf.savefig(fig)
            plt.close(fig)
        else:
            plt.show()

    params = np.array([minuit_obj.values[name] for name in minuit_obj.parameters])
    cov = np.array(minuit_obj.covariance)

    if skip_fit:
        yerr_prop = tuple(np.nan for _ in params)

    if model == "gaussian":
        dist = norm(loc=params[0], scale=params[1])
    elif model == "lognorm":
        dist = lognorm(s=params[1], scale=np.exp(params[0]))
    elif model == "alpha":
        dist = levy_stable(alpha=params[0], beta=1.0, loc=params[1], scale=params[2])

    best_params = [minuit_obj.values[name] for name in param_names]
    errors = [minuit_obj.errors[name] for name in param_names]
    logL = -minimize_func(*best_params)

    if model == "alpha":
        return (*best_params, np.nan, np.nan, logL, converged)



    basic_result = (
        *best_params,
        *errors,
        logL,
        converged,
    )

    if not return_full:
        return basic_result

    #extended diagnostics

    full_result = {
        "basic": basic_result,
        "params": [minuit_obj.values[name] for name in param_names],
        "errors": [minuit_obj.errors[name] for name in param_names],
        "logL": logL,
        "converged": converged,
        "is_valid": minuit_obj.fmin.is_valid,
        "has_valid_parameters": minuit_obj.fmin.has_valid_parameters,
        "has_made_posdef_covar": minuit_obj.fmin.has_made_posdef_covar,
        "hesse_failed": minuit_obj.fmin.hesse_failed,
        "has_covariance": minuit_obj.fmin.has_covariance,
        "edm": minuit_obj.fmin.edm,
        "nfcn": minuit_obj.fmin.nfcn,
        "has_call_limit": minuit_obj.fmin.has_reached_call_limit,
        "has_valid_covar": minuit_obj.fmin.has_accurate_covar,
        "at_limit": {
            name: minuit_obj.at_limit(name)
            for name in param_names
        },
        "fixed": {
            name: minuit_obj.fixed[name]
            for name in param_names
        },
        "limits": {
            name: minuit_obj.limits[name]
            for name in param_names
        },
        "minos_errors": minuit_obj.merrors if hasattr(minuit_obj, "merrors") else None,

    }
    print(full_result)

    return full_result




def fit_error_trend(flux, flux_err):
    """
        sigma_f^2 = C * flux
    Parameters
    ----------
    flux : array-like
        Observed flux values f_i.
    flux_err : array-like
        Observed errors sigma_{f_i}.

    Returns
    -------
    C : float
        Error scaling constant (1/(A*T)).
    """
    f = np.asarray(flux)
    s = np.asarray(flux_err)
    mask = f > 0
    f = f[mask]
    s = s[mask]
    C_values = (s**2) / f

    # Physical estimator: use the median
    C = np.median(C_values)

    return C


def simulate_distribution(flux, flux_err, flux_distribution_params, n_sim, model="gaussian", C=None):
    """
    Simulate n_sim light curves with flux-dependent errors following:

        sigma_f^2 = C * flux

    Parameters
    ----------
    n_sim : int
        Number of simulations (lightcurve realizations).
    flux_distribution_params
    model : str
        "gauss" or "lognorm".
    C : float
        Error scaling constant from _fit_error_trend().
        Required.

    Returns
    -------
    simulations : list of dict
        Each dict has:
            {"flux": array, "flux_err": array}
    """
    flux_distribution_params_dict = {"mean": flux_distribution_params[0], "std": flux_distribution_params[1]}
    if C is None:
        C = fit_error_trend(flux, flux_err)

    simulations = []

    for _ in range(n_sim):

        # Generate TRUE fluxes (one per data point)
        n = len(flux)

        if model == "gaussian":
            mu = flux_distribution_params_dict['mean']
            sig = flux_distribution_params_dict['std']
            f_true = norm.rvs(loc=mu, scale=sig, size=n)

        elif model == "lognorm":
            mu = flux_distribution_params_dict['mean']
            sigma = flux_distribution_params_dict['std']
            f_true = lognorm.rvs(s=sigma, scale=np.exp(mu), size=n)
        elif model == "gamma":
            a = flux_distribution_params_dict['a']
            b = flux_distribution_params_dict['b']
            f_true = gamma.rvs(a = a, scale = 1/b, size =n)
        elif model == "alpha":
            print(flux_distribution_params)
            alpha = flux_distribution_params[0]
            loc = flux_distribution_params[1]
            scale = flux_distribution_params[2]
            f_true = levy_stable.rvs(loc = loc, scale=scale, alpha=alpha, beta=1.0, size = n)



        else:
            raise ValueError("model must be 'gauss' or 'lognorm'.")

        # No negative flux
        f_true = np.clip(f_true, 0, None)

        #Compute physical measurement errors
        #          sigma_i^2 = C * f_true_i
        sigma = np.sqrt(C * f_true)

        # prevent sigma = 0 (causes likelihood collapse)
        sigma = np.clip(sigma, 1e-6, None)
        # Generate measured fluxes
        f_meas = np.random.normal(f_true, sigma)
        simulations.append((f_meas, sigma))

    return simulations

def plot_likelihood_distribution(
        phi, ephi, t,
        fit_params,
        path,
        model="gaussian",
        n_sim=100,
        title="Likelihood comparison"
):
    """
    Plot histogram of log-likelihoods from MC simulations and compare to real data.

    Parameters
    ----------
    phi, ephi : arrays
        Real flux and flux errors.
    fit_params : tuple
        Output of fitting() on real data:
        (mu, sigma, mu_err, sigma_err, logL_real)
    model : str
        "gaussian" or "lognorm"
    n_sim : int
        Number of MC simulations
    t : array
        Time array (only needed to get matching size)
    """

    print(f"\n### Likelihood comparison for model = {model} ###")

    if model == "alpha":
        alpha_real, mu_real, sigma_real = fit_params[0], fit_params[1], fit_params[2]
    else:
        mu_real, sigma_real = fit_params[0], fit_params[1]
    logL_real = fit_params[-2]

    # -------------------------------------------------
    # 1) Generate simulations using best-fit parameters
    # -------------------------------------------------

    print(f"Generating {n_sim} simulations...")

    phi_med = np.median(phi)
    phi_norm = phi / phi_med
    ephi_norm = ephi / phi_med
    C = fit_error_trend(phi_norm, ephi_norm)
    sims = simulate_distribution(phi_norm, ephi_norm, fit_params, n_sim=n_sim, model=model, C=C)
    # 2) Fit each simulation and collect TS (logL)
    print("Fitting simulations in parallel...")

    n_cores = cpu_count()
    args_list = [
        (phi_sim, ephi_sim, mu_real, sigma_real, model)
        for phi_sim, ephi_sim in sims
    ]


    with Pool(n_cores) as pool:
        results = pool.map(fit_single_sim, args_list, chunksize=10)

    # debug_indices = np.random.choice(len(sims), size=min(3, len(sims)), replace=False)
    # print("\n--- Debugging selected simulations ---")

    debug_pdf_path = f"{path}/debug_simulations_{model}.pdf"

    with PdfPages(debug_pdf_path) as debug_pdf:
        for (phi_sim, ephi_sim), result in zip(sims, results):
            if not isinstance(result, dict):
                continue

            fitting(
                phi_sim,
                ephi_sim,
                mu_real,
                sigma_real,
                model=model,
                pdf=debug_pdf,
                do_plot=True,
                skip_fit=True,
                fixed_params=result["params"],
            )

    TS_sim = np.array([r["logL"] for r in results if isinstance(r, dict) and np.isfinite(r["logL"])])
    converged_flags = np.array([r["converged"] if isinstance(r, dict) else False for r in results])
    n_failed = np.sum(~converged_flags)
    print(f"Fits that did not converge: {n_failed}/{len(converged_flags)}")

    #plot_likelihood_distribution

    # 3) Compute ΔTS and significance

    mean_sim, std_sim = norm.fit(TS_sim)


    delta_TS = logL_real - mean_sim
    significance = delta_TS / std_sim if std_sim > 0 else np.nan
    z_score = delta_TS / std_sim
    print(f"Z-score (signed) = {z_score:.2f}")
    physical_sigma = np.abs(z_score)
    print(f"Significance level = {physical_sigma:.2f} σ")

    print(f"\nReal-data logL = {logL_real:.3f}")
    print(f"Sim mean logL  = {mean_sim:.3f}")
    print(f"Sim std logL   = {std_sim:.3f}")
    print(f"ΔTS = {delta_TS:.3f}")
    print(f"Significance = {significance:.2f} σ")
    TS_obs = logL_real
    p_value = np.mean(TS_sim >= TS_obs)
    print(f"p-value = {p_value}")

    # --- NEW: diagnostic scatter of fitted sim parameters ---
    sim_means = [r["params"][0] for r in results if isinstance(r, dict)]
    sim_stds = [r["params"][1] for r in results if isinstance(r, dict)]

    fig2, axes = plt.subplots(1, 3, figsize=(15, 4))

    # param scatter
    axes[0].scatter(sim_means, sim_stds, alpha=0.6, label="Simulations")
    axes[0].axvline(mu_real, color="red", linestyle="--", label=f"Real mean={mu_real:.3f}")
    axes[0].axhline(sigma_real, color="blue", linestyle="--", label=f"Real std={sigma_real:.3f}")
    axes[0].set_xlabel("Fitted mean")
    axes[0].set_ylabel("Fitted std")
    axes[0].set_title("Sim fitted parameters vs real")
    axes[0].legend(fontsize=8)

    # mean distribution
    axes[1].hist(sim_means, bins=20, alpha=0.7, color="tab:orange")
    axes[1].axvline(mu_real, color="red", linestyle="--", label=f"Real={mu_real:.3f}")
    axes[1].set_xlabel("Fitted mean")
    axes[1].set_title("Distribution of sim means")
    axes[1].legend(fontsize=8)

    # std distribution
    axes[2].hist(sim_stds, bins=20, alpha=0.7, color="tab:green")
    axes[2].axvline(sigma_real, color="red", linestyle="--", label=f"Real={sigma_real:.3f}")
    axes[2].set_xlabel("Fitted std")
    axes[2].set_title("Distribution of sim stds")
    axes[2].legend(fontsize=8)

    plt.suptitle(f"Simulation parameter diagnostics | model={model} | n_sim={n_sim}", fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{path}/sim_param_diagnostics_{model}.pdf", dpi=150)
    plt.close(fig2)

    # --- NEW: print parameter summary table ---
    print("\n=== SIMULATION PARAMETER SUMMARY ===")
    print(f"{'':>10} {'mean_real':>12} {'sim_mean':>12} {'sim_std':>12} {'sim_min':>12} {'sim_max':>12}")
    for pname, preal, psim in zip(["mean", "std"], [mu_real, sigma_real], [sim_means, sim_stds]):
        print(f"{pname:>10} {preal:>12.4f} {np.mean(psim):>12.4f} "
              f"{np.std(psim):>12.4f} {np.min(psim):>12.4f} {np.max(psim):>12.4f}")
    # -------------------------------------------------
    # 4) Make the plot
    # -------------------------------------------------

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title(title + f" ({model})")
    ax.set_xlabel("Log-likelihood")
    ax.set_ylabel("Counts")

    # Histogram of sim log-likelihoods
    ax.hist(TS_sim, bins=20, alpha=0.7, color="tab:blue", label="Simulated TS")

    ax.plot(TS_sim, np.zeros_like(TS_sim) - 0.5, '|', color='black', alpha=0.5, markersize=10)

    # convergence info
    n_converged = np.sum(converged_flags)
    ax.text(
        0.05, 0.75,
        f"Converged: {n_converged}/{len(converged_flags)}\n"
        f"p-value: {p_value:.4f}",
        transform=ax.transAxes,
        va="top", ha="left", fontsize=10,
        bbox=dict(facecolor="lightyellow", alpha=0.8)
    )


    x_vals = np.linspace(min(TS_sim), max(TS_sim), 300)
    pdf_vals = norm.pdf(x_vals, mean_sim, std_sim)

    hist_counts, bin_edges = np.histogram(TS_sim, bins=20)
    scale = max(hist_counts) / max(pdf_vals)
    pdf_vals *= scale

    ax.plot(
        x_vals, pdf_vals,
        color="black", linewidth=2,
        label=f"Gauss fit: μ={mean_sim:.2f}, σ={std_sim:.2f}"
    )
    # Vertical lines
    ax.axvline(mean_sim, color="blue", linestyle="--",
               label=f"Sim mean = {mean_sim:.2f}")

    ax.axvline(logL_real, color="red", linestyle="-",
               label=f"CTAagnVar LC = {logL_real:.2f}")

    ax.text(
        0.05, 0.95,
        fr"$\Delta TS = {delta_TS:.2f}$" + "\n" +
        fr"${abs(delta_TS/std_sim):.2f}\sigma$",
        transform=ax.transAxes,
        va="top", ha="left", fontsize=12,
        bbox=dict(facecolor="white", alpha=0.8)
    )

    plt.legend(loc = 'lower right')
    plt.grid(alpha=0.3)
    #plt.show()
    plt.savefig(f"{path}/Likelihood_distribution_{model}.pdf", dpi=150)


# 5) Save results to file

    outfile = f"{path}/likelihood_results_{model}.txt"
    with open(outfile, "w") as f:
        f.write(f"Model: {model}\n")
        f.write(f"Real logL: {logL_real}\n")
        f.write(f"Mean sim logL: {mean_sim}\n")
        f.write(f"Std sim logL: {std_sim}\n")
        f.write(f"Delta TS: {delta_TS}\n")
        f.write(f"Significance: {significance} sigma\n")
        f.write("TS_sim array:\n")
        np.savetxt(f, TS_sim)


    return TS_sim, delta_TS, significance

def fit_single_sim(args):
    try:
        phi_sim, ephi_sim, mu_real, sigma_real, model = args

        # Remove the normalization here — data already normalized before simulation
        params_sim = fitting(
            phi_sim,       # already normalized
            ephi_sim,      # already normalized
            mu_real,
            sigma_real,
            model=model,
            do_plot=False,
            return_full=True,
            collect_stats=True,
        )
        return {
            "logL": params_sim["logL"],
            "params": params_sim["params"],
            "converged": params_sim["converged"],
        }
    except Exception:
        return {"logL": np.nan, "params": None, "converged": False}

def fit_single_sim_delta_TS(args):

    phi_sim, ephi_sim, mu_gauss, sigma_gauss, mu_ln, sigma_ln= args
    #print(f" this is np.min(phi_sim / phi_med) = {print(np.min(phi_sim/phi_med))}")
    phi_med = np.median(phi_sim)
    try:

        # avoid non-positive values for lognormal
        mask = phi_sim > 0
        phi_sim = phi_sim[mask]
        ephi_sim = ephi_sim[mask]

        # Gaussian fit
        params_gauss = fitting(
            phi_sim / phi_med,
            ephi_sim / phi_med,
            mu_gauss,
            sigma_gauss,
            model="gaussian"
        )

        logL_gauss = params_gauss[4]
        pdf_norm = lambda x: pdf_Gauss(x, params_gauss[0], params_gauss[1])

        # Lognormal fit
        params_ln = fitting(
            phi_sim / phi_med,
            ephi_sim / phi_med,
            mu_ln,
            sigma_ln,
            model="lognorm"
        )

        logL_ln = params_ln[4]
        pdf_ln = lambda x: pdf_LogNorm(x, params_ln[0], params_ln[1])

        delta_TS = -2 * (logL_ln - logL_gauss)

        return delta_TS, True, logL_ln, logL_gauss

    except Exception:
        return np.nan, False

def plot_delta_TS_gaussian_vs_lognorm(
        phi, ephi, t,
        fit_params_gauss,
        fit_params_lognorm,
        path,
        n_sim=100,
        title="ΔTS Gaussian vs Lognormal"
):
    """
    Simulate datasets from Gaussian fit, fit both Gaussian and Lognormal,
    compute ΔTS = -2*(logL_lognorm - logL_gauss) for each simulation,
    compare to real data ΔTS.
    """

    mu_real, sigma_real = fit_params_gauss[0], fit_params_gauss[1]
    mu_ln, sigma_ln =  fit_params_lognorm[0], fit_params_lognorm[1]
    logL_real_gauss = fit_params_gauss[4]
    logL_real_ln = fit_params_lognorm[4]

    # Real data ΔTS
    delta_TS_real = -2 * (logL_real_ln - logL_real_gauss)
    print(f"Real data ΔTS = {delta_TS_real:.3f}")

    # Compute C for flux errors
    C = fit_error_trend(phi, ephi)

    # Simulate datasets from Gaussian
    sims = simulate_distribution(phi, ephi, fit_params_gauss, n_sim=n_sim, model="gaussian", C=C)
    for i, (phi_sim, ephi_sim) in enumerate(sims):

        if phi_sim is None:
            print("Simulation", i, "is None")

        elif not np.all(np.isfinite(phi_sim)):
            print("Simulation", i, "has invalid values")

        elif np.any(phi_sim <= 0):
            print("Simulation", i, "has negative values")
    # Prepare arguments for parallel processing
    args_list = [
        (phi_sim, ephi_sim, mu_real, sigma_real, mu_ln, sigma_ln)
        for phi_sim, ephi_sim in sims
    ]

    # Use all available CPU cores
    n_cores = cpu_count()
    with Pool(n_cores) as pool:
        results = pool.map(fit_single_sim_delta_TS, args_list)

    delta_TS_sims = np.array([res[0] for res in results])
    converged_flags = np.array([res[1] for res in results])
    logL_ln_sim_array = np.array([res[2] for res in results])
    logL_gauss_sim_array = np.array([res[3] for res in results])




    # Remove simulations that did not converge or have NaN/Inf
    mask = converged_flags & np.isfinite(delta_TS_sims)
    delta_TS_sims_clean = delta_TS_sims[mask]

    n_failed = np.sum(~mask)
    if n_failed > 0:
        print(f"Warning: {n_failed}/{len(delta_TS_sims)} simulations removed due to non-finite ΔTS or failed fit.")

    # Now fit Gaussian safely
    mean_TS, std_TS = norm.fit(delta_TS_sims_clean)

    # Plot
    hist_counts, bin_edges = np.histogram(delta_TS_sims_clean, bins=20)
    x_vals = np.linspace(np.min(delta_TS_sims_clean), np.max(delta_TS_sims_clean), 300)
    pdf_vals = norm.pdf(x_vals, mean_TS, std_TS)

    # Scale PDF to histogram
    scale = max(hist_counts) / max(pdf_vals)
    pdf_vals *= scale

    # Plot histogram
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(delta_TS_sims_clean, bins=20, alpha=0.7, color="tab:blue", label="Simulated ΔTS")
    ax.plot(x_vals, pdf_vals, color="black", linewidth=2, label=f"Gaussian fit: μ={mean_TS:.2f}, σ={std_TS:.2f}")
    # Vertical line for real ΔTS
    ax.axvline(delta_TS_real, color="red", linestyle="-", label=f"CTAagnVar data ΔTS = {delta_TS_real:.2f}")
    ax.axvline(mean_TS, color = 'gray', linestyle ="--", label=f"MC mean ΔTS = {mean_TS:.2f}")


    overall_delta_TS =  delta_TS_real - mean_TS
    ax.text(
        0.05, 0.95,
        fr"$\Delta TS = {overall_delta_TS:.2f}$" + "\n" +
        fr"${abs(overall_delta_TS / std_TS):.2f}\sigma$",
        transform=ax.transAxes,
        va="top", ha="left", fontsize=12,
        bbox=dict(facecolor="white", alpha=0.8)
    )

    ax.set_title(title)
    ax.set_xlabel("ΔTS")
    ax.set_ylabel("Counts")
    ax.legend(loc = 'lower right')
    ax.grid(alpha=0.3)
    plt.savefig(f"{path}/Delta_TS_Gauss_vs_LN.pdf", dpi=150)
    plt.close(fig)

    print(f"Simulated ΔTS mean = {mean_TS:.3f}, std = {std_TS:.3f}")
    print(f"LogL lognorm CTAagnVar = {logL_real_ln}")
    print(f" LogL gauss CTAagnVar ={logL_real_gauss}")
    data = Table()

    data['Delta_TS_sim'] = delta_TS_sims
    data['LogL_ln'] = logL_ln_sim_array
    data['LogL_gauss'] = logL_gauss_sim_array


    ascii.write(data, f"{path}/Delta_TS_results_Gauss_LN.dat", overwrite=True)

    return delta_TS_sims, delta_TS_real, mean_TS, std_TS


def estimate_alpha_tail(data, tail_fraction=0.1):
    """
    Estimate alpha from the power-law tail decay using log-log regression.
    tail_fraction: fraction of data to use from each tail ( 0.1 =  10%)
    """
    n = len(data)
    sorted_data = np.sort(np.abs(data - np.median(data)))  # center first

    # Use the upper tail only
    tail_start = int(n * (1 - tail_fraction))
    tail_values = sorted_data[tail_start:]

    # survival function: P(X > x)
    ranks = np.arange(len(tail_values), 0, -1) / n

    # Log-log regression: log P(X > x) = -alpha * log(x) + const
    log_x = np.log(tail_values)
    log_p = np.log(ranks)

    slope, _ = np.polyfit(log_x, log_p, 1)
    alpha0 = np.clip(-slope, 0., 2.0)
    return alpha0


if __name__ == "__main__":

    #Load the data and plot the ligthcurve
    filename = args.p
    name = args.s
    path = args.fig_path + name
    if not os.path.exists(path):
        os.makedirs(path)
        print("Created directory: {path}")
    else:
        print("Directory already exists")

    title = f"{name} - CTAO simulation"
    t, phi, ephi = load_lightcurve(filename)
    phi_med = np.median(phi)
    print(f"This is phi median {phi_med}")

    #Plot the initial lightcurve as it is
    mask = plot_lightcurve(title, t, phi, ephi, path)
    phi = phi[~mask]
    ephi = ephi[~mask]

    pdf_path = f"{path}/likelihood_scans_initial_LC.pdf"

    with PdfPages(pdf_path) as pdf:
        mean, std = np.mean(phi ), np.std(phi)
        params_fited_norm = fitting(
            phi ,
            ephi ,
            mean,
            std,
            model="gaussian",
            pdf=pdf,
            do_plot=True,
            collect_stats=False,
        )
        v = np.var(phi )
        sigma2 = np.log(1 + v / mean ** 2)
        sigma = np.sqrt(sigma2)

        mu = np.log(mean) - sigma2 / 2

        params_fited_ln = fitting(
            phi,
            ephi,
            mu,
            sigma,
            model="lognorm",
            pdf=pdf,
            do_plot=True,
            collect_stats=False,
        )
        loc0 = np.median(phi )
        scale0 = np.std(phi)
        alpha0 = 1.5 #estimate_alpha_tail(phi )

        params_fited_alpha = fitting(
            phi ,
            ephi ,
            loc0,
            scale0,
            init_alpha = alpha0,
            model="alpha",
            pdf=pdf,
            do_plot=True,
            collect_stats=False,
        )

    #plot fit

    fig, ax = plt.subplots(figsize=(8, 6), nrows=1, ncols=1)
    plt.subplots_adjust(bottom=0.1, top=0.95, left=0.1, right=0.96, hspace=0.05)
    ax.set_title(title)
    ax.set_xlabel(r"log$_{10}$(energy flux [TeV cm$^{-2}$ s$^{-1}$])", ha="center", va='center', labelpad=12,
    			  fontsize=12)
    ax.set_ylabel(r"# entries", ha="center", va='center', labelpad=12, fontsize=12)
    ratio = phi
    mask = ratio > 0
    log_phi = np.log10(ratio[mask])
    hist, bins = hist_flux((phi)[mask], (ephi)[mask], 4)
    ax.stairs(hist, bins)
    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    sel = hist >= 1
    ax.errorbar(bin_centers[sel], hist[sel], yerr=np.sqrt(hist[sel]), fmt='', mfc='none', color='tab:blue', ls='none')
    bin_size_log10 = bins[1] - bins[0]
    log10phi_plot = np.linspace(np.min(log_phi), np.max(log_phi))
    phi_plot = np.power(10, log10phi_plot)
    normalization_pdf = phi.size * bin_size_log10 * np.log(10)
    pdf_ln = lambda x: pdf_LogNorm(x, params_fited_ln[0], params_fited_ln[1])
    label_lognorm = (
    	"Lognormal fit\n"
    	fr"$\mu = {params_fited_ln[0]:.2f} \pm {params_fited_ln[2]:.2f}$, "
    	fr"$\sigma = {params_fited_ln[1]:.2f} \pm {params_fited_ln[3]:.2f}$"
    )
    ax.plot(log10phi_plot, phi_plot * pdf_ln(phi_plot) * normalization_pdf, label=label_lognorm)

    pdf_norm = lambda x: pdf_Gauss(x, params_fited_norm[0], params_fited_norm[1])
    label_norm = (
    	"Gaussian fit\n"
    	fr"$\mu = {params_fited_norm[0]:.2f} \pm {params_fited_norm[2]:.2f}$, "
    	fr"$\sigma = {params_fited_norm[1]:.2f} \pm {params_fited_norm[3]:.2f}$"
    )
    ax.plot(log10phi_plot, phi_plot * pdf_norm(phi_plot) * normalization_pdf, label=label_norm)


    #alpha
    pdf_alpha = lambda x: pdf_alpha_stable(x, params_fited_alpha[0], params_fited_alpha[1], params_fited_alpha[2])
    label_alpha = (
        "Alpha fit\n"
        fr"$\mu = {params_fited_alpha[0]:.2f} \pm {params_fited_alpha[3]:.2f}$, "
        fr"$\sigma = {params_fited_alpha[1]:.2f} \pm {params_fited_alpha[4]:.2f}$,"
        fr"$\alpha = {params_fited_alpha[2]:.2f} \pm {params_fited_alpha[5]:.2f}$,"
    )
    ax.plot(log10phi_plot, phi_plot * pdf_alpha(phi_plot) * normalization_pdf, label=label_alpha)

    plt.legend()
    plt.savefig(f"{path}/fit_histo_.pdf", dpi=150)

    print("Perform the simulations and the fit for Gaussian")

    # plot_likelihood_distribution(
    #     phi=phi,
    #     ephi=ephi,
    #     t=t,
    #     fit_params=params_fited_ln,
    #     path = path,
    #     model="gaussian",
    #     n_sim=args.n_sim,
    #     title="Gaussian Model Likelihood Test"
    # )
