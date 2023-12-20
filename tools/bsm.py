# -*- coding: utf-8 -*-
"""
Created on 2023/11/20 16:04
@author: jhyu
"""
import numpy as np
import scipy.stats as spst
import scipy.optimize as spopt
import warnings

from . import option_abc as opt


class Bsm(opt.OptABC):

    @staticmethod
    def price_formula(strike, spot, sigma, texp, cp=1, intr=0.0, divr=0.0, is_fwd=False):
        """
        Black-Scholes-Merton model call/put option pricing formula (static method)

        Args:
            strike: strike price
            spot: spot (or forward)
            sigma: model volatility
            texp: time to expiry
            cp: 1/-1 for call/put option
            intr: interest rate (domestic interest rate)
            divr: dividend/convenience yield (foreign interest rate)
            is_fwd: if True, treat `spot` as forward price. False by default.

        Returns:
            Vanilla option price
        """
        disc_fac = np.exp(-texp*intr)
        fwd = np.array(spot)*(1.0 if is_fwd else np.exp(-texp*divr)/disc_fac)

        sigma_std = np.maximum(np.array(sigma)*np.sqrt(texp), np.finfo(float).tiny)

        # don't directly compute d1 just in case sigma_std is infty
        d1 = np.log(fwd/strike)/sigma_std
        d2 = d1 - 0.5*sigma_std
        d1 += 0.5*sigma_std

        cp = np.array(cp)
        price = fwd*spst.norm._cdf(cp*d1) - strike*spst.norm._cdf(cp*d2)
        price *= cp*disc_fac
        return price

    @staticmethod
    def d1sigma(d1, ln_k):
        sig = np.array(np.sqrt(d1**2 + 2*ln_k) + np.abs(d1))
        np.divide(2*ln_k, sig, out=sig, where=d1 < 0.)
        return sig

    @staticmethod
    def vega_std(sigma, ln_k):
        """
        Standardized Vega

        Args:
            sigma: volatility
            ln_k: log strike

        Returns:

        """
        # don't directly compute d1 just in case sigma_std is infty
        # handle the case ln_k = sigma = 0 (ATM)
        d1 = np.where(ln_k == 0., 0., -ln_k/sigma)
        d1 += 0.5*sigma
        vega = spst.norm._pdf(d1)
        return vega

    def delta(self, strike, spot, texp, cp=1):

        fwd, df, divf = self._fwd_factor(spot, texp)

        sigma_std = np.maximum(self.sigma*np.sqrt(texp), np.finfo(float).tiny)
        d1 = np.log(fwd/strike)/sigma_std
        d1 += 0.5*sigma_std

        delta = cp*spst.norm._cdf(cp*d1)  # formula according to wikipedia
        delta *= df if self.is_fwd else divf
        return delta

    def vega(self, strike, spot, texp, cp=1):

        fwd, df, _ = self._fwd_factor(spot, texp)

        sigma_std = np.maximum(self.sigma*np.sqrt(texp), np.finfo(float).tiny)
        d1 = np.log(fwd/strike)/sigma_std
        d1 += 0.5*sigma_std

        # formula according to wikipedia
        vega = df * fwd * spst.norm.pdf(d1) * np.sqrt(texp)
        return vega

    def theta(self, strike, spot, texp, cp=1):
        fwd, df, divf = self._fwd_factor(spot, texp)

        sigma_std = np.maximum(self.sigma*np.sqrt(texp), 100*np.finfo(float).tiny)
        d1 = np.log(fwd/strike)/sigma_std
        d2 = d1 - 0.5*sigma_std
        d1 += 0.5*sigma_std

        # still not perfect; need to consider the derivative w.r.t. divr and is_fwd = True
        divr_part = cp * fwd * spst.norm.cdf(cp * d1)
        divr_part *= self.intr if self.is_fwd else self.divr
        theta = (
                -0.5*spst.norm.pdf(d1)*fwd*self.sigma/np.sqrt(texp) -
                cp*self.intr*strike * spst.norm.cdf(cp*d2) + divr_part
        )
        theta *= df
        return theta

    def rho(self, strike, spot, texp, cp=1):

        fwd, df, _ = self._fwd_factor(spot, texp)

        sigma_std = np.maximum(self.sigma * np.sqrt(texp), np.finfo(float).tiny)
        d1 = np.log(fwd / strike) / sigma_std
        d2 = d1 - 0.5 * sigma_std
        d1 += 0.5 * sigma_std

        val = cp * strike * spst.norm.cdf(cp * d2)
        val -= (cp * fwd * spst.norm.cdf(cp * d1)) if self.is_fwd else 0
        val *= (np.array(texp) * df)
        return val

    def gamma(self, strike, spot, texp, cp=1):

        fwd, df, divf = self._fwd_factor(spot, texp)

        sigma_std = np.maximum(self.sigma*np.sqrt(texp), 100*np.finfo(float).tiny)
        d1 = np.log(fwd/strike)/sigma_std
        d1 += 0.5*sigma_std

        gamma = df * spst.norm.pdf(d1) / fwd / sigma_std  # formula according to wikipedia
        if not self.is_fwd:
            gamma *= (divf/df)**2
        return gamma

    def volga(self, strike, spot, texp, cp=1):
        """
        Second derivative w.r.t. sigma. Say, vomma = DvegaDsigma.
        """
        fwd, df, _ = self._fwd_factor(spot, texp)

        sigma_std = np.maximum(self.sigma*np.sqrt(texp), np.finfo(float).tiny)
        d1 = np.log(fwd/strike)/sigma_std
        d2 = d1 - 0.5*sigma_std
        d1 += 0.5*sigma_std

        # formula according to wikipedia
        vega2 = df*fwd*spst.norm.pdf(d1)*np.sqrt(texp) * d1*d2 / self.sigma
        return vega2

    def charm(self, strike, spot, texp, cp=1):
        """
        Delta decay.
        partial(V^2)/(partial(tau) * partial (S)) = Ddelta/Dtime.
        """
        fwd, df, divf = self._fwd_factor(spot, texp)
        sigma_std = np.maximum(self.sigma * np.sqrt(texp), 100 * np.finfo(float).tiny)
        d1 = np.log(fwd / strike) / sigma_std
        d2 = d1 - 0.5 * sigma_std
        d1 += 0.5 * sigma_std

        if self.is_fwd:
            val = (
                    cp * self.intr * spst.norm.cdf(cp * d1) +
                    spst.norm.pdf(d1) * d2 / (2 * np.array(texp))
            )
            val *= df
        else:
            val = (
                    cp * self.divr * spst.norm.cdf(cp * d1) -
                    spst.norm.pdf(d1) * (2 * (self.intr - self.divr) * np.array(texp) - d2 * sigma_std) /
                    (2 * np.array(texp) * sigma_std)
            )
            val *= divf
        return val

    def veta(self, strike, spot, texp, cp=1):
        """
        DvegaDtime.
        """
        fwd, df, divf = self._fwd_factor(spot, texp)
        sigma_std = np.maximum(self.sigma * np.sqrt(texp), 100 * np.finfo(float).tiny)
        d1 = np.log(fwd / strike) / sigma_std
        d2 = d1 - 0.5 * sigma_std
        d1 += 0.5 * sigma_std

        val = df * fwd * spst.norm.pdf(d1) * np.sqrt(texp)
        if self.is_fwd:
            val *= (self.intr - (1 + d1 * d2) / (2 * np.array(texp)))
        else:
            val *= (self.divr + (self.intr-self.divr) * d1 / sigma_std - (1 + d1 * d2) / (2 * np.array(texp)))
        return val

    def vanna(self, strike, spot, texp, cp=1):
        """
        DvegaDspot = DdeltaDsigma.
        """
        fwd, df, divf = self._fwd_factor(spot, texp)

        sigma_std = np.maximum(self.sigma * np.sqrt(texp), 100 * np.finfo(float).tiny)
        d1 = np.log(fwd / strike) / sigma_std
        d2 = d1 - 0.5 * sigma_std
        d1 += 0.5 * sigma_std

        val = - spst.norm.pdf(d1) * d2 / self.sigma
        val *= df if self.is_fwd else divf
        return val

    def speed(self, strike, spot, texp, cp=1):
        """
        Third derivative w.r.t. spot. Say, DgammaDspot.
        """
        fwd, df, divf = self._fwd_factor(spot, texp)

        sigma_std = np.maximum(self.sigma * np.sqrt(texp), 100 * np.finfo(float).tiny)
        d1 = np.log(fwd / strike) / sigma_std
        d1 += 0.5 * sigma_std

        val = - spst.norm.pdf(d1) / sigma_std * (d1 / sigma_std + 1)
        val *= (df / fwd**2) if self.is_fwd else (divf / spot**2)
        return val

    def ultima(self, strike, spot, texp, cp=1):
        """
        Third derivative w.r.t. sigma. Say, DvolgaDsigma.
        """
        fwd, df, divf = self._fwd_factor(spot, texp)

        sigma_std = np.maximum(self.sigma * np.sqrt(texp), 100 * np.finfo(float).tiny)
        d1 = np.log(fwd / strike) / sigma_std
        d2 = d1 - 0.5 * sigma_std
        d1 += 0.5 * sigma_std

        part2 = (d1**2 + d2**2 - (d1 * d2)**2) if self.is_fwd else (d1 * d2 * (1 - d1 * d2) + d1**2 + d2**2)
        val = - df * fwd * spst.norm.pdf(d1) * np.sqrt(texp) / self.sigma**2 * part2
        return val

    def color(self, strike, spot, texp, cp=1):
        """
        Gamma decay.
        partial(V^3)/(partial(tau) * partial (S^2)) = Dgamma/Dtime.
        """
        fwd, df, divf = self._fwd_factor(spot, texp)

        sigma_std = np.maximum(self.sigma * np.sqrt(texp), 100 * np.finfo(float).tiny)
        d1 = np.log(fwd / strike) / sigma_std
        d2 = d1 - 0.5 * sigma_std
        d1 += 0.5 * sigma_std

        if self.is_fwd:
            val = df * spst.norm.pdf(d1) / (2 * fwd * np.array(texp) * sigma_std) * (
                    2 * self.intr * np.array(texp) + 1 - d1*d2)
        else:
            val = divf * spst.norm.pdf(d1) / (2 * spot * np.array(texp) * sigma_std) * (
                    2 * self.divr * np.array(texp) + 1 +
                    (2 * (self.intr - self.divr) * np.array(texp) - d2 * sigma_std) / sigma_std * d1
            )
        return val

    def zomma(self, strike, spot, texp, cp=1):
        """
        DgammaDsigma.
        """
        fwd, df, divf = self._fwd_factor(spot, texp)

        sigma_std = np.maximum(self.sigma * np.sqrt(texp), 100 * np.finfo(float).tiny)
        d1 = np.log(fwd / strike) / sigma_std
        d2 = d1 - 0.5 * sigma_std
        d1 += 0.5 * sigma_std

        part1 = (df / fwd) if self.is_fwd else (divf / spot)
        val = part1 * spst.norm.pdf(d1) / (self.sigma**2 * np.array(texp)) * (d1 * d2 - 1)
        return val

    def impvol_naive(self, price, strike, spot, texp, cp=1, setval=False):
        """
        BSM implied volatility with Newton's method.

        Args:
            price: option price
            strike: strike price
            spot: spot (or forward) price
            texp: time to expiry
            cp: 1/-1 for call/put option
            setval: if True, sigma is set with the solved implied volatility

        Returns:
            implied volatility
        """

        fwd, df, divf = self._fwd_factor(spot, texp)

        strike_std = strike/fwd  # strike / fwd
        price_std = price/df/fwd  # forward price / fwd

        bsm_model = Bsm(0, is_fwd=True)
        p_min = bsm_model.price(strike_std, 1.0, texp, cp)
        bsm_model.sigma = np.inf
        p_max = bsm_model.price(strike_std, 1.0, texp, cp)
        scalar_output = np.isscalar(p_min) & np.isscalar(price_std)
        if np.isscalar(texp):
            texp += 1e-10
        else:
            texp[texp == 0] = 1e-10

        # Exclude optoin price below intrinsic value or above max value (1 for call or k for put)
        # ind_solve can be scalar or array. scalar can be fine in np.abs(p_err[ind_solve])
        ind_solve = (price_std - p_min > Bsm.IMPVOL_TOL) & (p_max - price_std > Bsm.IMPVOL_TOL)

        # initial guess = inflection point in sigma (volga=0)
        _sigma = np.ones_like(ind_solve)*np.sqrt(2*np.abs(np.log(strike_std))/texp)

        bsm_model.sigma = _sigma
        p_err = bsm_model.price(strike_std, 1.0, texp, cp) - price_std
        # print(np.sign(p_err), _sigma)

        if np.any(ind_solve):
            for k in range(32):  # usually iteration ends less than 10
                vega = bsm_model.vega(strike_std, 1.0, texp, cp)
                _sigma -= p_err/vega
                bsm_model.sigma = _sigma
                p_err = bsm_model.price(strike_std, 1.0, texp, cp) - price_std
                p_err_max = np.amax(np.abs(p_err[ind_solve]))

                # ignore the error of the elements with ind_solve = False
                if p_err_max < Bsm.IMPVOL_TOL:
                    break

            if p_err_max >= Bsm.IMPVOL_TOL:
                warn_msg = f"impvol_newton did not converged within {k} iterations: max error = {p_err_max}"
                warnings.warn(warn_msg, Warning)

        # Put Nan for the out-of-bound option prices
        _sigma = np.where(ind_solve, _sigma, np.nan)

        # Though not error is above tolerance, if the price is close to min or max, set 0 or inf
        _sigma = np.where(
            (np.abs(p_err) >= Bsm.IMPVOL_TOL)
            & (np.abs(price_std - p_min) <= Bsm.IMPVOL_TOL),
            0,
            _sigma,
        )
        _sigma = np.where(
            (np.abs(p_err) >= Bsm.IMPVOL_TOL)
            & (np.abs(price_std - p_max) <= Bsm.IMPVOL_TOL),
            np.inf,
            _sigma,
        )

        if scalar_output:
            _sigma = _sigma.item()

        if setval:
            self.sigma = _sigma

        return _sigma

    def _price_suboptimal(self, strike, spot, texp, cp=1, strike2=None):
        strike2 = strike if strike2 is None else strike2
        fwd, df, _ = self._fwd_factor(spot, texp)

        sigma_std = np.maximum(self.sigma*np.sqrt(texp), np.finfo(float).tiny)
        d1 = np.log(fwd/strike2)/sigma_std
        d2 = d1 - 0.5*sigma_std
        d1 += 0.5*sigma_std

        price = fwd*spst.norm._cdf(cp*d1) - strike*spst.norm._cdf(cp*d2)
        price *= cp*df
        return price

    def _barrier_params(self, barrier, spot):
        """
        Parameters used for barrier option pricing

        Args:
            barrier: barrier price
            spot: spot price

        Returns:
            barrier option pricing parameters (psi, spot_mirror)
        """
        psi = np.power(
            barrier/spot, 2*(self.intr - self.divr)/self.sigma**2 - 1
        )
        spot_reflected = barrier**2/spot
        return psi, spot_reflected

    def price_barrier(self, strike, barrier, spot, texp, cp=1, io=-1):
        """
        Barrier option price under the BSM model

        Args:
            strike: strike price
            barrier: knock-in/out barrier price
            spot: spot price
            texp: time to expiry
            cp: 1/-1 for call/put option
            io: +1 for knock-in, -1 for knock-out

        Returns:
            Barrier option price
        """

        psi, spot_reflected = self._barrier_params(barrier, spot)

        """
        `mirror_sign` is +1/-1 if call/put remains same/flipped in the reflection principle
        +1 if (barrier < spot AND call) or (barrier > spot AND put), -1 otherwise
        """
        mirror_sign = np.where(barrier < spot, 1, -1)*cp
        """
        This is a trick to handle a trivial case:
        Knock-out call with spot < strike < barrier is worth zero.
        Knock-out put with barrier < strike < spot is worth zero.
        Without explicit adjustment, Knock-out price is negative, Knock-in price is higher than vanilla.
        In both scenario (mirror_sign = -1), we set strike = barrier, which will do the adjustment.
        """
        barrier = np.where(
            mirror_sign > 0, barrier, cp*np.maximum(cp*strike, cp*barrier)
        )

        p_euro1 = np.where(
            mirror_sign > 0,
            0,
            self._price_suboptimal(strike, spot, texp, cp=cp, strike2=barrier),
        )

        p_euro2 = self._price_suboptimal(
            strike, spot_reflected, texp, cp=mirror_sign*cp
        )
        p_euro2 -= np.where(
            mirror_sign > 0,
            0,
            self._price_suboptimal(
                strike, spot_reflected, texp, cp=mirror_sign*cp, strike2=barrier
            ),
        )

        p = p_euro1 + psi*p_euro2  # knock-in price
        p = np.where(
            io > 0,
            p,  # knock-in type
            self._price_suboptimal(strike, spot, texp, cp=cp) - p,
        )

        return p

    def price_vsk(self, texp=1):
        """
        Variance, skewness, and ex-kurtosis. Assume mean=1.

        Args:
            texp: time-to-expiry

        Returns:
            (variance, skewness, and ex-kurtosis)

        References:
            https://en.wikipedia.org/wiki/Log-normal_distribution
        """
        var = np.expm1(texp*self.sigma**2)
        skew = (var + 3)*np.sqrt(var)
        exkurt = var*(var*(var*(var + 6) + 12) + 13)  # (1+var)**4 + 2*(1+var)**3 + 3*(1+var) - 6
        return var, skew, exkurt
