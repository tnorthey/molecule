from scipy.stats import chisquare

# make f_obs be "random.xyz"'s spectrum
# loop over all calculated spectra for f_exp

chi_sq, p = chisquare(f_obs, f_exp)

# save all values of chi^2 and p.

