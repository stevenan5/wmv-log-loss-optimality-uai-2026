import gmpy2

LINESTYLE_DICT = {
    "solid": (0, ()),
    "loosely dotted": (0, (1, 10)),
    "dotted": (0, (1, 5)),
    "densely dotted": (0, (1, 1)),
    "long dash with offset": (5, (10, 3)),
    "loosely dashed": (0, (5, 10)),
    "dashed_intermed": (0, (5, 8)),
    "dashed": (0, (5, 5)),
    "densely dashed": (0, (5, 1)),
    "loosely dashdotted": (0, (3, 10, 1, 10)),
    "dashdotted": (0, (3, 5, 1, 5)),
    "densely dashdotted": (0, (3, 1, 1, 1)),
    "dashdotdotted": (0, (3, 5, 1, 5, 1, 5)),
    "loosely dashdotdotted": (0, (3, 10, 1, 10, 1, 10)),
    "densely dashdotdotted": (0, (3, 1, 1, 1, 1, 1)),
}


def ind(n_classes, pattern, ell):
    return n_classes * pattern + ell


def _check_shape(vector, vec_shape):
    obs_shape = vector.shape
    if obs_shape != vec_shape:
        raise ValueError(
            "Observed shape ", obs_shape, " doesn't match expected shape ", vec_shape
        )


def stars_and_bars(sum, n_vars):
    # computes the number of ways that we can assign non-negative
    # integers to n_vars and have the result sum to `sum`
    return gmpy2.bincoef(sum + n_vars - 1, n_vars - 1)


# n_dims is the number of dimensions the simplex is in.
# there are n_dims degress of freedom, which is what
# determines the volume
def scaled_simplex_vol(scale, n_dims: int):
    den = gmpy2.fac(n_dims - 1)
    num = scale ** (n_dims - 1)
    return gmpy2.mpq(num, den)
