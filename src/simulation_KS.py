import numpy as np
from scipy.interpolate import RectBivariateSpline

EPSILON = 1e-3

def simul_shocks(n_sample, T, mparam, state_init=None):
    n_agt = mparam.n_agt
    ashock = np.zeros([n_sample, T])
    ishock = np.ones([n_sample, n_agt, T])
    if state_init:
        # convert productivity to 0/1 variable
        ashock[:, 0:1] = ((state_init["ashock"] - 1) / mparam.delta_a + 1) / 2
        ishock[..., 0] = state_init["ishock"]
    else:
        ashock[:, 0] = np.random.binomial(1, 0.5, n_sample)  # stationary distribution of Z is (0.5, 0.5)
        ur_rate = ashock[:, 0] * mparam.ur_g + (1 - ashock[:, 0]) * mparam.ur_b
        ur_rate = np.repeat(ur_rate[:, None], n_agt, axis=1)
        rand = np.random.uniform(0, 1, size=(n_sample, n_agt))
        ishock[rand < ur_rate, 0] = 0

    for t in range(1, T):
        if_keep = np.random.binomial(1, 0.875, n_sample)  # prob for Z to stay the same is 0.875
        ashock[:, t] = if_keep * ashock[:, t - 1] + (1 - if_keep) * (1 - ashock[:, t - 1])

    for t in range(1, T):
        a0, a1 = ashock[:, None, t - 1], ashock[:, None, t]
        y_agt = ishock[:, :, t - 1]
        ur_rate = (1 - a0) * (1 - a1) * (1 - y_agt) * mparam.p_bb_uu + (1 - a0) * (1 - a1) * y_agt * mparam.p_bb_eu
        ur_rate += (1 - a0) * a1 * (1 - y_agt) * mparam.p_bg_uu + (1 - a0) * a1 * y_agt * mparam.p_bg_eu
        ur_rate += a0 * (1 - a1) * (1 - y_agt) * mparam.p_gb_uu + a0 * (1 - a1) * y_agt * mparam.p_gb_eu
        ur_rate += a0 * a1 * (1 - y_agt) * mparam.p_gg_uu + a0 * a1 * y_agt * mparam.p_gg_eu
        rand = np.random.uniform(0, 1, size=(n_sample, n_agt))
        ishock[rand < ur_rate, t] = 0

    ashock = (ashock * 2 - 1) * mparam.delta_a + 1  # convert 0/1 variable to productivity
    return ashock, ishock


def simul_k(n_sample, T, mparam, policy, policy_type, state_init=None, shocks=None):
    # policy_type: "pde" or "nn_share"
    # return k_cross [n_sample, n_agt, T]
    assert policy_type in ["pde", "nn_share"], "Invalid policy type"
    n_agt = mparam.n_agt
    if shocks:
        ashock, ishock = shocks
        assert n_sample == ashock.shape[0], "n_sample is inconsistent with given shocks."
        assert T == ashock.shape[1], "T is inconsistent with given shocks."
        if state_init:
            assert np.array_equal(ashock[..., 0:1], state_init["ashock"]) and \
                np.array_equal(ishock[..., 0], state_init["ishock"]), \
                "Shock inputs are inconsistent with state_init"
    else:
        ashock, ishock = simul_shocks(n_sample, T, mparam, state_init)

    k_cross = np.zeros([n_sample, n_agt, T])
    if state_init is not None:
        assert n_sample == state_init["k_cross"].shape[0], "n_sample is inconsistent with state_init."
        k_cross[:, :, 0] = state_init["k_cross"]
    else:
        k_cross[:, :, 0] = mparam.k_ss
    csmp = np.zeros([n_sample, n_agt, T-1])
    wealth = k_cross.copy()
    if policy_type == "pde":
        for t in range(1, T):
            wealth[:, :, t] = next_wealth(k_cross[:, :, t-1], ashock[:, t-1:t], ishock[:, :, t-1], mparam)
            k_cross_t = policy(k_cross[:, :, t-1], ashock[:, t-1:t], ishock[:, :, t-1])
            # avoid csmp being too small or even negative
            k_cross[:, :, t] = np.clip(k_cross_t, EPSILON, wealth[:, :, t]-np.minimum(1.0, 0.8*wealth[:, :, t]))
            csmp[:, :, t-1] = wealth[:, :, t] - k_cross[:, :, t]
    if policy_type == "nn_share":
        for t in range(1, T):
            wealth[:, :, t] = next_wealth(k_cross[:, :, t-1], ashock[:, t-1:t], ishock[:, :, t-1], mparam)
            csmp_t = policy(k_cross[:, :, t-1], ashock[:, t-1:t], ishock[:, :, t-1]) * wealth[:, :, t]
            csmp_t = np.clip(csmp_t, EPSILON, wealth[:, :, t]-EPSILON)
            k_cross[:, :, t] = wealth[:, :, t] - csmp_t
            csmp[:, :, t-1] = csmp_t
    simul_data = {"k_cross": k_cross, "csmp": csmp, "ashock": ashock, "ishock": ishock}
    return simul_data


def next_wealth(k_cross, ashock, ishock, mparam):
    k_mean = np.mean(k_cross, axis=1, keepdims=True)
    tau = np.where(ashock < 1, mparam.tau_b, mparam.tau_g)  # labor tax rate based on ashock
    emp = np.where(ashock < 1, mparam.l_bar*mparam.er_b, mparam.l_bar*mparam.er_g)  # total labor supply based on ashock
    R = 1 - mparam.delta + ashock * mparam.alpha*(k_mean / emp)**(mparam.alpha-1)
    wage = ashock*(1-mparam.alpha)*(k_mean / emp)**(mparam.alpha)
    wealth = R * k_cross + (1-tau)*wage*mparam.l_bar*ishock + mparam.mu*wage*(1-ishock)
    return wealth


def k_policy_bspl(k_cross, ashock, ishock, splines):
    k_next = np.zeros_like(k_cross)
    k_mean = np.repeat(np.mean(k_cross, axis=1, keepdims=True), k_cross.shape[1], axis=1)

    idx = ((ashock < 1) & (ishock == 0))
    k_tmp, km_tmp = k_cross[idx], k_mean[idx]
    k_next[idx] = splines['00'](k_tmp, km_tmp, grid=False)

    idx = ((ashock < 1) & (ishock == 1))
    k_tmp, km_tmp = k_cross[idx], k_mean[idx]
    k_next[idx] = splines['01'](k_tmp, km_tmp, grid=False)

    idx = ((ashock > 1) & (ishock == 0))
    k_tmp, km_tmp = k_cross[idx], k_mean[idx]
    k_next[idx] = splines['10'](k_tmp, km_tmp, grid=False)

    idx = ((ashock > 1) & (ishock == 1))
    k_tmp, km_tmp = k_cross[idx], k_mean[idx]
    k_next[idx] = splines['11'](k_tmp, km_tmp, grid=False)

    return k_next


def construct_bspl(mats):
    # mats is saved in Matlab through
    # "save(filename, 'kprime', 'k', 'km', 'agshock', 'idshock', 'kmts', 'kcross');"
    splines = {
        '00': RectBivariateSpline(mats['k'], mats['km'], mats['kprime'][:, :, 0, 0]),
        '01': RectBivariateSpline(mats['k'], mats['km'], mats['kprime'][:, :, 0, 1]),
        '10': RectBivariateSpline(mats['k'], mats['km'], mats['kprime'][:, :, 1, 0]),
        '11': RectBivariateSpline(mats['k'], mats['km'], mats['kprime'][:, :, 1, 1]),
    }
    return splines
