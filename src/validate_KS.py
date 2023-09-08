import json
import os
import numpy as np
from absl import app
from absl import flags
from param import KSParam
from dataset import KSInitDataSet
from value import ValueTrainer
from policy import KSPolicyTrainer
from simulation_KS import simul_shocks, simul_k

flags.DEFINE_string("model_path", "../data/simul_results/KS/game_nn_n50_test",
                    """The path to load json file.""",
                    short_name='m')
flags.DEFINE_integer('n_agt', -1, "Number of agents in validation simulation")
flags.DEFINE_boolean('save', True, "Save simulation results or not.")
FLAGS = flags.FLAGS

def bellman_err(simul_data, shocks, ptrainer, value_fn, prefix="", nnext=100, nc=10, seed=None, nt=None):
    if seed:
        np.random.seed(seed)
    k_cross, csmp = simul_data["k_cross"], simul_data["csmp"]
    K = np.mean(k_cross, axis=1, keepdims=True)
    if nt:
        nt = min(nt, csmp.shape[-1])
    else:
        nt = csmp.shape[-1]
    mparam = ptrainer.mparam
    # compute error for n_path * n_agt * nt states
    t_idx = np.random.choice(csmp.shape[-1], nt)
    n_agt = csmp.shape[1]
    k_now, k_next = k_cross[:, :, t_idx], k_cross[:, :, t_idx+1]
    knormed_now = ptrainer.init_ds.normalize_data(k_now, key="agt_s")
    knormed_next = ptrainer.init_ds.normalize_data(k_next, key="agt_s")
    knormed_mean_now = np.repeat(np.mean(knormed_now, axis=-2, keepdims=True), n_agt, axis=1)
    knormed_mean_next = np.repeat(np.mean(knormed_next, axis=-2, keepdims=True), n_agt, axis=1)
    K_now, K_next = np.repeat(K[:, :, t_idx], n_agt, axis=1), np.repeat(K[:, :, t_idx+1], n_agt, axis=1)
    c_now = csmp[:, :, t_idx]
    ashock = shocks[0][:, t_idx]
    ishock = shocks[1][:, :, t_idx]

    if_keep = np.random.binomial(1, 0.875, [ashock.shape[0], ashock.shape[1], nnext])  # prob for Z to stay the same
    # ashock = 1-delta or 1+delta
    ashock_next = if_keep * ashock[:, :, None] + (1 - if_keep) * (2 - ashock[:, :, None])
    ashock_01 = (((ashock-1) / mparam.delta_a + 1) / 2).astype(int)
    ashock_next_01 = (((ashock_next-1) / mparam.delta_a + 1) / 2).astype(int)
    y_agt = ishock.copy()
    ishock0 = np.zeros_like(ishock)
    ishock1 = np.ones_like(ishock)

    def gm_fn(knormed):  # k_normalized of shape B * n_agt * T
        knormed = knormed.transpose((0, 2, 1))[:, :, :, None]
        basis = [None] * len(ptrainer.vtrainers)
        gm = [None] * len(ptrainer.vtrainers)
        for i, vtr in enumerate(ptrainer.vtrainers):
            basis[i] = vtr.gm_model.basis_fn(knormed).numpy()
            basis[i] = basis[i].transpose((0, 2, 1, 3))
            gm[i] = np.repeat(np.mean(basis[i], axis=1, keepdims=True), n_agt, axis=1)
        return basis, gm

    basic_s_now = np.stack([k_now, K_now, np.repeat(ashock[:, None, :], n_agt, axis=1), ishock], axis=-1)
    if ptrainer.init_ds.config["n_fm"] == 2 and "pde" not in prefix:
        knormed_sqr_mean_now = np.repeat(np.mean(knormed_now**2, axis=1, keepdims=True), n_agt, axis=1)
        knormed_sqr_mean_next = np.repeat(np.mean(knormed_next**2, axis=1, keepdims=True), n_agt, axis=1)
        fm_extra_now = knormed_sqr_mean_now-knormed_mean_now**2
        fm_extra_now = fm_extra_now[:, :, :, None]
    else:
        fm_extra_now = None
    if ptrainer.init_ds.config["n_gm"] > 0 and "pde" not in prefix:
        _, gm_now = gm_fn(knormed_now)
        gm_basis_next, gm_next = gm_fn(knormed_next)
    else:
        gm_now, gm_next = None, None
    v_now = value_fn(basic_s_now, fm_extra_now, gm_now)
    def next_value_fn(c_tmp):
        k_next_tmp = k_next + (c_now - c_tmp)
        K_next_tmp = K_next + (k_next_tmp - k_next) / n_agt
        knormed_next_tmp = ptrainer.init_ds.normalize_data(k_next_tmp, key="agt_s")
        knormed_mean_next_tmp = knormed_mean_next + (knormed_next_tmp - knormed_next) / n_agt
        basic_s_next_tmp = [k_next_tmp, K_next_tmp]
        if ptrainer.init_ds.config["n_fm"] == 2 and "pde" not in prefix:
            knormed_sqr_mean_next_tmp = knormed_sqr_mean_next + (knormed_next_tmp**2 - knormed_next**2) / n_agt
            fm_extra_next_tmp = knormed_sqr_mean_next_tmp - knormed_mean_next_tmp**2
            fm_extra_next_tmp = fm_extra_next_tmp[:, :, :, None]
        else:
            fm_extra_next_tmp = None
        if ptrainer.init_ds.config["n_gm"] > 0 and "pde" not in prefix:
            gm_basis_next_tmp, _ = gm_fn(knormed_next_tmp)
            gm_next_tmp = [gm_next[i] + (gm_basis_next_tmp[i] - gm_basis_next[i]) / n_agt for i in range(len(gm_next))]
        else:
            gm_next_tmp = None
        v_tmp = np.zeros_like(v_now)
        for j in range(nnext):
            ashock_next_tmp = np.repeat(ashock_next[:, None, :, j], n_agt, axis=1)
            basic_s_next0_tmp = np.stack(basic_s_next_tmp + [ashock_next_tmp, ishock0], axis=-1)
            basic_s_next1_tmp = np.stack(basic_s_next_tmp + [ashock_next_tmp, ishock1], axis=-1)
            v_next0 = value_fn(basic_s_next0_tmp, fm_extra_next_tmp, gm_next_tmp)
            v_next1 = value_fn(basic_s_next1_tmp, fm_extra_next_tmp, gm_next_tmp)
            # convert to 0,1 for computing ishock transition
            a0, a1 = ashock_01[:, None, :], ashock_next_01[:, None, :, j]
            ur_rate = (1 - a0) * (1 - a1) * (1 - y_agt) * mparam.p_bb_uu + (1 - a0) * (1 - a1) * y_agt * mparam.p_bb_eu
            ur_rate += (1 - a0) * a1 * (1 - y_agt) * mparam.p_bg_uu + (1 - a0) * a1 * y_agt * mparam.p_bg_eu
            ur_rate += a0 * (1 - a1) * (1 - y_agt) * mparam.p_gb_uu + a0 * (1 - a1) * y_agt * mparam.p_gb_eu
            ur_rate += a0 * a1 * (1 - y_agt) * mparam.p_gg_uu + a0 * a1 * y_agt * mparam.p_gg_eu
            v_tmp += mparam.beta*(v_next0 * ur_rate + v_next1 * (1-ur_rate)) + np.log(c_tmp)
        v_tmp /= nnext
        return v_tmp
    # Bellman expectation error
    v_next = next_value_fn(c_now)
    err_blmexpct = v_now - v_next
    # Bellman expectation error
    v_next = np.zeros_like(v_now)
    c_max = c_now + np.minimum(k_next-1e-6, 5)  # sampliewise cmax
    c_min = c_now - np.minimum(c_now*0.95, 5)  # sampliewise cmin
    dc = (c_max - c_min) / nc
    for i in range(nc+1):
        c_tmp = c_min + dc * i
        v_tmp = next_value_fn(c_tmp)
        v_next = np.maximum(v_tmp, v_next)
    err_blmopt = v_now - v_next
    print("Bellman error of %3s: %.6f (expectation), %.6f (optimality)" % \
        (prefix.upper(), np.abs(err_blmexpct).mean(), np.abs(err_blmopt).mean()))
    return err_blmexpct, err_blmopt


def main(argv):
    del argv
    print("Validating the model from {}".format(FLAGS.model_path))
    with open(os.path.join(FLAGS.model_path, "config.json"), 'r') as f:
        config = json.load(f)
    config["dataset_config"]["n_path"] = config["simul_config"]["n_path"]
    config["init_with_bchmk"] = True
    if FLAGS.n_agt > 0:
        config["n_agt"] = FLAGS.n_agt
    mparam = KSParam(config["n_agt"], config["beta"], config["mats_path"])

    init_ds = KSInitDataSet(mparam, config)
    value_config = config["value_config"]
    vtrainers = [ValueTrainer(config) for i in range(value_config["num_vnet"])]
    for i, vtr in enumerate(vtrainers):
        vtr.load_model(os.path.join(FLAGS.model_path, "value{}.h5".format(i)))
    ptrainer = KSPolicyTrainer(vtrainers, init_ds, os.path.join(FLAGS.model_path, "policy.h5"))

    # long simulation
    simul_config = config["simul_config"]
    n_path = simul_config["n_path"]
    T = simul_config["T"]
    state_init = init_ds.next_batch(n_path)
    shocks = simul_shocks(n_path, T, mparam, state_init)
    simul_data_bchmk = simul_k(
        n_path, T, mparam, init_ds.k_policy_bchmk, policy_type="pde",
        state_init=state_init, shocks=shocks
    )
    simul_data_nn = simul_k(
        n_path, T, mparam, ptrainer.current_c_policy, policy_type="nn_share",
        state_init=state_init, shocks=shocks
    )

    # calculate path stats
    def path_stats(simul_data, prefix=""):
        k_mean = np.mean(simul_data["k_cross"], axis=1)
        discount = np.power(mparam.beta, np.arange(simul_data["csmp"].shape[-1]))
        util_sum = np.sum(np.log(simul_data["csmp"])*discount, axis=-1)
        print(
            "%8s: total utilily: %.5f, mean of k: %.5f, std of k: %.5f, max of k: %.5f, max of K: %.5f" % (
                prefix.upper(), util_sum.mean(), simul_data["k_cross"].mean(), simul_data["k_cross"].std(),
                simul_data["k_cross"].max(), k_mean.max())
        )
    path_stats(simul_data_bchmk, "KS")
    path_stats(simul_data_nn, "NN")

    # compute Bellman expectation error
    # value_fn_pde unavailable so far
    def value_fn_nn(basic_s, fm_extra, gm):
        basic_s = ptrainer.init_ds.normalize_data(basic_s, key="basic_s")
        if ptrainer.init_ds.config["n_fm"] == 0:
            basic_s = np.concatenate([basic_s[..., 0:1], basic_s[..., 2:]], axis=-1)
        if fm_extra is not None:
            n_state = basic_s.shape[-1] + fm_extra.shape[-1]
            state_fix = np.concatenate([basic_s, fm_extra], axis=-1)
        else:
            n_state = basic_s.shape[-1]
            state_fix = basic_s
        if gm is not None:
            n_state += gm[0].shape[-1]
            state = [None] * len(vtrainers)
            for i in range(len(vtrainers)):
                state[i] = np.concatenate([state_fix, gm[i]], axis=-1)
                state[i] = state[i].transpose((0, 2, 1, 3)).reshape((-1, config['n_agt'], n_state))
        else:
            state = [state_fix.transpose((0, 2, 1, 3)).reshape((-1, config['n_agt'], n_state))] * len(vtrainers)
        v = 0
        for i, vtr in enumerate(vtrainers):
            v += vtr.model(state[i]).numpy()
        v /= len(vtrainers)
        v = ptrainer.init_ds.unnormalize_data(v, key="value")
        # reshape and transpose back to path * n_agt * time
        v = v.reshape([basic_s.shape[0], basic_s.shape[2], basic_s.shape[1]])
        v = np.transpose(v, (0, 2, 1))
        return v
    err_blm_nn = bellman_err(simul_data_nn, shocks, ptrainer, value_fn_nn, "nn", seed=1, nt=100)

    if FLAGS.save:
        to_save = {
            "k_cross_bchmk": simul_data_bchmk["k_cross"],
            "k_cross_nn": simul_data_nn["k_cross"],
            "csmp_bchmk": simul_data_bchmk["csmp"],
            "csmp_nn": simul_data_nn["csmp"],
            "err_blmexpct_nn": err_blm_nn[0],
            "err_blmopt_nn": err_blm_nn[1],
            "ashock": shocks[0],
            "ishock": shocks[1],
        }
        np.savez(os.path.join(FLAGS.model_path, "paths.npz"), **to_save)

if __name__ == '__main__':
    app.run(main)
