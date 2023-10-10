import json
import os
import time
from absl import app
from absl import flags
from param import JFVParam
from dataset import JFVInitDataSet
from value import ValueTrainer
from policy import JFVPolicyTrainer
from util import print_elapsedtime

flags.DEFINE_string("config_path", "./configs/JFV_DSS/game_nn_n50.json",
                    """The path to load json file.""",
                    short_name='c')
flags.DEFINE_string("exp_name", "test",
                    """The suffix used in model_path for save.""",
                    short_name='n')
FLAGS = flags.FLAGS

def main(argv):
    del argv
    folder = "JFV_DSS" if "DSS" in FLAGS.config_path else "JFV_SSS"
    with open(FLAGS.config_path, 'r') as f:
        config = json.load(f)
    print("Solving the problem based on the config path {}".format(FLAGS.config_path))
    mparam = JFVParam(config["n_agt"], config["dt"], config["mats_path"], config["with_ashock"])
    # save config at the beginning for checking
    model_path = "../data/simul_results/{}/{}_{}_n{}_{}".format(
        folder,
        "game" if config["policy_config"]["opt_type"] == "game" else "sp",
        config["dataset_config"]["value_sampling"],
        config["n_agt"],
        FLAGS.exp_name,
    )
    os.makedirs(model_path, exist_ok=True)
    with open(os.path.join(model_path, "config_beg.json"), 'w') as f:
        json.dump(config, f)

    start_time = time.monotonic()

    # initial value training
    init_ds = JFVInitDataSet(mparam, config)
    value_config = config["value_config"]
    if config["init_with_bchmk"]:
        init_policy = init_ds.c_policy_bchmk
        policy_type = "pde"
        # TODO: change all "pde" to "conventional"
    else:
        init_policy = init_ds.c_policy_const_share
        policy_type = "nn_share"
    train_vds, valid_vds = init_ds.get_valuedataset(init_policy, policy_type, update_init=False)
    vtrainers = [ValueTrainer(config) for i in range(value_config["num_vnet"])]
    for vtr in vtrainers:
        vtr.train(train_vds, valid_vds, value_config["num_epoch"], value_config["batch_size"])

    # iterative policy and value training
    policy_config = config["policy_config"]
    ptrainer = JFVPolicyTrainer(vtrainers, init_ds)
    ptrainer.train(policy_config["num_step"], policy_config["batch_size"])

    # save config and models
    with open(os.path.join(model_path, "config.json"), 'w') as f:
        json.dump(config, f)
    for i, vtr in enumerate(vtrainers):
        vtr.save_model(os.path.join(model_path, "value{}.h5".format(i)))
    ptrainer.save_model(os.path.join(model_path, "policy.h5"))

    end_time = time.monotonic()
    print_elapsedtime(end_time - start_time)

if __name__ == '__main__':
    app.run(main)
