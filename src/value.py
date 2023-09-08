import numpy as np
import tensorflow as tf
import util

DTYPE = "float64"
tf.keras.backend.set_floatx(DTYPE)
if DTYPE == "float64":
    NP_DTYPE = np.float64
elif DTYPE == "float32":
    NP_DTYPE = np.float32
else:
    raise ValueError("Unknown dtype.")

class ValueTrainer():
    def __init__(self, config):
        self.config = config
        self.value_config = config["value_config"]
        d_in = config["n_basic"] + config["n_fm"] + config["n_gm"]
        self.model = util.FeedforwardModel(d_in, 1, self.value_config, name="v_net")
        if config["n_gm"] > 0:
            # TODO generalize to multi-dimensional agt_s
            self.gm_model = util.GeneralizedMomModel(1, config["n_gm"], config["gm_config"], name="v_gm")
        self.train_vars = None
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.value_config["lr"], epsilon=1e-8,
            beta_1=0.99, beta_2=0.99
        )

    @tf.function
    def prepare_state(self, input_data):
        if self.config["n_fm"] == 2:
            k_var = tf.math.reduce_variance(input_data["agt_s"], axis=-2, keepdims=True)
            k_var = tf.tile(k_var, [1, input_data["agt_s"].shape[-2], 1])
            state = tf.concat([input_data["basic_s"], k_var], axis=-1)
        elif self.config["n_fm"] == 0:
            state = tf.concat([input_data["basic_s"][..., 0:1], input_data["basic_s"][..., 2:]], axis=-1)
        elif self.config["n_fm"] == 1:  # so far always add k_mean in the basic_state
            state = input_data["basic_s"]
        if self.config["n_gm"] > 0:
            gm = self.gm_model(input_data["agt_s"])
            state = tf.concat([state, gm], axis=-1)
        return state

    @tf.function
    def value_fn(self, input_data):
        state = self.prepare_state(input_data)
        value = self.model(state)
        return value

    @tf.function
    def loss(self, input_data):
        y_pred = self.value_fn(input_data)
        y = input_data["value"]
        loss = tf.reduce_mean(tf.square(y_pred - y))
        loss_dict = {"loss": loss}
        return loss_dict

    def grad(self, input_data):
        with tf.GradientTape(persistent=True) as tape:
            loss = self.loss(input_data)["loss"]
        train_vars = self.model.trainable_variables
        if self.config["n_gm"] > 0:
            train_vars += self.gm_model.trainable_variables
        self.train_vars = train_vars
        grad = tape.gradient(
            loss,
            train_vars,
            unconnected_gradients=tf.UnconnectedGradients.ZERO,
        )
        del tape
        return grad

    @tf.function
    def train_step(self, train_data):
        grad = self.grad(train_data)
        self.optimizer.apply_gradients(
            zip(grad, self.train_vars)
        )

    def train(self, train_dataset, valid_dataset, num_epoch=None, batch_size=None):
        train_dataset = train_dataset.batch(batch_size)

        for epoch in range(num_epoch+1):
            for train_data in train_dataset:
                self.train_step(train_data)
            if epoch % 20 == 0:
                for valid_data in valid_dataset:
                    val_loss = self.loss(valid_data)
                    print(
                        "Epoch: %d, validation loss: %g" % (epoch, val_loss["loss"])
                    )

    def save_model(self, path="value_model.h5"):
        self.model.save_weights(path)
        if self.config["n_gm"] > 0:
            self.gm_model.save_weights(path.replace(".h5", "_gm.h5"))

    def load_model(self, path):
        self.model.load_weights_after_init(path)
        if self.config["n_gm"] > 0:
            self.gm_model.load_weights_after_init(path.replace(".h5", "_gm.h5"))
