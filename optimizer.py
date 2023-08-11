import tensorflow as tf
from tensorflow import keras


class MAMGD(keras.optimizers.Optimizer):
    def __init__(
        self,
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999,
        k=0.00001,
        epsilon=1e-7,
        weight_decay=None,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        jit_compile=True,
        name="MAMGD",
        **kwargs
    ):
        super().__init__(
            name=name,
            weight_decay=weight_decay,
            clipnorm=clipnorm,
            clipvalue=clipvalue,
            global_clipnorm=global_clipnorm,
            use_ema=use_ema,
            ema_momentum=ema_momentum,
            ema_overwrite_frequency=ema_overwrite_frequency,
            jit_compile=jit_compile,
            **kwargs
        )
        self._learning_rate = self._build_learning_rate(learning_rate)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.k = k
        self.epsilon = epsilon

    def build(self, var_list):
        super().build(var_list)
        if hasattr(self, "_built") and self._built:
            return
        self._momentums = []
        self._accumulators = []
        self._var = []
        self._grad = []
        for var in var_list:
            self._momentums.append(
                self.add_variable_from_reference(
                    model_variable=var, variable_name="v"
                )
            )
            self._accumulators.append(
                self.add_variable_from_reference(
                    model_variable=var, variable_name="R"
                )
            )
            self._var.append(
                self.add_variable_from_reference(
                    model_variable=var, variable_name="var"
                )
            )
            self._grad.append(
                self.add_variable_from_reference(
                    model_variable=var, variable_name="G"
                )
            )
        self._built = True

    def update_step(self, gradient, variable):

      lr = tf.cast(self.learning_rate, variable.dtype)
      local_step = tf.cast(self.iterations + 1, variable.dtype)
      beta_1 = tf.cast(self.beta_1 * tf.exp(-self.k * (local_step - 1)), variable.dtype)
      beta_2 = tf.cast(self.beta_2 * tf.exp(-self.k * (local_step - 1)), variable.dtype)
      beta_1_power = tf.pow(beta_1, local_step)
      beta_2_power = tf.pow(beta_2, local_step)

      var_key = self._var_key(variable)
      v = self._momentums[self._index_dict[var_key]]
      R = self._accumulators[self._index_dict[var_key]]
      var = self._var[self._index_dict[var_key]]
      G = self._grad[self._index_dict[var_key]]

      alpha = lr * (1 - beta_2_power) / (1 - beta_1_power)

      v.assign(beta_1 * v + (1 - beta_1) * gradient)
      R.assign(beta_2 * R + (1 - beta_2) * gradient**2)
      var0 = var
      G0 = G
      var.assign(variable)
      G.assign(gradient)
      variable.assign_sub((v * alpha) / (tf.sqrt(tf.square((gradient - G0) / ((variable - var0) + self.epsilon)) + R) + self.epsilon))

    def get_config(self):
        config = super().get_config()

        config.update(
            {
                "learning_rate": self._serialize_hyperparameter(
                    self._learning_rate
                ),
                "beta_1": self.beta_1,
                "beta_2": self.beta_2,
                "k": self.k,
                "epsilon": self.epsilon,
            }
        )
        return config
