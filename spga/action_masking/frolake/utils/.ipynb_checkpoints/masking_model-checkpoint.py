"""
Models used for action masking

Resources:
---------
https://github.com/ray-project/ray/blob/master/rllib/examples/models/action_mask_model.py
"""

from gym.spaces import Box
from ray.rllib.models.tf.fcnet import FullyConnectedNetwork
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.models.tf.tf_modelv2 import TFModelV2

tf1, tf, tfv = try_import_tf()


class ActionMaskModel(TFModelV2):
    """
    Used for PPO action masking
    
    """
    def __init__(self, 
                 obs_space, 
                 action_space, 
                 num_outputs, 
                 model_config, 
                 name, 
                 true_obs_shape=(16,), # VALUE CHANGED BASED ON ENVIRONMENT (cpole = pos, vel)
                 **kwargs
    ):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)

        self.internal_model = FullyConnectedNetwork(
            Box(-1, 1, shape=true_obs_shape),
            action_space,
            num_outputs,
            model_config,
            name + "_internal",
        )


    def forward(self, input_dict, state, seq_lens):
        # Extract the available actions tensor from the observation.
        action_mask = input_dict["obs"]["action_mask"]

        # Compute the unmasked logits.
        logits, _ = self.internal_model({"obs": input_dict["obs"]["state"]})

        # Convert action_mask into a [0.0 || -inf]-type mask.
        inf_mask = tf.maximum(tf.math.log(action_mask), tf.float32.min)

        # Return masked logits.
        return logits + inf_mask, state

    def value_function(self):
        return self.internal_model.value_function()
    
    
class ActionMaskModel_8x8(TFModelV2):
    """
    Used for PPO action masking
    
    """
    def __init__(self, 
                 obs_space, 
                 action_space, 
                 num_outputs, 
                 model_config, 
                 name, 
                 true_obs_shape=(64,), # VALUE CHANGED BASED ON ENVIRONMENT (cpole = pos, vel)
                 **kwargs
    ):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)

        self.internal_model = FullyConnectedNetwork(
            Box(-1, 1, shape=true_obs_shape),
            action_space,
            num_outputs,
            model_config,
            name + "_internal",
        )


    def forward(self, input_dict, state, seq_lens):
        # Extract the available actions tensor from the observation.
        action_mask = input_dict["obs"]["action_mask"]

        # Compute the unmasked logits.
        logits, _ = self.internal_model({"obs": input_dict["obs"]["state"]})

        # Convert action_mask into a [0.0 || -inf]-type mask.
        inf_mask = tf.maximum(tf.math.log(action_mask), tf.float32.min)

        # Return masked logits.
        return logits + inf_mask, state

    def value_function(self):
        return self.internal_model.value_function()
    
    
    
class ActionMaskModel_16x16(TFModelV2):
    """
    Used for PPO action masking
    
    """
    def __init__(self, 
                 obs_space, 
                 action_space, 
                 num_outputs, 
                 model_config, 
                 name, 
                 true_obs_shape=(256,), # VALUE CHANGED BASED ON ENVIRONMENT (cpole = pos, vel)
                 **kwargs
    ):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)

        self.internal_model = FullyConnectedNetwork(
            Box(-1, 1, shape=true_obs_shape),
            action_space,
            num_outputs,
            model_config,
            name + "_internal",
        )


    def forward(self, input_dict, state, seq_lens):
        # Extract the available actions tensor from the observation.
        action_mask = input_dict["obs"]["action_mask"]

        # Compute the unmasked logits.
        logits, _ = self.internal_model({"obs": input_dict["obs"]["state"]})

        # Convert action_mask into a [0.0 || -inf]-type mask.
        inf_mask = tf.maximum(tf.math.log(action_mask), tf.float32.min)

        # Return masked logits.
        return logits + inf_mask, state

    def value_function(self):
        return self.internal_model.value_function()
    
    
class ActionMaskModel_32x32(TFModelV2):
    """
    Used for PPO action masking
    
    """
    def __init__(self, 
                 obs_space, 
                 action_space, 
                 num_outputs, 
                 model_config, 
                 name, 
                 true_obs_shape=(1024,), # VALUE CHANGED BASED ON ENVIRONMENT (cpole = pos, vel)
                 **kwargs
    ):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)

        self.internal_model = FullyConnectedNetwork(
            Box(-1, 1, shape=true_obs_shape),
            action_space,
            num_outputs,
            model_config,
            name + "_internal",
        )


    def forward(self, input_dict, state, seq_lens):
        # Extract the available actions tensor from the observation.
        action_mask = input_dict["obs"]["action_mask"]

        # Compute the unmasked logits.
        logits, _ = self.internal_model({"obs": input_dict["obs"]["state"]})

        # Convert action_mask into a [0.0 || -inf]-type mask.
        inf_mask = tf.maximum(tf.math.log(action_mask), tf.float32.min)

        # Return masked logits.
        return logits + inf_mask, state

    def value_function(self):
        return self.internal_model.value_function()
    