from legged_gym import LEGGED_GYM_ROOT_DIR
import os, sys

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, export_policy_as_jit, task_registry, Logger, get_load_path, class_to_dict
from legged_gym.algorithm.mlp_encoder import MLP_Encoder
from legged_gym.algorithm.actor_critic import ActorCritic

import numpy as np
import torch
import copy

class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = "elu"  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        orthogonal_init = False

class mlp_encoder:
        output_detach = True
        num_input_dim = 30 * 10
        num_output_dim = 3 + 32
        hidden_dims = [256, 128]
        activation = "elu"
        orthogonal_init = False

def export_policy_as_onnx(resume_path):
    loaded_dict = torch.load(resume_path)
    # encoder
    encoder_class = eval("MLP_Encoder")
    encoder = encoder_class(**class_to_dict(mlp_encoder)).to("cpu")
    encoder.load_state_dict(loaded_dict['student_encoder_state_dict'])
    encoder_path = "student_encoder.onnx"
    encoder_model = copy.deepcopy(encoder.encoder).to("cpu")
    encoder_model.eval()
    dummy_input = torch.randn(encoder.num_input_dim)
    input_names = ["nn_input"]
    output_names = ["nn_output"]
    
    torch.onnx.export(
        encoder_model,
        dummy_input,
        encoder_path,
        verbose=True,
        input_names=input_names,
        output_names=output_names,
        export_params=True,
        opset_version=13,
    )
    print("Exported encoder as onnx script to: ", encoder_path)

    # actor_critic
    actor_critic_class = eval("ActorCritic")

    actor_input_dim = 30+ 35 + 3
    critic_input_dim = 36 + 30 + 117 + 3
    actor_critic = actor_critic_class(
        actor_input_dim, critic_input_dim, 6, **class_to_dict(policy)
    ).to("cpu")
    actor_critic.load_state_dict(loaded_dict['model_state_dict'])

    # export policy as an onnx file
    policy_path = "policy.onnx"
    model = copy.deepcopy(actor_critic.actor).to("cpu")
    model.eval()

    dummy_input = torch.randn(actor_input_dim)
    input_names = ["nn_input"]
    output_names = ["nn_output"]

    torch.onnx.export(
        model,
        dummy_input,
        policy_path,
        verbose=True,
        input_names=input_names,
        output_names=output_names,
        export_params=True,
        opset_version=13,
    )
    print("Exported policy as onnx script to: ", policy_path)


if __name__ == '__main__':
    
    # check ROBOT_TYPE validity
    resume_path = "/home/syw/Downloads/model_20300.pt"
    export_policy_as_onnx(resume_path)
