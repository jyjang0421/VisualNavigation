#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import random
from typing import Type, Union

import habitat
from habitat import Config, Env, RLEnv, VectorEnv, make_dataset
import numpy as np
import os

def check_no_stop(task_config):
    if "STOP" not in task_config.TASK.POSSIBLE_ACTIONS:
        task_config.TASK.SUCCESS.TYPE = "Success_woSTOP"
    task_config.TASK.SUCCESS.SUCCESS_DISTANCE = task_config.TASK.SUCCESS_DISTANCE
    return task_config

def add_panoramic_camera(task_config):
    task_config.SIMULATOR.RGB_SENSOR_LEFT = task_config.SIMULATOR.RGB_SENSOR.clone()
    task_config.SIMULATOR.RGB_SENSOR_LEFT.TYPE = "PanoramicPartRGBSensor"
    task_config.SIMULATOR.RGB_SENSOR_LEFT.ORIENTATION = [0, 0.5 * np.pi, 0]
    task_config.SIMULATOR.RGB_SENSOR_LEFT.ANGLE = "left"
    task_config.SIMULATOR.RGB_SENSOR_RIGHT = task_config.SIMULATOR.RGB_SENSOR.clone()
    task_config.SIMULATOR.RGB_SENSOR_RIGHT.TYPE = "PanoramicPartRGBSensor"
    task_config.SIMULATOR.RGB_SENSOR_RIGHT.ORIENTATION = [0, -0.5 * np.pi, 0]
    task_config.SIMULATOR.RGB_SENSOR_RIGHT.ANGLE = "right"
    task_config.SIMULATOR.RGB_SENSOR_BACK = task_config.SIMULATOR.RGB_SENSOR.clone()
    task_config.SIMULATOR.RGB_SENSOR_BACK.TYPE = "PanoramicPartRGBSensor"
    task_config.SIMULATOR.RGB_SENSOR_BACK.ORIENTATION = [0, np.pi, 0]
    task_config.SIMULATOR.RGB_SENSOR_BACK.ANGLE = "back"
    task_config.SIMULATOR.RGB_SENSOR_TOP = task_config.SIMULATOR.RGB_SENSOR.clone()
    task_config.SIMULATOR.RGB_SENSOR_TOP.TYPE = "PanoramicPartRGBSensor"
    task_config.SIMULATOR.RGB_SENSOR_TOP.ORIENTATION = [0.5*np.pi, 0, 0]
    task_config.SIMULATOR.RGB_SENSOR_TOP.ANGLE = "top"
    task_config.SIMULATOR.RGB_SENSOR_BOTTOM = task_config.SIMULATOR.RGB_SENSOR.clone()
    task_config.SIMULATOR.RGB_SENSOR_BOTTOM.TYPE = "PanoramicPartRGBSensor"
    task_config.SIMULATOR.RGB_SENSOR_BOTTOM.ORIENTATION = [-0.5*np.pi, 0, 0]
    task_config.SIMULATOR.RGB_SENSOR_BOTTOM.ANGLE = "bottom"
    task_config.SIMULATOR.AGENT_0.SENSORS += ['RGB_SENSOR_LEFT', 'RGB_SENSOR_RIGHT', 'RGB_SENSOR_BACK','RGB_SENSOR_TOP','RGB_SENSOR_BOTTOM']
    
    task_config.TASK.PANORAMIC_RGB_SENSOR = habitat.Config()
    task_config.TASK.PANORAMIC_RGB_SENSOR.TYPE = "PanoramicRGBSensor"
    task_config.TASK.PANORAMIC_RGB_SENSOR.HEIGHT = 256
    task_config.TASK.PANORAMIC_RGB_SENSOR.WIDTH = 256
    task_config.TASK.CUBE_RGB_SENSOR = habitat.Config()
    task_config.TASK.CUBE_RGB_SENSOR.TYPE = "CubeRGBSensor"
    task_config.TASK.CUBE_RGB_SENSOR.HEIGHT = 256
    task_config.TASK.CUBE_RGB_SENSOR.WIDTH = 256
    
    task_config.TASK.PANORAMIC_IMAGEGOAL_SENSOR = habitat.Config()
    task_config.TASK.PANORAMIC_IMAGEGOAL_SENSOR.TYPE = "PanoramicImageGoalSensor"
    task_config.TASK.PANORAMIC_IMAGEGOAL_SENSOR.HEIGHT = 256
    task_config.TASK.PANORAMIC_IMAGEGOAL_SENSOR.WIDTH = 256
    task_config.TASK.CUBE_IMAGEGOAL_SENSOR = habitat.Config()
    task_config.TASK.CUBE_IMAGEGOAL_SENSOR.TYPE = "CubeImageGoalSensor"
    task_config.TASK.CUBE_IMAGEGOAL_SENSOR.HEIGHT = 256
    task_config.TASK.CUBE_IMAGEGOAL_SENSOR.WIDTH = 256
    
    return task_config
#dataset filter
def filter_fn(episode):
    if episode.info['geodesic_distance'] < 4.0 and episode.info['geodesic_distance'] > 1.5 :
        return True
    else:
        return False

def make_env_fn(
    config: Config, env_class: Type[Union[Env, RLEnv]]
) -> Union[Env, RLEnv]:
    r"""Creates an env of type env_class with specified config and rank.
    This is to be passed in as an argument when creating VectorEnv.

    Args:
        config: root exp config that has core env config node as well as
            env-specific config node.
        env_class: class type of the env to be created.

    Returns:
        env object created according to specification.
    """
    print('make-env')
    config.defrost()
    config.TASK_CONFIG = add_panoramic_camera(config.TASK_CONFIG)
    config.TASK_CONFIG = check_no_stop(config.TASK_CONFIG)
    config.freeze()
    print(config.TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS)
    print(config.TASK_CONFIG.TASK.SENSORS)
    dataset = make_dataset(
        config.TASK_CONFIG.DATASET.TYPE, config=config.TASK_CONFIG.DATASET,
    )
    dataset = dataset.filter_episodes(filter_fn)
    env = env_class(config=config, dataset=dataset)
    env.seed(config.TASK_CONFIG.SEED)
    return env

#env를 여러개를 합치기
def construct_envs(
    config: Config,
    env_class: Type[Union[Env, RLEnv]],
    workers_ignore_signals: bool = False,
) -> VectorEnv:
    r"""Create VectorEnv object with specified config and env class type.
    To allow better performance, dataset are split into small ones for
    each individual env, grouped by scenes.

    :param config: configs that contain num_processes as well as information
    :param necessary to create individual environments.
    :param env_class: class type of the envs to be created.
    :param workers_ignore_signals: Passed to :ref:`habitat.VectorEnv`'s constructor

    :return: VectorEnv object created according to specification.
    """

    num_processes = config.NUM_PROCESSES
    configs = []
    env_classes = [env_class for _ in range(num_processes)]
    
    
    dataset = make_dataset(config.TASK_CONFIG.DATASET.TYPE)
    dataset = dataset.filter_episodes(filter_fn)

    scenes = config.TASK_CONFIG.DATASET.CONTENT_SCENES
    if "*" in config.TASK_CONFIG.DATASET.CONTENT_SCENES:
        scenes = dataset.get_scenes_to_load(config.TASK_CONFIG.DATASET)
    
    if num_processes > 1:
        if len(scenes) == 0:
            raise RuntimeError(
                "No scenes to load, multiple process logic relies on being able to split scenes uniquely between processes"
            )

        if len(scenes) < num_processes:
            raise RuntimeError(
                "reduce the number of processes as there "
                "aren't enough number of scenes"
            )

        random.shuffle(scenes)

    scene_splits = [[] for _ in range(num_processes)]
    for idx, scene in enumerate(scenes):
        scene_splits[idx % len(scene_splits)].append(scene)

    assert sum(map(len, scene_splits)) == len(scenes)

    for i in range(num_processes):
        proc_config = config.clone()
        proc_config.defrost()

        task_config = proc_config.TASK_CONFIG
        task_config.SEED = task_config.SEED + i
        if len(scenes) > 0:
            task_config.DATASET.CONTENT_SCENES = scene_splits[i]

        task_config.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = (
            config.SIMULATOR_GPU_ID
        )

        task_config.SIMULATOR.AGENT_0.SENSORS = config.SENSORS
        
        proc_config.freeze()
        configs.append(proc_config)

    envs = habitat.VectorEnv(
        make_env_fn=make_env_fn,
        env_fn_args=tuple(zip(configs, env_classes)),
        workers_ignore_signals=workers_ignore_signals,
    )
    return envs

