import os
import time
from typing import Any, List, Optional

import attr
import habitat_sim
import numpy as np
from gym import spaces

from utils.cube2equi import single_c2e
import map_utils

from habitat.config import Config
from habitat.core.dataset import Dataset, Episode
from habitat.core.embodied_task import (EmbodiedTask, Measure,
                                        SimulatorTaskAction)
from habitat.core.logging import logger
from habitat.core.registry import registry
from habitat.core.simulator import (AgentState, DepthSensor, RGBSensor, Sensor,
                                    SensorTypes, Simulator)
from habitat.core.utils import not_none_validator, try_cv2_import
from habitat.tasks.nav.nav import (DistanceToGoal, NavigationEpisode,
                                   NavigationGoal, NavigationTask, Success)
from habitat.tasks.nav.object_nav_task import (ObjectGoal,
                                               ObjectGoalNavEpisode,
                                               ObjectViewLocation)
from habitat.tasks.utils import cartesian_to_polar
from habitat.utils.geometry_utils import (quaternion_from_coeff,
                                          quaternion_rotate_vector)
from habitat.utils.visualizations import fog_of_war, maps

cv2 = try_cv2_import()

def make_panoramic(left, front, right, back):
    return np.concatenate([left, front, right, back],1)
MAP_THICKNESS_SCALAR: int = 4096
@registry.register_measure
class CustomTopDownMap(Measure):
    r"""Top Down Map measure"""

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self._sim = sim
        self._config = config
        self._grid_delta = config.MAP_PADDING
        self._step_count = None
        self._map_resolution = config.MAP_RESOLUTION
        self._ind_x_min = None
        self._ind_x_max = None
        self._ind_y_min = None
        self._ind_y_max = None
        self._previous_xy_location = None
        self._top_down_map = None
        self._shortest_path_points = None
        self.line_thickness = int(
            np.round(self._map_resolution * 2 / MAP_THICKNESS_SCALAR)
        )
        self.point_padding = 2 * int(
            np.ceil(self._map_resolution / MAP_THICKNESS_SCALAR)
        )
        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "custom_top_down_map"

    def get_original_map(self):
        top_down_map = maps.get_topdown_map_from_sim(
            self._sim,
            map_resolution=self._map_resolution,
            draw_border=self._config.DRAW_BORDER,
        )

        if self._config.FOG_OF_WAR.DRAW:
            self._fog_of_war_mask = np.zeros_like(top_down_map)
        else:
            self._fog_of_war_mask = None

        return top_down_map

    def _draw_point(self, position, point_type):
        t_x, t_y = maps.to_grid(
            position[2],
            position[0],
            self._top_down_map.shape[0:2],
            sim=self._sim,
        )
        self._top_down_map[
            t_x - self.point_padding : t_x + self.point_padding + 1,
            t_y - self.point_padding : t_y + self.point_padding + 1,
        ] = point_type

    def _draw_goals_view_points(self, episode):
        if self._config.DRAW_VIEW_POINTS:
            for goal in episode.goals:
                try:
                    if goal.view_points is not None:
                        for view_point in goal.view_points:
                            self._draw_point(
                                view_point.agent_state.position,
                                maps.MAP_VIEW_POINT_INDICATOR,
                            )
                except AttributeError:
                    pass

    def _draw_goals_positions(self, episode):
        if self._config.DRAW_GOAL_POSITIONS:

            for goal in episode.goals:
                try:
                    self._draw_point(
                        goal.position, maps.MAP_TARGET_POINT_INDICATOR
                    )
                except AttributeError:
                    pass

                
    def _draw_goals_aabb(self, episode):
        if self._config.DRAW_GOAL_AABBS:
            for goal in episode.goals:
                try:
                    sem_scene = self._sim.semantic_annotations()
                    object_id = goal.object_id
                    assert int(
                        sem_scene.objects[object_id].id.split("_")[-1]
                    ) == int(
                        goal.object_id
                    ), f"Object_id doesn't correspond to id in semantic scene objects dictionary for episode: {episode}"

                    center = sem_scene.objects[object_id].aabb.center
                    x_len, _, z_len = (
                        sem_scene.objects[object_id].aabb.sizes / 2.0
                    )
                    # Nodes to draw rectangle
                    corners = [
                        center + np.array([x, 0, z])
                        for x, z in [
                            (-x_len, -z_len),
                            (-x_len, z_len),
                            (x_len, z_len),
                            (x_len, -z_len),
                            (-x_len, -z_len),
                        ]
                    ]

                    map_corners = [
                        maps.to_grid(
                            p[2],
                            p[0],
                            self._top_down_map.shape[0:2],
                            sim=self._sim,
                        )
                        for p in corners
                    ]

                    maps.draw_path(
                        self._top_down_map,
                        map_corners,
                        maps.MAP_TARGET_BOUNDING_BOX,
                        self.line_thickness,
                    )
                except AttributeError:
                    pass
    
    def _draw_regions_positions(self):
        if True:
            for reg in self.regions:
                try:
                    self._draw_point(
                        self.regions[reg]["center"], maps.MAP_TARGET_POINT_INDICATOR
                    )
                except AttributeError:
                    pass
                
    def _draw_doors_positions(self):
        if True:
            for door in self.doors:
                try:
                    self._draw_point(
                        self.doors[door]["center"], maps.MAP_TARGET_POINT_INDICATOR
                    )
                except AttributeError:
                    pass
    
    def _draw_objects_positions(self):
        if True:
            for obj in self.objects:
                try:
                    self._draw_point(
                        self.objects[obj]["center"], maps.MAP_TARGET_POINT_INDICATOR
                    )
                except AttributeError:
                    pass
                
    def _draw_objects_aabb(self):
        if True:
            for obj in self.objects:
                try:
                    center = self.objects[obj]["center"]
                    x_len, _, z_len = (
                        self.objects[obj]["dims"] / 2.0
                    )
                    # Nodes to draw rectangle
                    corners = [
                        center + np.array([x, 0, z])
                        for x, z in [
                            (-x_len, -z_len),
                            (-x_len, z_len),
                            (x_len, z_len),
                            (x_len, -z_len),
                            (-x_len, -z_len),
                        ]
                    ]

                    map_corners = [
                        maps.to_grid(
                            p[2],
                            p[0],
                            self._top_down_map.shape[0:2],
                            sim=self._sim,
                        )
                        for p in corners
                    ]

                    #Fill aabb with sem color
                    if self._config.FILL_AABBS:
                        map_utils.draw_rectangle(
                            self._top_down_map,
                            [map_corners[0],map_corners[2]],
                            self.objects[obj]["category"]
                        )

                    maps.draw_path(
                        self._top_down_map,
                        map_corners,
                        maps.MAP_TARGET_BOUNDING_BOX,
                        self.line_thickness,
                    )
                except AttributeError:
                    pass
                
    def _draw_shortest_path(
        self, episode: Episode, agent_position: AgentState
    ):
        if self._config.DRAW_SHORTEST_PATH:
            self._shortest_path_points = (
                self._sim.get_straight_shortest_path_points(
                    agent_position, episode.goals[0].position
                )
            )
            self._shortest_path_points = [
                maps.to_grid(
                    p[2], p[0], self._top_down_map.shape[0:2], sim=self._sim
                )
                for p in self._shortest_path_points
            ]
            maps.draw_path(
                self._top_down_map,
                self._shortest_path_points,
                maps.MAP_SHORTEST_PATH_COLOR,
                self.line_thickness,
            )

    def reset_metric(self, episode, *args: Any, **kwargs: Any):
        self._step_count = 0
        self._metric = None
        self._top_down_map = self.get_original_map()
        agent_position = self._sim.get_agent_state().position
        a_x, a_y = maps.to_grid(
            agent_position[2],
            agent_position[0],
            self._top_down_map.shape[0:2],
            sim=self._sim,
        )
        self._previous_xy_location = (a_y, a_x)

        self.update_fog_of_war_mask(np.array([a_x, a_y]))
        
        scene = self._sim.semantic_annotations()
        lev = scene.levels[0]
        self.regions = {}
        self.objects = {}
        self.doors = {}
        
        for region in lev.regions:
            self.regions[region.id] = {
                "category":region.category.name(),
                "center":region.aabb.center,
                "dims":region.aabb.sizes,
                "objects":region.objects
            }
              
            for obj in region.objects:
                if obj.category.name() is "door":
                    self.doors[obj.id] = {
                        "region":region.id,
                        "category":obj.category.name(),
                        "center":obj.aabb.center,
                        "dims":obj.aabb.sizes
                    }
                if obj.category.name() not in ["picture", "wall", "ceiling", "misc", "door", "objects", "floor", "void", "window", "column", "beam", "lighting"]:
                    self.objects[obj.id] = {
                        "region":region.id,
                        "category":obj.category.name(),
                        "center":obj.aabb.center,
                        "dims":obj.aabb.sizes
                    }
        
        # draw source and target parts last to avoid overlap
        self._draw_goals_view_points(episode)
        self._draw_goals_aabb(episode)
        self._draw_goals_positions(episode)

        self._draw_regions_positions()
        
        self._draw_objects_aabb()
        self._draw_objects_positions()
        
        self._draw_doors_positions()
        
        self._draw_shortest_path(episode, agent_position)

        if self._config.DRAW_SOURCE:
            self._draw_point(
                episode.start_position, maps.MAP_SOURCE_POINT_INDICATOR
            )

    def update_metric(self, episode, action, *args: Any, **kwargs: Any):
        self._step_count += 1
        house_map, map_agent_x, map_agent_y = self.update_map(
            self._sim.get_agent_state().position
        )

        self._metric = {
            "map": house_map,
            "fog_of_war_mask": self._fog_of_war_mask,
            "agent_map_coord": (map_agent_x, map_agent_y),
            "agent_angle": self.get_polar_angle(),
        }

    def get_polar_angle(self):
        agent_state = self._sim.get_agent_state()
        # quaternion is in x, y, z, w format
        ref_rotation = agent_state.rotation

        heading_vector = quaternion_rotate_vector(
            ref_rotation.inverse(), np.array([0, 0, -1])
        )

        phi = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
        z_neg_z_flip = np.pi
        return np.array(phi) + z_neg_z_flip

    def update_map(self, agent_position):
        a_x, a_y = maps.to_grid(
            agent_position[2],
            agent_position[0],
            self._top_down_map.shape[0:2],
            sim=self._sim,
        )
        # Don't draw over the source point
        if self._top_down_map[a_x, a_y] != maps.MAP_SOURCE_POINT_INDICATOR:
            color = 10 + min(
                self._step_count * 245 // self._config.MAX_EPISODE_STEPS, 245
            )

            thickness = self.line_thickness
            cv2.circle(
                self._top_down_map,
                (a_y, a_x),
                2,
                [0,0,0]
            )
            cv2.line(
                self._top_down_map,
                self._previous_xy_location,
                (a_y, a_x),
                color,
                thickness=thickness,
            )

        self.update_fog_of_war_mask(np.array([a_x, a_y]))

        self._previous_xy_location = (a_y, a_x)
        return self._top_down_map, a_x, a_y

    def update_fog_of_war_mask(self, agent_position):
        if self._config.FOG_OF_WAR.DRAW:
            self._fog_of_war_mask = fog_of_war.reveal_fog_of_war(
                self._top_down_map,
                self._fog_of_war_mask,
                agent_position,
                self.get_polar_angle(),
                fov=self._config.FOG_OF_WAR.FOV,
                max_line_len=self._config.FOG_OF_WAR.VISIBILITY_DIST
                / maps.calculate_meters_per_pixel(
                    self._map_resolution, sim=self._sim
                ),
            )

            
@registry.register_sensor(name="CustomObjectSensor")
class CustomObjectGoalSensor(Sensor):
    
    def __init__(
        self, sim, config: Config, dataset: Dataset, *args: Any, **kwargs: Any
    ):
        self._sim = sim
        self._dataset = dataset
        use_depth = 'DEPTH_SENSOR' in self._sim.config.AGENT_0.SENSORS
        use_rgb = 'RGB_SENSOR' in self._sim.config.AGENT_0.SENSORS
        self.channel = use_depth + 3 * use_rgb
        self.height = self._sim.config.RGB_SENSOR.HEIGHT if use_rgb else self._sim.config.DEPTH_SENSOR.HEIGHT
        self.width = self._sim.config.RGB_SENSOR.WIDTH if use_rgb else self._sim.config.DEPTH_SENSOR.WIDTH
        self.curr_episode_id = -1
        self.curr_scene_id = ''
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "objectgoal"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.COLOR

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        sensor_shape = (1,)
        if self.config.GOAL_SPEC == 'OBJECT_IMG':
            return spaces.Box(low=0, high=1.0, shape=(self.height, self.width*4, self.channel+1), dtype=np.float32)
        max_value = (self.config.GOAL_SPEC_MAX_VAL - 1,)
        if self.config.GOAL_SPEC == "TASK_CATEGORY_ID":
            max_value = max(
                self._dataset.category_to_task_category_id.values()
            )

        return spaces.Box(
            low=0, high=max_value, shape=sensor_shape, dtype=np.int64
        )

    def get_observation(
        self,
        observations,
        *args: Any,
        episode: ObjectGoalNavEpisode,
        **kwargs: Any,
    ) -> Optional[int]:
        if self.config.GOAL_SPEC == "TASK_CATEGORY_ID":
            if len(episode.goals) == 0:
                logger.error(
                    f"No goal specified for episode {episode.episode_id}."
                )
                return None
            if not isinstance(episode.goals[0], ObjectGoal):
                logger.error(
                    f"First goal should be ObjectGoal, episode {episode.episode_id}."
                )
                return None
            category_name = episode.object_category
            return np.array(
                [self._dataset.category_to_task_category_id[category_name]],
                dtype=np.int64,
            )
        elif self.config.GOAL_SPEC == "OBJECT_ID":
            return np.array([episode.goals[0].object_name_id], dtype=np.int64)
        elif self.config.GOAL_SPEC == "OBJECT_IMG":
            episode_id = episode.episode_id
            scene_id = episode.scene_id
            if (self.curr_episode_id != episode_id) or (self.curr_scene_id != scene_id):
                viewpoint = episode.goals[0].view_points[0].agent_state
                obs = self._sim.get_observations_at(viewpoint.position, viewpoint.rotation)
                rgb_array = make_panoramic(obs['rgb_left'], obs['rgb'], obs['rgb_right'], obs['rgb_back'])/255.
                depth_array = make_panoramic(obs['depth_left'], obs['depth'], obs['depth_right'], obs['depth_back'])

                #rgb_array = obs['rgb']/255.0
                #depth_array = obs['depth']
                category_array = np.ones_like(depth_array) * self._dataset.category_to_task_category_id[episode.object_category]
                self.goal_obs = np.concatenate([rgb_array, depth_array, category_array],2)
                self.curr_episode_id = episode_id
                self.curr_scene_id = scene_id
            return self.goal_obs
        else:
            raise RuntimeError(
                "Wrong GOAL_SPEC specified for ObjectGoalSensor."
            )

@registry.register_sensor(name="PanoramicPartRGBSensor")
class PanoramicPartRGBSensor(RGBSensor):
    def __init__(self, config, **kwargs: Any):
        self.config = config
        self.angle = config.ANGLE
        self.sim_sensor_type = habitat_sim.SensorType.COLOR
        super().__init__(config=config)


    # Defines the name of the sensor in the sensor suite dictionary
    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "rgb_" + self.angle

    # Defines the size and range of the observations of the sensor
    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=0,
            high=255,
            shape=(self.config.HEIGHT, self.config.WIDTH, 3),
            dtype=np.uint8,
        )
    def get_observation(self, obs, *args: Any, **kwargs: Any) -> Any:
        obs = obs.get(self.uuid, None)
        return obs[:,:,:3]

@registry.register_sensor(name="PanoramicPartDepthSensor")
class PanoramicPartDepthSensor(DepthSensor):
    def __init__(self, config):
        self.sim_sensor_type = habitat_sim.SensorType.DEPTH
        self.angle = config.ANGLE

        if config.NORMALIZE_DEPTH:
            self.min_depth_value = 0
            self.max_depth_value = 1
        else:
            self.min_depth_value = config.MIN_DEPTH
            self.max_depth_value = config.MAX_DEPTH

        super().__init__(config=config)

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=self.min_depth_value,
            high=self.max_depth_value,
            shape=(self.config.HEIGHT, self.config.WIDTH, 1),
            dtype=np.float32)

    # Defines the name of the sensor in the sensor suite dictionary
    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "depth_" + self.angle

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.DEPTH

    # This is called whenver reset is called or an action is taken
    def get_observation(self, obs,*args: Any, **kwargs: Any):
        obs = obs.get(self.uuid, None)
        if isinstance(obs, np.ndarray):
            obs = np.clip(obs, self.config.MIN_DEPTH, self.config.MAX_DEPTH)

            obs = np.expand_dims(
                obs, axis=2
            )  # make depth observation a 3D array
        else:
            obs = obs.clamp(self.config.MIN_DEPTH, self.config.MAX_DEPTH)

            obs = obs.unsqueeze(-1)

        if self.config.NORMALIZE_DEPTH:
            # normalize depth observation to [0, 1]
            obs = (obs - self.config.MIN_DEPTH) / (
                self.config.MAX_DEPTH - self.config.MIN_DEPTH
            )

        return obs

@registry.register_sensor(name="PanoramicRGBSensor")
class PanoramicRGBSensor(Sensor):
    def __init__(self, sim, config, **kwargs: Any):
        self.sim = sim
        super().__init__(config=config)
        self.config = config

    # Defines the name of the sensor in the sensor suite dictionary
    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "panoramic_rgb"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.COLOR
    # Defines the size and range of the observations of the sensor
    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=0,
            high=255,
            shape=(self.config.HEIGHT, self.config.WIDTH, 3),
            dtype=np.uint8,
        )
    # This is called whenver reset is called or an action is taken
    def get_observation(self, observations,*args: Any, **kwargs: Any):
        cube = {
            'F': observations['rgb'].transpose(2,0,1),
            'R': observations['rgb_right'].transpose(2,0,1),
            'B': observations['rgb_back'].transpose(2,0,1),
            'L': observations['rgb_left'].transpose(2,0,1),
            'U': observations['rgb_top'].transpose(2,0,1),
            'D': observations['rgb_bottom'].transpose(2,0,1)
        }
        out = single_c2e('dict', cube, 720, 360).transpose(1,2,0)
        return cv2.resize(out, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
    
@registry.register_sensor(name="CubeRGBSensor")
class CubeRGBSensor(Sensor):
    def __init__(self, sim, config, **kwargs: Any):
        self.sim = sim
        super().__init__(config=config)
        self.config = config

    # Defines the name of the sensor in the sensor suite dictionary
    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "cube_rgb"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.COLOR
    # Defines the size and range of the observations of the sensor
    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=0,
            high=255,
            shape=(self.config.HEIGHT, self.config.WIDTH, 3),
            dtype=np.uint8,
        )
    # This is called whenver reset is called or an action is taken
    def get_observation(self, observations,*args: Any, **kwargs: Any):
        out = make_panoramic(
            observations['rgb_left'],
            observations['rgb'],
            observations['rgb_right'],
            observations['rgb_back']
        )
        return cv2.resize(out, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)

@registry.register_sensor(name="PanoramicDepthSensor")
class PanoramicDepthSensor(DepthSensor):
    def __init__(self, sim, config, **kwargs: Any):
        self.sim_sensor_type = habitat_sim.SensorType.DEPTH

        if config.NORMALIZE_DEPTH:
            self.min_depth_value = 0
            self.max_depth_value = 1
        else:
            self.min_depth_value = config.MIN_DEPTH
            self.max_depth_value = config.MAX_DEPTH

        super().__init__(config=config)

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=self.min_depth_value,
            high=self.max_depth_value,
            shape=(self.config.HEIGHT, self.config.WIDTH*4, 1),
            dtype=np.float32)

    # Defines the name of the sensor in the sensor suite dictionary
    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "panoramic_depth"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.DEPTH

    # This is called whenver reset is called or an action is taken
    def get_observation(self, observations,*args: Any, **kwargs: Any):
        return make_panoramic(observations['depth_left'],observations['depth'],observations['depth_right'],observations['depth_back'])

@registry.register_sensor(name="CubeImageGoalSensor")
class CubeImageGoalSensor(Sensor):    
    def __init__(
        self, *args: Any, sim: Simulator, config: Config, **kwargs: Any
    ):
        self._sim = sim
        self._rgb_sensor_uuid = 'rgb'
        self._current_episode_id: Optional[str] = None
        self._current_image_goal = None
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "cube_image_goal"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.COLOR

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return self._sim.sensor_suite.observation_spaces.spaces[
            self._rgb_sensor_uuid
        ]

    def _get_pointnav_episode_image_goal(self, episode: NavigationEpisode):
        goal_position = np.array(episode.goals[0].position, dtype=np.float32)
        # to be sure that the rotation is the same for the same episode_id
        # since the task is currently using pointnav Dataset.
        seed = abs(hash(episode.episode_id)) % (2 ** 32)
        rng = np.random.RandomState(seed)
        angle = rng.uniform(0, 2 * np.pi)
        source_rotation = [0, np.sin(angle / 2), 0, np.cos(angle / 2)]
        goal_observations = self._sim.get_observations_at(
            position=goal_position.tolist(), rotation=source_rotation
        )
        output_image_goal = make_panoramic(
            goal_observations['rgb_left'],
            goal_observations['rgb'],
            goal_observations['rgb_right'],
            goal_observations['rgb_back']
        )
        return cv2.resize(output_image_goal, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
        
    def get_observation(
        self,
        *args: Any,
        observations,
        episode: NavigationEpisode,
        **kwargs: Any,
    ):
        episode_uniq_id = f"{episode.scene_id} {episode.episode_id}"
        if episode_uniq_id == self._current_episode_id:
            return self._current_image_goal

        self._current_image_goal = self._get_pointnav_episode_image_goal(
            episode
        )
        self._current_episode_id = episode_uniq_id

        return self._current_image_goal
    
@registry.register_sensor(name="PanoramicImageGoalSensor")
class PanoramicImageGoalSensor(Sensor):    
    def __init__(
        self, *args: Any, sim: Simulator, config: Config, **kwargs: Any
    ):
        self._sim = sim
        self._rgb_sensor_uuid = 'rgb'
        self._current_episode_id: Optional[str] = None
        self._current_image_goal = None
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "panoramic_image_goal"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.COLOR

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return self._sim.sensor_suite.observation_spaces.spaces[
            self._rgb_sensor_uuid
        ]

    def _get_pointnav_episode_image_goal(self, episode: NavigationEpisode):
        goal_position = np.array(episode.goals[0].position, dtype=np.float32)
        # to be sure that the rotation is the same for the same episode_id
        # since the task is currently using pointnav Dataset.
        seed = abs(hash(episode.episode_id)) % (2 ** 32)
        rng = np.random.RandomState(seed)
        angle = rng.uniform(0, 2 * np.pi)
        source_rotation = [0, np.sin(angle / 2), 0, np.cos(angle / 2)]
        goal_observations = self._sim.get_observations_at(
            position=goal_position.tolist(), rotation=source_rotation
        )
        
        cube = {
            'F': goal_observations['rgb'].transpose(2,0,1),
            'R': goal_observations['rgb_right'].transpose(2,0,1),
            'B': goal_observations['rgb_back'].transpose(2,0,1),
            'L': goal_observations['rgb_left'].transpose(2,0,1),
            'U': goal_observations['rgb_top'].transpose(2,0,1),
            'D': goal_observations['rgb_bottom'].transpose(2,0,1)
        }
        output_image_goal = single_c2e('dict', cube, 720, 360).transpose(1,2,0)
        return cv2.resize(output_image_goal, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
        
    def get_observation(
        self,
        *args: Any,
        observations,
        episode: NavigationEpisode,
        **kwargs: Any,
    ):
        episode_uniq_id = f"{episode.scene_id} {episode.episode_id}"
        if episode_uniq_id == self._current_episode_id:
            return self._current_image_goal

        self._current_image_goal = self._get_pointnav_episode_image_goal(
            episode
        )
        self._current_episode_id = episode_uniq_id

        return self._current_image_goal

@registry.register_measure(name='Success_woSTOP')
class Success_woSTOP(Success):
    r"""Whether or not the agent succeeded at its task

    This measure depends on DistanceToGoal measure.
    """

    cls_uuid: str = "success"

    def update_metric(
        self, *args: Any, episode, task: EmbodiedTask, **kwargs: Any
    ):
        distance_to_target = task.measurements.measures[
            DistanceToGoal.cls_uuid
        ].get_metric()

        if (
           distance_to_target < self._config.SUCCESS_DISTANCE
        ):
            self._metric = 1.0
        else:
            self._metric = 0.0
