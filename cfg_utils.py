import habitat
import numpy as np

    
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