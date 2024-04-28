import esphome.codegen as cg
import esphome.config_validation as cv
from esphome.const import CONF_ID
from esphome.components import esp32, binary_sensor, sensor
from esphome.const import (
    CONF_SENSOR_ID
)

DEPENDENCIES = ["esp32_camera"]
AUTO_LOAD = ["binary_sensor", "md5"]

CONF_DEBUG_CAMERA_IMAGE = "debug_camera_image"

litter_robot_presence_detector_ns = cg.esphome_ns.namespace("litter_robot_presence_detector")
LitterRobotPresenceDetectorConstructor = litter_robot_presence_detector_ns.class_("LitterRobotPresenceDetector", cg.Component, binary_sensor.BinarySensor)

# MULTI_CONF = True

CONFIG_SCHEMA = (
  binary_sensor.binary_sensor_schema(LitterRobotPresenceDetectorConstructor)
  .extend(
    {
        cv.GenerateID(): cv.declare_id(LitterRobotPresenceDetectorConstructor),
        # cv.Required(CONF_SENSOR_ID): cv.use_id(sensor.Sensor)
        cv.Optional(CONF_DEBUG_CAMERA_IMAGE, default=False): cv.boolean
    }
  )
  .extend(cv.COMPONENT_SCHEMA)
)

async def to_code(config):
    var = await binary_sensor.new_binary_sensor(config)
    # cg.new_Pvariable(config[CONF_ID])
    await cg.register_component(var, config)

    esp32.add_idf_component(
        name="esp-tflite-micro",
        repo="https://github.com/espressif/esp-tflite-micro",
    )

    # inferrence could take a long time, set Watchdog timeout to 10s
    esp32.add_idf_sdkconfig_option("CONFIG_ESP_TASK_WDT_TIMEOUT_S", 20)

    cg.add_build_flag("-DTF_LITE_STATIC_MEMORY")
    cg.add_build_flag("-DTF_LITE_DISABLE_X86_NEON")
    cg.add_build_flag("-DCONFIG_NN_OPTIMIZATIONS")

    if config[CONF_DEBUG_CAMERA_IMAGE]:
        cg.add_define("DEBUG_CAMERA_IMAGE", 1)
