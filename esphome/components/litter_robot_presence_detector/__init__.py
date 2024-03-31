import esphome.codegen as cg
import esphome.config_validation as cv
from esphome.const import CONF_ID

DEPENDENCIES = ["esp32_camera"]
AUTO_LOAD = ["sensor"]

litter_robot_presence_detector_ns = cg.esphome_ns.namespace("litter_robot_presence_detector")
LitterRobotPresenceDetectorConstructor = litter_robot_presence_detector_ns.class_("LitterRobotPresenceDetector", cg.PollingComponent)

MULTI_CONF = True
CONFIG_SCHEMA = cv.Schema(
    {
        cv.GenerateID(): cv.declare_id(LitterRobotPresenceDetectorConstructor),
    }
).extend(cv.COMPONENT_SCHEMA)

async def to_code(config):
    var = cg.new_Pvariable(config[CONF_ID])
    await cg.register_component(var, config)