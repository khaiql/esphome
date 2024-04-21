#pragma once

#ifdef USE_ESP32

#include <freertos/FreeRTOS.h>
#include <freertos/semphr.h>

#include "esphome/core/component.h"
#include "esphome/core/application.h"
#include "esphome/components/esp32_camera/esp32_camera.h"
#include "esphome/components/sensor/sensor.h"

#include <tensorflow/lite/core/c/common.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/micro/micro_mutable_op_resolver.h>

namespace esphome {
namespace litter_robot_presence_detector {

constexpr uint8_t PRESENCE = 1;
constexpr uint8_t EMPTY = 0;

class LitterRobotPresenceDetector : public PollingComponent, public sensor::Sensor {
 public:
  // constructor
  LitterRobotPresenceDetector() : PollingComponent(10000) {}

  void on_shutdown() override;
  void setup() override;
  void update() override;
  void dump_config() override;
  float get_setup_priority() const override;

 protected:
  std::shared_ptr<esphome::esp32_camera::CameraImage> wait_for_image_();
  SemaphoreHandle_t semaphore_;
  std::shared_ptr<esphome::esp32_camera::CameraImage> image_;
  uint8_t *tensor_arena_{nullptr};
  const tflite::Model *model{nullptr};
  // tflite::MicroResourceVariables *mrv_{nullptr};
  tflite::MicroInterpreter *interpreter{nullptr};
  TfLiteTensor *input{nullptr};
  TfLiteTensor *output{nullptr};

  bool inferring_{false};
  bool setup_model();
  bool register_preprocessor_ops(tflite::MicroMutableOpResolver<9> &micro_op_resolver);
  bool start_infer(std::shared_ptr<esphome::esp32_camera::CameraImage> image);
  bool is_cat_presence();
};
}  // namespace litter_robot_presence_detector
}  // namespace esphome

#endif
