#ifdef USE_ESP32
#include "litter_robot_presence_detector.h"
#include "esphome/core/log.h"

#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "person_detect_model_data.h"

namespace esphome {
namespace litter_robot_presence_detector {

static const char *const TAG = "litter_robot_presence_detector";
static const uint32_t MODEL_ARENA_SIZE = 300 * 1024;
static const double QUANTIZED_THESHOLD_FLOAT = 0.28578;  // Taken from the model

float LitterRobotPresenceDetector::get_setup_priority() const { return setup_priority::AFTER_CONNECTION; }

void LitterRobotPresenceDetector::on_shutdown() {
  this->inferring_ = false;
  this->image_ = nullptr;
  vSemaphoreDelete(this->semaphore_);
  this->semaphore_ = nullptr;
}

bool LitterRobotPresenceDetector::register_preprocessor_ops(tflite::MicroMutableOpResolver<7> &micro_op_resolver) {
  if (micro_op_resolver.AddConv2D() != kTfLiteOk) {
    ESP_LOGE(TAG, "failed to register ops AddConv2D");
    return false;
  }

  if (micro_op_resolver.AddMaxPool2D() != kTfLiteOk) {
    ESP_LOGE(TAG, "failed to register ops AddMaxPool2D");
    return false;
  }

  if (micro_op_resolver.AddQuantize() != kTfLiteOk) {
    ESP_LOGE(TAG, "failed to register ops AddQuantize");
    return false;
  }

  if (micro_op_resolver.AddDepthwiseConv2D() != kTfLiteOk) {
    ESP_LOGE(TAG, "failed to register ops AddDepthwiseConv2D");
    return false;
  }

  if (micro_op_resolver.AddFullyConnected() != kTfLiteOk) {
    ESP_LOGE(TAG, "failed to register ops AddFullyConnected");
    return false;
  }

  if (micro_op_resolver.AddMean() != kTfLiteOk) {
    ESP_LOGE(TAG, "failed to register ops AddMean");
    return false;
  }

  if (micro_op_resolver.AddLogistic() != kTfLiteOk) {
    ESP_LOGE(TAG, "failed to register ops AddLogistic");
    return false;
  }

  return true;
}

bool LitterRobotPresenceDetector::setup_model() {
  ExternalRAMAllocator<uint8_t> arena_allocator(ExternalRAMAllocator<uint8_t>::ALLOW_FAILURE);
  this->tensor_arena_ = arena_allocator.allocate(MODEL_ARENA_SIZE);
  if (this->tensor_arena_ == nullptr) {
    ESP_LOGE(TAG, "Could not allocate the streaming model's tensor arena.");
    return false;
  }

  this->model = ::tflite::GetModel(g_person_detect_model_data);
  if (this->model->version() != TFLITE_SCHEMA_VERSION) {
    ESP_LOGE(TAG,
             "Model provided is schema version %d not equal "
             "to supported version %d.\n",
             model->version(), TFLITE_SCHEMA_VERSION);
    return false;
  }

  static tflite::MicroMutableOpResolver<7> micro_op_resolver;
  if (!this->register_preprocessor_ops(micro_op_resolver)) {
    ESP_LOGE(TAG, "Register ops failed");
    return false;
  }

  static tflite::MicroInterpreter static_interpreter(this->model, micro_op_resolver, this->tensor_arena_,
                                                     MODEL_ARENA_SIZE);
  this->interpreter = &static_interpreter;

  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    ESP_LOGE(TAG, "AllocateTensors() failed");
    return false;
  }

  ESP_LOGD(TAG, "setup model successfully");

  // Get information about the memory area to use for the model's input.
  this->input = this->interpreter->input(0);
  this->output = this->interpreter->output(0);
  return true;
}

void LitterRobotPresenceDetector::setup() {
  ESP_LOGD(TAG, "Begin setup");

  if (!this->setup_model()) {
    ESP_LOGE(TAG, "setup model failed");
    this->mark_failed();
    return;
  }

  // SETUP CAMERA
  if (!esp32_camera::global_esp32_camera || esp32_camera::global_esp32_camera->is_failed()) {
    ESP_LOGW(TAG, "setup litter robot presence detector failed");
    this->mark_failed();
    return;
  }

  this->semaphore_ = xSemaphoreCreateBinary();

  esp32_camera::global_esp32_camera->add_image_callback([this](std::shared_ptr<esp32_camera::CameraImage> image) {
    ESP_LOGD(TAG, "received image");
    if (!this->inferring_ && image->was_requested_by(esp32_camera::API_REQUESTER)) {
      this->image_ = std::move(image);
      // xSemaphoreGive(this->semaphore_);
    }
  });

  ESP_LOGD(TAG, "setup litter robot presence detector successfully");
}

void LitterRobotPresenceDetector::update() {
  if (!this->is_ready()) {
    ESP_LOGW(TAG, "not ready yet, skip!");
    return;
  }

  if (this->inferring_) {
    ESP_LOGI(TAG, "litter robot presence detector is inferring, skip!");
    return;
  }

  esp32_camera::global_esp32_camera->request_image(esphome::esp32_camera::API_REQUESTER);
  auto image = this->wait_for_image_();

  if (!image) {
    ESP_LOGW(TAG, "SNAPSHOT: failed to acquire frame");
    return;
  }

  this->inferring_ = true;
  if (!this->start_infer(image)) {
    ESP_LOGE(TAG, "infer failed");
  } else {
    auto presence = this->is_cat_presence();
    this->publish_state(presence);
    ESP_LOGI(TAG, "cat detected: %s", presence ? "yes" : "no");
  }
  this->inferring_ = false;
}

void LitterRobotPresenceDetector::dump_config() {
  if (this->is_failed()) {
    ESP_LOGE(TAG, "  Setup Failed");
    return;
  }
  ESP_LOGCONFIG(TAG, "Input");
  ESP_LOGCONFIG(TAG, "  - dim_size: %d", input->dims->size);
  ESP_LOGCONFIG(TAG, "  - input_dims (%d,%d,%d,%d)", input->dims->data[0], input->dims->data[1], input->dims->data[2],
                input->dims->data[3]);
  ESP_LOGCONFIG(TAG, "  - zero_point=%d scale=%f", input->params.zero_point, input->params.scale);
  ESP_LOGCONFIG(TAG, "  - input_type: %d", input->type);
  ESP_LOGCONFIG(TAG, "Output");
  ESP_LOGCONFIG(TAG, "  - dim_size: %d", output->dims->size);
  ESP_LOGCONFIG(TAG, "  - dims (%d,%d)", output->dims->data[0], output->dims->data[1]);
  ESP_LOGCONFIG(TAG, "  - zero_point=%d scale=%f", output->params.zero_point, output->params.scale);
  ESP_LOGCONFIG(TAG, "  - output_type: %d", output->type);
}

std::shared_ptr<esphome::esp32_camera::CameraImage> LitterRobotPresenceDetector::wait_for_image_() {
  std::shared_ptr<esphome::esp32_camera::CameraImage> image;
  image.swap(this->image_);

  // if (!image) {
  //   // retry as we might still be fetching image
  //   xSemaphoreTake(this->semaphore_, 20000);
  //   image.swap(this->image_);
  // }

  return image;
}

bool LitterRobotPresenceDetector::start_infer(std::shared_ptr<esphome::esp32_camera::CameraImage> image) {
  camera_fb_t *rb = image->get_raw_buffer();
  size_t len = image->get_data_length();
  memcpy(this->input->data.uint8, rb->buf, image->get_data_length());
  if (kTfLiteOk != this->interpreter->Invoke()) {
    ESP_LOGE(TAG, "Invoke failed");
    return false;
  }
  return true;
}

bool LitterRobotPresenceDetector::is_cat_presence() {
  output = this->output;
  auto score = output->data.uint8[0];
  float score_f = (score - output->params.zero_point) * output->params.scale;
  ESP_LOGD(TAG, "infer score %d, float=%f", score, score_f);
  return score_f >= QUANTIZED_THESHOLD_FLOAT;
}
}  // namespace litter_robot_presence_detector
}  // namespace esphome
#endif
