// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <iostream>
#include <vector>
#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/dnn.hpp"
#include "paddle_api.h"  // NOLINT
/////////////////////////////////////////////////////////////////////////
// If this demo is linked to static library:libpaddle_api_light_bundled.a
// , you should include `paddle_use_ops.h` and `paddle_use_kernels.h` to
// avoid linking errors such as `unsupport ops or kernels`.
/////////////////////////////////////////////////////////////////////////
// #include "paddle_use_kernels.h"  // NOLINT
// #include "paddle_use_ops.h"      // NOLINT

using namespace cv;
using namespace paddle::lite_api;  // NOLINT

struct Detection
{
    int class_id;
    float confidence;
    cv::Rect box;
};

std::vector<std::string> className= {"lefttop","leftbottom","rightbottom","righttop"};

cv::Mat letterbox(const cv::Mat &img, cv::Size new_shape, cv::Scalar color, bool _auto, bool scaleFill, bool scaleup, int stride)
{
    float width = img.cols;
    float height = img.rows;
    float r = min(new_shape.width / width, new_shape.height / height);
    if (!scaleup)
        r = min(r, 1.0f);
    int new_unpadW = int(round(width * r));
    int new_unpadH = int(round(height * r));
    int dw = new_shape.width - new_unpadW;
    int dh = new_shape.height - new_unpadH;
    if (_auto)
    {
        dw %= stride;
        dh %= stride;
    }
    dw /= 2, dh /= 2;
    Mat dst;
    resize(img, dst, Size(new_unpadW, new_unpadH), 0, 0, cv::INTER_LINEAR);
    int top = int(round(dh - 0.1));
    int bottom = int(round(dh + 0.1));
    int left = int(round(dw - 0.1));
    int right = int(round(dw + 0.1));
    copyMakeBorder(dst, dst, top, bottom, left, right, cv::BORDER_CONSTANT, color);
    return dst;
}

// fill tensor with mean and scale and trans layout: nhwc -> nchw, neon speed up
void neon_mean_scale(const float* din,
                     float* dout,
                     int size,
                     const std::vector<float> mean,
                     const std::vector<float> scale) {
  if (mean.size() != 3 || scale.size() != 3) {
    std::cerr << "[ERROR] mean or scale size must equal to 3\n";
    exit(1);
  }
  float32x4_t vmean0 = vdupq_n_f32(mean[0]);
  float32x4_t vmean1 = vdupq_n_f32(mean[1]);
  float32x4_t vmean2 = vdupq_n_f32(mean[2]);
  float32x4_t vscale0 = vdupq_n_f32(1.f / scale[0]);
  float32x4_t vscale1 = vdupq_n_f32(1.f / scale[1]);
  float32x4_t vscale2 = vdupq_n_f32(1.f / scale[2]);
  float* dout_c0 = dout;
  float* dout_c1 = dout + size;
  float* dout_c2 = dout + size * 2;
  int i = 0;
  for (; i < size - 3; i += 4) {
    float32x4x3_t vin3 = vld3q_f32(din);
    float32x4_t vsub0 = vsubq_f32(vin3.val[0], vmean0);
    float32x4_t vsub1 = vsubq_f32(vin3.val[1], vmean1);
    float32x4_t vsub2 = vsubq_f32(vin3.val[2], vmean2);
    float32x4_t vs0 = vmulq_f32(vsub0, vscale0);
    float32x4_t vs1 = vmulq_f32(vsub1, vscale1);
    float32x4_t vs2 = vmulq_f32(vsub2, vscale2);
    vst1q_f32(dout_c0, vs0);
    vst1q_f32(dout_c1, vs1);
    vst1q_f32(dout_c2, vs2);
    din += 12;
    dout_c0 += 4;
    dout_c1 += 4;
    dout_c2 += 4;
  }
  for (; i < size; i++) {
    *(dout_c0++) = (*(din++) - mean[0]) * scale[0];
    *(dout_c1++) = (*(din++) - mean[1]) * scale[1];
    *(dout_c2++) = (*(din++) - mean[2]) * scale[2];
  }
}

cv::Mat pre_process(const cv::Mat& img, int width, int height, float* data) {
  cv::Mat img0 = letterbox(img, cv::Size(1280, 1280), cv::Scalar(114, 114, 114), false, false, true, 32);
  cv::Mat rgb_img;
  cv::cvtColor(img0, rgb_img, cv::COLOR_BGR2RGB);
  // cv::resize(
  //     rgb_img, rgb_img, cv::Size(width, height), 0.f, 0.f, cv::INTER_CUBIC);
  cv::imwrite("rgb1.jpg",rgb_img);
  cv::Mat imgf;
  rgb_img.convertTo(imgf, CV_32FC3, 1 / 255.f);
  std::vector<float> mean = {0.0f, 0.0f, 0.0f};
  std::vector<float> scale = {1.0f, 1.0f, 1.0f};
  const float* dimg = reinterpret_cast<const float*>(imgf.data);
  neon_mean_scale(dimg, data, width * height, mean, scale);
  return img0;
}


void RunModel(std::string model_file, std::string img_path) {
  // 1. Set MobileConfig
  MobileConfig config;
  config.set_model_from_file(model_file);

  // 2. Create PaddlePredictor by MobileConfig
  std::shared_ptr<PaddlePredictor> predictor =
      CreatePaddlePredictor<MobileConfig>(config);

  const float IN_HEIGHT = 1280.0;
  const float IN_WIDTH = 1280.0;
  const int in_height = 1280;
  const int in_width = 1280;
  const float SCORE_THRESHOLD = 0.2;
  const float NMS_THRESHOLD = 0.4;
  const float CONFIDENCE_THRESHOLD = 0.4;
  const std::vector<cv::Scalar> colors = {cv::Scalar(255, 255, 0), cv::Scalar(0, 255, 0), cv::Scalar(0, 255, 255), cv::Scalar(255, 0, 0)};


  // 3. Prepare input data from image
  // input 0
  std::unique_ptr<Tensor> input_tensor0(std::move(predictor->GetInput(0)));
  input_tensor0->Resize({1, 3, in_height, in_width});
  auto* data0 = input_tensor0->mutable_data<float>();
  cv::Mat img = imread(img_path, cv::IMREAD_COLOR);
  cv::Mat img0 =pre_process(img, in_width, in_height, data0);
  // input1
  // std::unique_ptr<Tensor> input_tensor1(std::move(predictor->GetInput(1)));
  // input_tensor1->Resize({1, 2});
  // auto* data1 = input_tensor1->mutable_data<int>();
  // data1[0] = img.rows;
  // data1[1] = img.cols;

  // 4. Run predictor
  predictor->Run();

  // 5. Get output and post process
  std::unique_ptr<const Tensor> output_tensor(
      std::move(predictor->GetOutput(0)));
  std::vector<Detection> output;
  float x_factor =  img0.cols/ IN_WIDTH;
  float y_factor = img0.rows / IN_HEIGHT;
  auto* data = output_tensor->data<float>();  
  const int dimensions = 9;
  const int rows = 100800;
  std::vector<int> class_ids;
  std::vector<float> confidences;
  std::vector<cv::Rect> boxes;
  for (int i = 0; i < rows; ++i) {
      float confidence = data[4];
      if (confidence >= CONFIDENCE_THRESHOLD) {
          // float * classes_scores = data + 5;
          float classes_scores[4] = {data[5],data[6],data[7],data[8]};
          cv::Mat scores(1, className.size(), CV_32FC1, classes_scores);

          // cv::Scalar intensity = scores.at<float>(0, 3);
          // float *data1 = (float *)scores.data; 
          cv::Point class_id;
          double max_class_score;
          minMaxLoc(scores, 0, &max_class_score, 0, &class_id);
          if (max_class_score > SCORE_THRESHOLD) {
              confidences.push_back(confidence);
              class_ids.push_back(class_id.x);
              float x = data[0];
              float y = data[1];
              float w = data[2];
              float h = data[3];
              int left = int((x - 0.5 * w) * x_factor);
              int top = int((y - 0.5 * h) * y_factor);
              int width = int(w * x_factor);
              int height = int(h * y_factor);
              boxes.push_back(cv::Rect(left, top, width, height));
          }
      }
      data += 9;
  }  
  std::vector<int> nms_result;
  cv::dnn::NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, nms_result);
  for (int i = 0; i < nms_result.size(); i++) {
      int idx = nms_result[i];
      Detection result;
      result.class_id = class_ids[idx];
      result.confidence = confidences[idx];
      result.box = boxes[idx];
      output.push_back(result);
  }
  int detections = output.size();
  for (int i = 0; i < detections; ++i)
        {
            auto detection = output[i];
            auto box = detection.box;
            auto classId = detection.class_id;
            const auto color = colors[classId % colors.size()];
            cv::rectangle(img0, box, color, 3);
            cv::rectangle(img0, cv::Point(box.x, box.y - 20), cv::Point(box.x + box.width, box.y), color, cv::FILLED);
            cv::putText(img0, className[classId].c_str(), cv::Point(box.x, box.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
        }
        cv::imwrite("result1.jpg",img0);
}

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "[ERROR] usage: " << argv[0] << " model_file image_path\n";
    exit(1);
  }
  std::string model_file = argv[1];
  std::string img_path = argv[2];
  RunModel(model_file, img_path);
  return 0;
}
