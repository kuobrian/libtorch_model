# pragma once

#include <memory>

#include <torch/script.h>
#include <torch/torch.h>

#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDAEvent.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn/dnn.hpp>

#include "utils.h"
#include "json.hpp"
using json = nlohmann::json;

cv::Mat DrawResults(auto& detections, 
                        const cv::Mat frame,
                        std::vector<std::string> class_names) {
    cv::Mat img = frame.clone();
    if (!detections.empty()) {
        for (const auto& detection : detections[0]) {
            const auto& box = detection.bbox;
            float score = detection.score;
            int class_idx = detection.class_idx;

            cv::rectangle(img, box, cv::Scalar(0, 0, 255), 2);
            std::stringstream ss;
            ss << std::fixed << std::setprecision(2) << score;
            std::string s = class_names[class_idx] + " " + ss.str();

            auto font_face = cv::FONT_HERSHEY_DUPLEX;
            auto font_scale = 1.0;
            int thickness = 1;
            int baseline=0;
            auto s_size = cv::getTextSize(s, font_face, font_scale, thickness, &baseline);
            cv::rectangle(img,
                    cv::Point(box.tl().x, box.tl().y - s_size.height - 5),
                    cv::Point(box.tl().x + s_size.width, box.tl().y),
                    cv::Scalar(0, 0, 255), -1);
            cv::putText(img, s, cv::Point(box.tl().x, box.tl().y - 5),
                        font_face , font_scale, cv::Scalar(255, 255, 255), thickness);
        }
    }
    return img;
}

class ModelDetector {
    private:
        torch::Device device_;
        torch::jit::script::Module module_;
        bool half_;
        int heigth;
        int width;
        float conf_thres;
        float iou_thres;

        void WarmUp(int n);
        void LoadWeight(const std::string& path);

    public:
        json cfg;
        std::vector<std::string> classes;
        
        ModelDetector(json cfg, const torch::DeviceType& device_type);
        
        void ImagePreprocess(const cv::Mat& src, cv::Mat& dst);

        static std::vector<std::string> LoadNames(const std::string& data_path);

        std::vector<std::vector<Detection>> Run(const cv::Mat& img);

        static std::vector<float> LetterboxImage(const cv::Mat& src, cv::Mat& dst,
                                            const cv::Size& out_size = cv::Size(640, 640));
        
        static void Preprocess(const cv::Mat& src, cv::Mat& dst);
        static std::vector<std::vector<Detection>> PostProcessing(const torch::Tensor& detections,
                                                              float pad_w, float pad_h, float scale, const cv::Size& img_shape,
                                                              float conf_thres = 0.4, float iou_thres = 0.6);

        static void ScaleCoordinates(std::vector<Detection>& data, float pad_w, float pad_h,
                                 float scale, const cv::Size& img_shape);

    
        static torch::Tensor xywh2xyxy(const torch::Tensor& x);

        static void Tensor2Detection(const at::TensorAccessor<float, 2>& offset_boxes,
                                        const at::TensorAccessor<float, 2>& det,
                                        std::vector<cv::Rect>& offset_box_vec,
                                        std::vector<float>& score_vec);

};
