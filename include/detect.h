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

class Detector {
    public:
        Detector(const torch::DeviceType& device_type);
        Detector(const std::string& model_path, const torch::DeviceType& device_type);
        
        void debug(std::string s);
        void ImagePreprocess(const cv::Mat& src, cv::Mat& dst);
        std::vector<std::vector<Detection>>
        Run(const cv::Mat& img, float conf_threshold, float iou_threshold);
    
    private:
        torch::jit::script::Module module_;
        torch::Device device_;
        bool half_;

        static std::vector<float> LetterboxImage(const cv::Mat& src, cv::Mat& dst,
                                            const cv::Size& out_size = cv::Size(640, 640));
        
        
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
