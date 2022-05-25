#include <iostream>
#include <memory>
#include <chrono>

#include "inference.h"
#include "cxxopts.hpp"
using json = nlohmann::json;

std::string CONFIG_FILE = "../model_setting.json";

void DemoVideo(std::string source, auto& detector) {
    cv::VideoCapture cap(source);
    if(!cap.isOpened()){
        std::cout << "Error opening video stream or file" << std::endl;
        return ;
    }

    while(true) {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) {
            std::cerr << "Error loading frame!\n";
        }

        auto detections = detector.Run(frame);
        cv::Mat result = DrawResults(detections, frame, detector.classes);
        cv::namedWindow("Result", cv::WINDOW_AUTOSIZE);
        cv::imshow("Result", result);

        int key = cv::waitKey(20);
        if (key == 'q') break;

    }
    cap.release();

}

void DemoSingleImage(std::string source, auto& detector) {
    cv::Mat img = cv::imread(source);
    if (img.empty()) {
        std::cerr << "demo_single_image:: Error loading the image!\n";
        return;
    }
    auto detections = detector.Run(img);
    cv::Mat result = DrawResults(detections, img, detector.classes);

    cv::namedWindow("Result", cv::WINDOW_AUTOSIZE);
    cv::imshow("Result", result);
    cv::waitKey(0);
}


int main(int argc, const char* argv[]) {
    json cfg;
    std::ifstream configFile(CONFIG_FILE);
    configFile >> cfg;

    // set device type - CPU/GPU
    torch::DeviceType device_type;
    if (torch::cuda::is_available() && cfg["Model"]["gpu"]) {
        device_type = torch::kCUDA;
    } else {
        device_type = torch::kCPU;
    }

    ModelDetector detector(cfg, device_type);
    if (detector.classes.empty()) {
        return -1;
    }


    std::cout << "Run" << std::endl;
    if (cfg["System"]["using_video"].get<bool>()) {
        DemoVideo(cfg["System"]["video_path"].get<std::string>(), detector);
    } 
    else {
        DemoSingleImage(cfg["System"]["image_path"].get<std::string>(), detector);
    }

    return 0;
}