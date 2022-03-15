#include <iostream>
#include <memory>
#include <chrono>

#include "detect.h"
#include "cxxopts.hpp"


std::vector<std::string> LoadNames(const std::string& data_path) {
    std::vector<std::string> class_names;
    std::ifstream infile(data_path);
    if (infile.is_open()) {
        std::string line;
        while (getline (infile,line)) {
            class_names.emplace_back(line);
        }
        infile.close();
    }
    else {
        std::cerr << "Error loading the class names!\n";
    }

    return class_names;
}

void DemoVideo(std::string source,
                auto& detector,
                float conf_thres,
                float iou_thres,
                const std::vector<std::string>& class_names,
                bool label = true) {
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

        // detector.ImagePreprocess(frame, dst);
        auto detections = detector.Run(frame, conf_thres, iou_thres);
        if (!detections.empty()) {
            for (const auto& det : detections[0]) {
                const auto& box = det.bbox;
                float score = det.score;
                int class_idx = det.class_idx;
                cv::rectangle(frame, box, cv::Scalar(0, 0, 255), 2);
                if (label) {
                    std::stringstream ss;
                    ss << std::fixed << std::setprecision(2) << score;
                    std::string s = class_names[class_idx] + " " + ss.str();

                    auto font_face = cv::FONT_HERSHEY_DUPLEX;
                    auto font_scale = 1.0;
                    int thickness = 1;
                    int baseline=0;
                    auto s_size = cv::getTextSize(s, font_face, font_scale, thickness, &baseline);
                    cv::rectangle(frame,
                            cv::Point(box.tl().x, box.tl().y - s_size.height - 5),
                            cv::Point(box.tl().x + s_size.width, box.tl().y),
                            cv::Scalar(0, 0, 255), -1);
                    cv::putText(frame, s, cv::Point(box.tl().x, box.tl().y - 5),
                                font_face , font_scale, cv::Scalar(255, 255, 255), thickness);
                }

            }
        }
        cv::namedWindow("Result", cv::WINDOW_AUTOSIZE);
        cv::imshow("Result", frame);

        int key = cv::waitKey(20);
        if (key == 'q') break;

    }
    cap.release();

}

void DemoSingleImage(std::string source,
                        auto& detector,
                        float conf_thres,
                        float iou_thres,
                        const std::vector<std::string>& class_names,
                        bool label = true) {
    cv::Mat img = cv::imread(source);
    if (img.empty()) {
        std::cerr << "demo_single_image:: Error loading the image!\n";
        return;
    }
    auto detections = detector.Run(img, conf_thres, iou_thres);
    if (!detections.empty()) {
        for (const auto& detection : detections[0]) {
            const auto& box = detection.bbox;
            float score = detection.score;
            int class_idx = detection.class_idx;

            cv::rectangle(img, box, cv::Scalar(0, 0, 255), 2);

            if (label) {
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
    }

    cv::namedWindow("Result", cv::WINDOW_AUTOSIZE);
    cv::imshow("Result", img);
    cv::waitKey(0);
}

void WarmUp(int w, int h, auto& detector) {
    std::cout << "Run once on empty image" << std::endl;
    auto temp_img = cv::Mat::zeros(h, w, CV_32FC3);
    detector.debug("warm up");
    
}


int main(int argc, const char* argv[]) {
    cxxopts::Options parser(argv[0], "A LibTorch inference implementation of the models");
    // TODO: add other args
    parser.allow_unrecognised_options().add_options()
            ("weights", "model.torchscript.pt path", cxxopts::value<std::string>())
            ("data", "label names", cxxopts::value<std::string>()->default_value("../weights/coco.names"))
            ("img-src", "image source", cxxopts::value<std::string>()->default_value("../images/bus.jpg"))
            ("video-src", "image source", cxxopts::value<std::string>()->default_value("/home/brian/Documents/trap_2/0.mp4"))
            ("using-video", "using video", cxxopts::value<bool>()->default_value("false"))
            ("conf-thres", "object confidence threshold", cxxopts::value<float>()->default_value("0.4"))
            ("iou-thres", "IOU threshold for NMS", cxxopts::value<float>()->default_value("0.5"))
            ("gpu", "Enable cuda device or cpu", cxxopts::value<bool>()->default_value("false"))
            ("view-img", "display results", cxxopts::value<bool>()->default_value("false"))
            ("h,help", "Print usage");
    auto opt = parser.parse(argc, argv);

    if (opt.count("help")) {
        std::cout << parser.help() << std::endl;
        exit(0);
    }

    bool is_gpu = opt["gpu"].as<bool>();

    // set device type - CPU/GPU
    torch::DeviceType device_type;
    if (torch::cuda::is_available() && is_gpu) {
        device_type = torch::kCUDA;
    } else {
        device_type = torch::kCPU;
    }

    // load class names from dataset for visualization
    std::string data_path = opt["data"].as<std::string>();
    std::vector<std::string> class_names = LoadNames(data_path);
    if (class_names.empty()) {
        return -1;
    }

    // load network
    std::string weights = opt["weights"].as<std::string>();
    auto detector = Detector(weights, device_type);

    WarmUp(640, 640, detector);
    float conf_thres = opt["conf-thres"].as<float>();
    float iou_thres = opt["iou-thres"].as<float>();

    std::cout << "Run" << std::endl;
    if (opt["using-video"].as<bool>()) {
        std::string source = opt["video-src"].as<std::string>();
        DemoVideo(source, detector, conf_thres, iou_thres, class_names);
    } else {
        std::string source = opt["img-src"].as<std::string>();
        DemoSingleImage(source, detector, conf_thres, iou_thres, class_names);
    }

    



    return 0;
}