#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <string.h>
#include "tool/YamlTool.h"
#include "tool/FileSystemTool.h"
#include "PPSeg.h"
using namespace std;
using namespace perception;
int main()
{

    PP_Seg ppSeg;
    string config_path = CURRENT_FOLDER_PATH + "/data/config.yaml";
    YAML::Node master_config = *tool::YamlTool::GetConfig(config_path);
    YAML::Node camera_config = master_config["perception"]["camera"];
    YAML::Node seg_config = camera_config["detector"]["ppseg"];

    SegOptions option;
    option.model_path = seg_config["model"].as<string>();
    option.classes = seg_config["label"].as<int>();

    ppSeg.init(option);
    cv::VideoCapture reader("/home/bingda/Downloads/1.mp4");
    cv::Mat tframe;
    cv::Mat raw_frame;
    while (reader.read(tframe))
    {
        std::vector<int> pred = ppSeg.detect(tframe);
        raw_frame = ppSeg.display_masked_image(pred, tframe);
        cv::imshow("pic", tframe);
        cv::imshow("result", raw_frame);
        cv::waitKey(20);
    }
    return 0;
}
