#include <opencv2/opencv.hpp>
#include "rknn_api.h"
#include <sys/time.h>
#include <vector>

struct SegOptions
{
    std::string model_path;
    int classes;
};

class PP_Seg {
public:
    PP_Seg();

    bool init(const SegOptions&);
    std::vector<int> detect(cv::Mat src_img);
    cv::Mat display_masked_image(std::vector<int> pred, cv::Mat raw_frame);

private:
    SegOptions option_;
    rknn_context ctx = 0;
    int ret = 0;
    int req_height = 0;
    int req_width = 0;
    int req_channel = 0;
    int height = 0;
    int width = 0;
    int channel = 0;
    int stride = 0;
    unsigned char *model_data;
    rknn_input_output_num io_num;
};

