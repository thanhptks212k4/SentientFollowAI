#include <iostream>
#include <thread>
#include <fstream>
#include <libobsensor/ObSensor.hpp> 
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main() {
    // Setup camera
    ob::Context ctx;
    ctx.setLoggerSeverity(OB_LOG_SEVERITY_ERROR);

    auto devList = ctx.queryDeviceList();
    if (devList->deviceCount() == 0) {
        cerr << "Error: No camera found!" << endl;
        return -1;
    }

    ob::Pipeline pipe;
    std::shared_ptr<ob::Config> config = std::make_shared<ob::Config>();

    // Configure depth stream
    try {
        auto depthProfiles = pipe.getStreamProfileList(OB_SENSOR_DEPTH);
        auto depthProfile = depthProfiles->getVideoStreamProfile(640, 480, OB_FORMAT_Y16, 30);
        config->enableStream(depthProfile);
    } catch (...) {
        config->enableStream(pipe.getStreamProfileList(OB_SENSOR_DEPTH)->getProfile(0));
    }

    // Configure color stream
    try {
        auto colorProfiles = pipe.getStreamProfileList(OB_SENSOR_COLOR);
        auto colorProfile = colorProfiles->getVideoStreamProfile(640, 480, OB_FORMAT_RGB, 30);
        config->enableStream(colorProfile);
    } catch (...) {
        config->enableStream(pipe.getStreamProfileList(OB_SENSOR_COLOR)->getProfile(0));
    }

    try {
        pipe.start(config);
    } catch (const ob::Error& e) {
        cerr << "Start error: " << e.getMessage() << endl;
        return -1;
    }

    cout << "CAMERA_READY" << endl;
    cout.flush();

    int frame_count = 0;
    
    while (true) {
        auto frameSet = pipe.waitForFrames(1000);

        if (frameSet != nullptr) {
            auto depthFrame = frameSet->depthFrame();
            auto colorFrame = frameSet->colorFrame();
            
            if (depthFrame != nullptr && colorFrame != nullptr) {
                frame_count++;
                
                // Get dimensions
                int depth_width = depthFrame->width();
                int depth_height = depthFrame->height();
                int color_width = colorFrame->width();
                int color_height = colorFrame->height();
                
                // Get data pointers
                uint16_t* depth_data = (uint16_t*)depthFrame->data();
                uint8_t* color_data = (uint8_t*)colorFrame->data();
                
                // Convert color to OpenCV format and save as temp file
                Mat colorMat(color_height, color_width, CV_8UC3, color_data);
                Mat bgrMat;
                cvtColor(colorMat, bgrMat, COLOR_RGB2BGR);
                
                // Save to temp files with frame number
                string color_file = "/tmp/astra_fast_color.jpg";
                string depth_file = "/tmp/astra_fast_depth.bin";
                
                // Save color as JPEG
                imwrite(color_file, bgrMat);
                
                // Save depth as binary
                ofstream depth_out(depth_file, ios::binary);
                depth_out.write((char*)depth_data, depth_width * depth_height * sizeof(uint16_t));
                depth_out.close();
                
                // Output frame info
                cout << "FRAME:" << frame_count 
                     << ",DEPTH:" << depth_width << "x" << depth_height
                     << ",COLOR:" << color_width << "x" << color_height;
                
                // Output center depth
                uint16_t center_depth = depth_data[(depth_height/2) * depth_width + (depth_width/2)];
                cout << ",CENTER_DEPTH:" << center_depth;
                
                // Output file paths
                cout << ",COLOR_FILE:" << color_file
                     << ",DEPTH_FILE:" << depth_file << endl;
                
                cout.flush();
            }
        }
        
        // Target 30 FPS
        this_thread::sleep_for(chrono::milliseconds(33));
    }

    pipe.stop();
    return 0;
}