
// * @file main.cpp
// * @brief 100m走の動画から各選手の歩幅、歩数を算出
// * @author 八木賢太郎
// * @date 2018/1/3
// */

#include <opencv2/opencv.hpp>
#include "src/panorama.h"
#include "src/openpose/myOpenPose.h"



//#include <openpose/flags.hpp>
//#include "src/nDegreeApproximation.h"
//#include "src/basicFunction/basicFunction.h"
using namespace std;
using namespace yagi;
using namespace cv;



int main() {
    //! 入力動画のファイル名
    string video_name = "Rio100m";
    string folder_path = "/home/yagi/sfmDR/inputVideos/" + video_name + "/";

    Panorama pano;
    pano.INIT_PROCESSING = false;
//    pano.USE_LASTMASK = false;
//    pano.USE_LAST_TRACKLINE = false;
//    pano.USE_LAST_CORNERS = false;
//    pano.SELECT_TARGET_RANE = false;
    pano.MAX_TRANSLATION = 5;
    pano.TARGET_RUNNER_ID = 6;
    pano.MASK_MARGIN = 50;
    pano.OP_MASK_RADIUS = 10;
    pano.FIRST_IM_ID = 10;
    pano.LAST_IM_ID = 245;
    pano.PROJECTION_STEP = 10;
    pano.RANSAC_LOOP_LIMIT = 10;
    pano.RANSAC_INLIER_RANGE = 1;
    pano.STROBO_RESIZE_MARGIN = 30;
    pano.SHOW_LOADED_IMAGE = false;
    pano.SHOW_ONLINE_POINTS = false;
    pano.SHOW_MASK_REGIONS = false;
    pano.SHOW_TRANSLATION = false;
    pano.SHOW_HOMOGRAPHY = false;
    pano.SHOW_TRACKLINES = false;
    pano.SHOW_PANORAMA = false;
    pano.SHOW_STROBO_PROCESS = false;
    pano.SHOW_RUNNER_CANDIDATES = false;
    pano.ESTIMATE_STEPS = false;
    pano.GENERATE_STROBO = false;
//    pano.GENERATE_VIRTUALRACE = false;
    pano.VIRTUAL_TARGET_VIDEO = "Bolt958";
//    pano.REMOVE_OTHER_RUNNERS = false;



    pano.setVariables(video_name);
    if (pano.INIT_PROCESSING){
        pano.videotoImage();
        outputTextFromVideo(folder_path, folder_path + video_name + ".mp4", folder_path );
    }else{
        pano.loadingData();
        pano.masking();
        pano.trackDetection();
        pano.trackingRunner();
        pano.makePanorama();
        pano.saveData();
        if(pano.GENERATE_STROBO)
            pano.makeStroboRangeImage();
        if(pano.ESTIMATE_STEPS)
            pano.estimateStepPoints();
        if(pano.GENERATE_VIRTUALRACE)
            pano.makeVirtualRaceImages();
    }
    return 0;
}
