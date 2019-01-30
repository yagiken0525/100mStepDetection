//
// Created by yagi on 18/01/10.
//

#include "panorama.h"
#include "basicFunction/basicFunction.h"

using namespace yagi;
using namespace std;

//
//void display(const std::shared_ptr<std::vector<op::Datum>>& datumsPtr)
//{
//    // User's displaying/saving/other processing here
//    // datum.cvOutputData: rendered frame with pose or heatmaps
//    // datum.poseKeypoints: Array<float> with the estimated pose
//    if (datumsPtr != nullptr && !datumsPtr->empty())
//    {
//        // Display image
//        cv::imshow("User worker GUI", datumsPtr->at(0).cvOutputData);
//        cv::waitKey(1);
//    }
//    else
//        op::log("Nullptr or empty datumsPtr found.", op::Priority::High);
//}
//
//void printKeypoints(const std::shared_ptr<std::vector<op::Datum>>& datumsPtr)
//{
//    // Example: How to use the pose keypoints
//    if (datumsPtr != nullptr && !datumsPtr->empty())
//    {
//        // Alternative 1
//        op::log("Body keypoints: " + datumsPtr->at(0).poseKeypoints.toString());
//
//        // // Alternative 2
//        // op::log(datumsPtr->at(0).poseKeypoints);
//
//        // // Alternative 3
//        // std::cout << datumsPtr->at(0).poseKeypoints << std::endl;
//
//        // // Alternative 4 - Accesing each element of the keypoints
//        // op::log("\nKeypoints:");
//        // const auto& poseKeypoints = datumsPtr->at(0).poseKeypoints;
//        // op::log("Person pose keypoints:");
//        // for (auto person = 0 ; person < poseKeypoints.getSize(0) ; person++)
//        // {
//        //     op::log("Person " + std::to_string(person) + " (x, y, score):");
//        //     for (auto bodyPart = 0 ; bodyPart < poseKeypoints.getSize(1) ; bodyPart++)
//        //     {
//        //         std::string valueToPrint;
//        //         for (auto xyscore = 0 ; xyscore < poseKeypoints.getSize(2) ; xyscore++)
//        //             valueToPrint += std::to_string(   poseKeypoints[{person, bodyPart, xyscore}]   ) + " ";
//        //         op::log(valueToPrint);
//        //     }
//        // }
//        // op::log(" ");
//    }
//    else
//        op::log("Nullptr or empty datumsPtr found.", op::Priority::High);
//}
/**
 * @fn
 * 画像の読み込みを行う
 * @brief 画像の読み込みを行う
 * @param (show_image) 読み込んだ画像を表示
 */
void Panorama::loadImage() {

    //画像リストopen
    ifstream ifs(_image_txt_path);

    //imshowうまくいかない時用
    string line;
    int string_size = 0;

    //画像枚数カウンター
    int line_counter = 0;

    cv::Mat preEdge = cv::Mat::zeros(1,1, CV_8U);
    // cv::namedWindow("image", CV_WINDOW_NORMAL);
    while (getline(ifs, line)) {

        //imshowがうまくいかないときここ原因(下4行をコメントアウト)
        if(this->_video_name == "kiryu998" || this->_video_name == "Rio100m" || this->_video_name == "100mT42") {
            if (string_size == 0 || (string_size + 1) == line.size()) {
                line.erase(line.size() - 1);
            }
            string_size = line.size();
        }

        //ImageInfoに画像を格納していく
        Panorama::ImageInfo image_info;

        //img_namesに画像の名前格納
        this->img_names.push_back(line);

        //カラー、グレースケール,hsv
        cv::Mat image = cv::imread(line);
        cv::Mat gray_image = cv::imread(line, 0);
        cv::Mat img_hsv;
        cv::cvtColor(image, img_hsv, cv::COLOR_BGR2HSV);

        //エッジ画像（縦方向微分）
        cv::Mat edge = cv::Mat::zeros(gray_image.size(), CV_8UC1);
        for (int i = 1; i < gray_image.rows; i ++){
            for (int j = 0; j < gray_image.cols; j ++){
                if ((gray_image.at<unsigned char>(i, j) - gray_image.at<unsigned char>(i - 1, j)) > 40){
                    edge.at<unsigned char>(i, j) = 255;
                }
            }
        }

//        cv::imshow("edge", gray_image);
//        cv::imshow("oedge", preEdge);
//        cv::waitKey(0);

        //画像格納
        image_info.image = image;
        image_info.gray_image = gray_image;
        image_info.hsv_image = img_hsv;
        image_info.edge = edge;
        preEdge = gray_image;

        //DensePoseのマスク
//        cout << this->_image_folder + _video_name + "/maskImages/image" + yagi::digitString(line_counter, 4) + "_IUV.png" << endl;
        cv::Mat mask = cv::imread(this->_image_folder + _video_name + "/maskImages/image" + yagi::digitString(line_counter, 4) + "_IUV.png", 0);
        cv::threshold(mask, mask, 10, 255, cv::THRESH_BINARY);
        image_info.denseMask = mask;

        imList.push_back(image_info);

        //デバック
        if (SHOW_LOADED_IMAGE) {
            cv::imshow("color", image);
//            cv::imshow("gray", gray_image);
            cout << "Frame: " << line_counter << endl;
            cv::waitKey(0);
        }
        line_counter++;
    }
    IMG_WIDTH = imList[0].image.cols;
    IMG_HEIGHT = imList[0].image.rows;
}
