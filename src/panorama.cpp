#include "panorama.h"
#include <opencv2/xfeatures2d.hpp>
#include "plotRunnerPose/mymatplot.h"
#include "basicFunctions/basicFunction.h"
#include "virtualRace.h"
#include "nDegreeApproximation.h"


using namespace std;
using namespace yagi;
using namespace cv;

Panorama::Panorama(string video_name) {
    _video_name = video_name;
    _project_path = "../projects/" + video_name + "/";
    _image_folder = _project_path + "/images/";
    _result_folder = _project_path + "/results/";
    _video_folder = "../videos/";
    _txt_folder = _project_path + "/texts/";
    _openpose_list_path = _txt_folder + "/human_pose_info.txt";
    _image_list_path = _txt_folder + "/imagelist.txt";

    //ディレクトリ作成
    myMkdir(_project_path);
    myMkdir(_image_folder);
    myMkdir(_result_folder);
    myMkdir(_txt_folder);
}

//グローバル変数
struct mouseParam {
    int x;
    int y;
};

cv::Point2f point;
bool myclicked = false;

//変数格納
//void Panorama::setVariables(string video_name) {
//    _video_name = video_name;
//    _image_list_path = _image_folder + _video_name + _imagelist_name;
//    _openpose_list_path = _image_folder + _video_name + _poselist_name;
//}


//ビデオのトリミング
void Panorama::videotoImage() {

    namespace vti = videoToImage;

    vti::trimVideo(_video_name, _video_folder, _image_folder, _image_list_path, videoType);

}

void Panorama::loadingData(){
    loadImage();
    detectHumanArea();
}

void Panorama::masking(){
    selectMaskArea();
    maskHumanArea();
}

void Panorama::trackDetection() {
    selectTrack();
    trackTracking();
    startFinishLineSelect();
}

void Panorama::trackingRunner(){
    selectRunnerCandidates();

//    ofstream opCsvData(_txt_folder + "opData.csv");
//    for (int i = 0; i < imList.size(); i++) {
//        ImageInfo im = imList[i];
//        cv::Mat dummy = im.image;
//        for (OpenPoseBody hb: im.Runners) {
//            cv::circle(dummy, hb._body_parts_coord[0], 3, cv::Scalar(0,0,255), 3);
//            cv::circle(dummy, hb._body_parts_coord[8], 3, cv::Scalar(0,0,255), 3);
//            cv::circle(dummy, hb._body_parts_coord[14], 3, cv::Scalar(0,0,255), 3);
//            for(cv::Point2f pt : hb._body_parts_coord){
//                float x = pt.x - hb._body_parts_coord[0].x;
//                float y = pt.y - hb._body_parts_coord[0].y;
//                opCsvData << x << " " << y << " ";
//            }
//            cv::imshow("op for keras", dummy);
//            int key = cv::waitKey();
//            opCsvData << key - 177 << endl;
//        }
//    }

    trackTargetRunner();
    getOpenPoseMask();
}

void Panorama::makePanorama(){
    getTranslation();
    getHomographyFromTranslation();
//    transformPanoramaTopview();
    generatePanorama();

    //Runnerのいない画像を生成
//    if(REMOVE_OTHER_RUNNERS){
//        for(ImageInfo im: imList){
//            cv::Mat noRunnerImage;
//            warpPerspective(this->smallPanoramaImage, noRunnerImage, im.mulH.inv() ,cv::Size(IMG_WIDTH, IMG_HEIGHT));
//
//            cv::imshow("noRunner", noRunnerImage);
//            cv::waitKey();
//        }
//    }
}

void Panorama::estimateStepPoints(){
    getOverviewHomography();
    projectTrackLine();
    projectOverview();
}

void Panorama::transformPanoramaTopview(){
    vector<cv::Point2f> affineSecPts;


    //1,８レーン上の一点を抽出
    const int x = 300;
    int lineNum = imList[0].grads.size();
    cv::Point2f pt1(x, x*imList[0].grads[0] + imList[0].segments[0]);
    cv::Point2f pt2(x, x*imList[0].grads[lineNum-1] + imList[0].segments[lineNum-1]);
    affineSecPts.push_back(startLineCornerPoints[0]);
    affineSecPts.push_back(pt1);
    affineSecPts.push_back(pt2);
    affineSecPts.push_back(startLineCornerPoints[1]);
    cv::Mat dummyIm = imList[0].image;

    for(cv::Point2f pt: affineSecPts){
        cv::circle(dummyIm, pt, 2, cv::Scalar(255,0,0), 2);
        cv::imshow("a", dummyIm);
        cv::waitKey();
    }


    //投影先の座標
    vector<cv::Point2f> affineTarPts;
    affineTarPts.push_back(cv::Point2f(0,0));
    affineTarPts.push_back(cv::Point2f(x,0));
    affineTarPts.push_back(cv::Point2f(x,x));
    affineTarPts.push_back(cv::Point2f(0,x));


//    for(cv::Point2f pt : affineSecPts){
//        cv::circle(dummyIm, pt, 2, cv::Scalar(255,0,0), 2);
//        cv::imshow("pt", dummyIm);
//        cv::waitKey();
//    }

    //アフィン変換
    cv::Mat topViewIm = cv::Mat::zeros(x, x, CV_8UC3);
    affineH = cv::findHomography(affineSecPts, affineTarPts);
//    cv::warpAffine(dummyIm, topViewIm, H, cv::Size(x, x));

//    affineH = cv::Mat::zeros(3, 3, CV_64F);
//    affineH .at<double>(0, 0) = H.at<double>(0,0);
//    affineH .at<double>(0, 1) = H.at<double>(0,1);
//    affineH .at<double>(0, 2) = H.at<double>(0,2);
//    affineH .at<double>(1, 0) = H.at<double>(1,0);
//    affineH .at<double>(1, 1) = H.at<double>(1,1);
//    affineH .at<double>(1, 2) = H.at<double>(1,2);
//    affineH .at<double>(2, 2) = 1.0;

//    //８レーン上の一点を抽出
//    const int x = 300;
//    int lineNum = imList[0].grads.size();
//    cv::Point2f pt(x, x*imList[0].grads[lineNum-1] + imList[0].segments[lineNum-1]);
//    affineSecPts.push_back(pt);
//
//    //投影先の座標
//    vector<cv::Point2f> affineTarPts;
//    cv::Mat dummyIm = imList[0].image;
//    affineTarPts.push_back(cv::Point2f(0,0));
//    affineTarPts.push_back(cv::Point2f(0,x));
//    affineTarPts.push_back(cv::Point2f(x,x));
//
////    for(cv::Point2f pt : affineSecPts){
////        cv::circle(dummyIm, pt, 2, cv::Scalar(255,0,0), 2);
////        cv::imshow("pt", dummyIm);
////        cv::waitKey();
////    }
//
//    //アフィン変換
//    cv::Mat topViewIm = cv::Mat::zeros(x, x, CV_8UC3);
//    cv::Mat H = cv::getAffineTransform(affineSecPts, affineTarPts);
////    cv::warpAffine(dummyIm, topViewIm, H, cv::Size(x, x));
//
//    affineH = cv::Mat::zeros(3, 3, CV_64F);
//    affineH .at<double>(0, 0) = H.at<double>(0,0);
//    affineH .at<double>(0, 1) = H.at<double>(0,1);
//    affineH .at<double>(0, 2) = H.at<double>(0,2);
//    affineH .at<double>(1, 0) = H.at<double>(1,0);
//    affineH .at<double>(1, 1) = H.at<double>(1,1);
//    affineH .at<double>(1, 2) = H.at<double>(1,2);
//    affineH .at<double>(2, 2) = 1.0;

    cout << affineH << endl;
//    cout << H << endl;

    cv::warpPerspective(dummyIm, topViewIm, affineH, topViewIm.size());
    cv::imshow("topView", topViewIm);
    cv::waitKey();

}

void Panorama::selectRunnerCandidates() {

    cout << "[Select runner candidate]" << endl;

    int frame_index = 0;
    for (int i = 0; i < imList.size() - 1; i++) {
        ImageInfo im = imList[i];

        // 1レーン目の直線方程式
        float a1, b1;
        a1 = im.grads[0];
        b1 = im.segments[0];

        // 9レーン目の直線方程式
        float a9, b9;
        a9 = im.grads[im.grads.size() - 1];
        b9 = im.segments[im.grads.size() - 1];

        //このフレームに映る人物の中から選手候補となる人物を選択
        for (OpenPoseBody hb: im.Runners) {

            cv::Point2f head = hb.getBodyCoord()[0];
            cv::Point2f R_leg = hb.getBodyCoord()[10];
            cv::Point2f L_leg = hb.getBodyCoord()[13];

            //頭部、両足座標(0,0)なら無視
            if ((head.x == 0 && R_leg.x == 0 && L_leg.x == 0)) {
                continue;

                //１、頭座標が9レーン目の外なら選手でない
            } else {
                if (head.y > ((a9 * head.x) + b9)) {
                    continue;
                }

                //2、足座標が１レーン目の外なら選手でない
                if ((R_leg.y < (a1 * R_leg.x + b1)) || (L_leg.y < (a1 * L_leg.x + b1))) {
                    continue;
                }

                //3、足座標が9レーン目の外なら選手でない
                if ((R_leg.y > (a9 * R_leg.x + b9)) || (L_leg.y > (a9 * L_leg.x + b9))) {
                    continue;
                }
            }

            imList[i].runnerCandidate.push_back(hb);
        }

        //デバッグランナー候補のみ画像に重畳
        cv::Scalar FaceColor(255, 0, 0);
        vector<cv::Scalar> colors;
        yagi::setColor(&colors);

        if (SHOW_RUNNER_CANDIDATES) {
            cv::Mat image = im.image.clone();
            for (OpenPoseBody hb: im.runnerCandidate) {
                cv::circle(image, hb.getBodyCoord()[0], 5, FaceColor, 5);
                cv::circle(imList[i].trackLineAndOpenPoseImage, hb.getBodyCoord()[0], 5, FaceColor, 5);
            }
            cv::imshow("runner candidates", image);
            cv::waitKey(1);
            imList[frame_index++].runnerCandidatesImage = image;
        }

        //20190131実験
        //各関節位置を重畳
//        int humanID = 0;
//        for (OpenPoseBody hb: imList[i].runnerCandidate) {
//            for(cv::Point2f pt : hb.getBodyCoord()) {
//                cv::circle(imList[i].trackLineAndOpenPoseImage, pt, 2, colors[humanID], 2);
//            }
//            humanID++;
//        }
//
//
//
//
//        cv::imshow("a", imList[i].trackLineAndOpenPoseImage);
//        cv::waitKey();
    }

    cout << "[Select runner candidate finished]" << endl;


}



void Panorama::generatePanorama() {

    cout << "[Generate panoramic image]" << endl;

    vector<cv::Scalar> colors;
    yagi::setColor(&colors);

    //パノラマ画像生成
    cv::Mat mul_H = cv::Mat::zeros(3, 3, CV_64F);
    mul_H.at<double>(0, 0) = 1;
    mul_H.at<double>(1, 1) = 1;
    mul_H.at<double>(2, 2) = 1;
    mul_H = mul_H;

    cout << mul_H << endl;
    cout << affineH.size() << endl;

    imList[0].mulH = mul_H.clone();
    imList[0].H = mul_H.clone();

    //0フレーム目をベースとする
    cv::Mat base = imList[0].image;
//    cv::Mat AffineBase;
//    cv::warpPerspective(base, base, affineH, cv::Size(100, 100));
    cv::Mat result;
    cv::Mat result_pano;

    //パノラマ画像に0フレーム目を貼り付け
    for (int x = 0; x < base.rows; x++) {
        for (int y = 0; y < base.cols; y++) {
            this->PanoramaImage.at<cv::Vec3b>(x, y) = base.at<cv::Vec3b>(x, y);
        }
    }
    cv::Vec3b BLACK(0, 0, 0);
    int frame_counter = 1;
    int x_min, y_min, x_max, y_max;
    for (frame_counter; frame_counter< imList.size();frame_counter++) {

        ImageInfo im = imList[frame_counter];

        //ホモグラフィー掛け合わせ
        mul_H *= im.H;

        //ホモグラフィーの更新
        cv::Mat mul_clone = mul_H.clone();
        imList[frame_counter].mulH = mul_clone;

        //端の4点の斜影位置を求める
        vector<cv::Point2f> edge_points;
        vector<cv::Point2f> points;
        cv::Point2f pt1(0, 0);
        cv::Point2f pt2(im.image.cols, 0);
        cv::Point2f pt3(0, im.image.rows);
        cv::Point2f pt4(im.image.cols, im.image.rows);
        points.push_back(pt1);
        points.push_back(pt2);
        points.push_back(pt3);
        points.push_back(pt4);
        cout << mul_H << endl;
        mycalcWarpedPoint(points, &edge_points, mul_H);

        //パノラマ画像の更新
        x_min = int(edge_points[0].x < edge_points[2].x ? edge_points[0].x : edge_points[2].x);
        x_max = int(edge_points[1].x > edge_points[3].x ? edge_points[1].x : edge_points[3].x);
        y_min = int(edge_points[0].y < edge_points[1].y ? edge_points[0].y : edge_points[1].y);
        y_max = int(edge_points[2].y > edge_points[3].y ? edge_points[2].y : edge_points[3].y);
        x_min = (x_min > 0 ? x_min : 0);
        y_min = (y_min > 0 ? y_min : 0);

        //ワーピング
        cv::warpPerspective(imList[frame_counter].image, result_pano, mul_H,
                            cv::Size(x_max, y_max), CV_HAL_BORDER_CONSTANT);
        this->imList[frame_counter].panorama_scale_im = result_pano;

        cout << frame_counter << " th frame is added to Panorama" << endl;
        for (int x = x_min; x < x_max ; x++) {
            for (int y = y_min; y < y_max ; y++) {
                if ((result_pano.at<cv::Vec3b>(y, x) != BLACK)) {
                    PanoramaImage.at<cv::Vec3b>(y, x) = result_pano.at<cv::Vec3b>(y, x);
                }
            }
        }

        //パノラマの途中経過
        cv::Rect rect(0, 0, x_max, y_max);
        cv::Mat smallPanorama(PanoramaImage, rect);

        if(SHOW_PANORAMA) {
            cv::Mat smallDummy = smallPanorama.clone();
            cv::resize(smallDummy, smallDummy, cv::Size(), 0.1, 0.1);
            cv::imshow("small panorama", smallDummy);
//            cv::imshow("im", im.image);
            cv::waitKey(0);
        }

    }

    cv::Rect rect(0, 0, x_max, y_max);
    cv::Mat panorama(PanoramaImage, rect);

    //画面に表示できるようにリサイズ
//    cv::imwrite(_result_folder + "panorama.jpg", panorama);
    this->OriginalPanorama = panorama.clone();

    this->Panorama_width = panorama.cols;
    this->Panorama_height = panorama.rows;

    cv::Mat smallPanorama;
    cv::resize(panorama, smallPanorama, cv::Size(), smallPanorama_width / Panorama_width,
               smallPanorama_height / Panorama_height);
    this->smallPanoramaImage = smallPanorama;

    cv::imshow("panorama", this->smallPanoramaImage);
    cv::waitKey();
    cv::imwrite(_result_folder + "/panorama.jpg", this->smallPanoramaImage);

    cv::destroyAllWindows();

    cout << "[Generate panoramic image finished]" << endl;

}

void Panorama::generateInversePanorama() {

    cout << "panorama" << endl;

    vector<cv::Scalar> colors;
    yagi::setColor(&colors);

    //パノラマ画像生成
    cv::Mat mul_H = cv::Mat::zeros(3, 3, CV_64F);
    mul_H.at<double>(0, 0) = 1;
    mul_H.at<double>(1, 1) = 1;
    mul_H.at<double>(2, 2) = 1;

//    //すべてのホモグラフィーを逆行列に変換
//    int frame = 0;
//    for (ImageInfo im: imList){
//        if(frame < imList.size() - 1){
//            im.inverseH = im.inverseH.inv();
//        }else{
//            cv::Mat lastH = mul_H.clone();
//            im.inverseH = lastH.inv();
//        }
//        cout << frame << " " << im.inverseH << endl;
//        frame++;
//    }

    //最終フレームをベースとする
    cv::Mat base = imList[imList.size() - 1].image;
    cv::Mat result;
    cv::Mat result_pano;

    //パノラマ画像に最終フレームを貼り付け
    for (int x = 0; x < base.rows; x++) {
        for (int y = 0; y < base.cols; y++) {
            this->PanoramaImage.at<cv::Vec3b>(x, -(y - base.cols)) = base.at<cv::Vec3b>(x, y);
        }
    }

    //最終フレームから開始
    int frame_counter = imList.size() - 1;
    int x_min, y_min, x_max, y_max;
    for (frame_counter; frame_counter >= 0; frame_counter--) {
        ImageInfo im = imList[frame_counter];

        //ホモグラフィー掛け合わせ
        mul_H = mul_H*im.inverseH;

        //ホモグラフィーの更新
        cv::Mat mul_clone = mul_H.clone();
        imList[frame_counter].mulInvH = mul_clone;

        //端の4点の斜影位置を求める
        vector<cv::Point2f> edge_points;
        vector<cv::Point2f> points;
        cv::Point2f pt1(0, 0);
        cv::Point2f pt2(im.image.cols, 0);
        cv::Point2f pt3(0, im.image.rows);
        cv::Point2f pt4(im.image.cols, im.image.rows);
        points.push_back(pt1);
        points.push_back(pt2);
        points.push_back(pt3);
        points.push_back(pt4);
        mycalcWarpedPoint(points, &edge_points, mul_H);

//        //パノラマ画像の更新
        cv::Vec3b BLACK(0, 0, 0);
//        for (int i = 0; i < 4; i++){
//            cv::circle(this->imList[imList.size() - 1].image, edge_points[i], 5, BLACK);
//            cout << edge_points[i] << endl;
//        }
//        cv::imshow("edge", this->imList[imList.size() - 1].image);
//        cv::waitKey();

        x_min = (edge_points[0].x < edge_points[2].x ? edge_points[0].x : edge_points[2].x);
        x_max = (edge_points[1].x > edge_points[3].x ? edge_points[1].x : edge_points[3].x);
        y_min = (edge_points[0].y < edge_points[1].y ? edge_points[0].y : edge_points[1].y);
        y_max = (edge_points[2].y > edge_points[3].y ? edge_points[2].y : edge_points[3].y);
        x_min = (x_min > 0 ? x_min : 0);
        y_min = (y_min > 0 ? y_min : 0);
//
//        //ワーピング
//        cout << x_max << " " << y_max << endl;
        cv::warpPerspective(imList[frame_counter].image, result_pano, mul_H, cv::Size(x_max, y_max));
        this->imList[frame_counter].panorama_scale_im = result_pano;
        cout << frame_counter << " th frame is added to Panorama" << endl;
        for (int x = x_min; x < x_max; x++) {
            for (int y = y_min; y < y_max; y++) {
                if (result_pano.at<cv::Vec3b>(y, x) != BLACK) {
                    PanoramaImage.at<cv::Vec3b>(y, x) = result_pano.at<cv::Vec3b>(y, x);
                }
            }
        }
//
        //パノラマの途中経過
        cv::Rect rect(0, 0, 640, 320);
        cv::Mat smallPanorama(PanoramaImage, rect);
        cv::imshow("small panorama", smallPanorama);
        cv::waitKey(0);

//        if (frame_counter == (imList.size() - 1)) {
//            break;
//        }
    }

    //最後のフレームの x_max, y_maxに合わせてパノラマ画像をクロップ
//    cv::Rect rect(0, 0, x_max, y_max);
//    cv::Mat smallPanorama(PanoramaImage, rect);

    //画面に表示できるようにリサイズ
//    cv::imwrite("../panoramaImage/" + this->_video_name + ".jpg", smallPanorama);
//    this->OriginalPanorama = smallPanorama.clone();

    //150フレーム目へワーピング
//    cv::Mat affine = cv::Mat::zeros(2, 3, CV_64F);
//    affine.at<double>(0,0) = 1;
//    affine.at<double>(1,1) = 1;
//    affine.at<double>(0,2) = 2000;
//    affine.at<double>(1,2) = 2000;
//
//
//    cv::warpPerspective( this->OriginalPanorama, this->OriginalPanorama, imList[230].H.inv(),
//                         cv::Size(10000, 5000));
//        cv::warpAffine(this->OriginalPanorama, this->OriginalPanorama, affine,
//                   cv::Size(10000, 5000));
//    cv::imshow("panorama", OriginalPanorama);
//    cv::imwrite("a.jpg", this->OriginalPanorama);
//    cv::waitKey();

//    this->Panorama_width = smallPanorama.cols;
//    this->Panorama_height = smallPanorama.rows;
//    this->Panorama_width = smallPanorama_width / smallPanorama.cols;
//    this->Panorama_height = smallPanorama_height / smallPanorama.rows;
//    cv::resize(smallPanorama, smallPanorama, cv::Size(), smallPanorama_width / smallPanorama.cols,
//               smallPanorama_height / smallPanorama.rows);
//    this->PanoramaImage = smallPanorama;
//
//    cv::imshow("panorama_inverse", this->PanoramaImage);
//    cv::waitKey();
//    cv::destroyAllWindows();
}

void Panorama::generateNthFramePanorama() {

    cout << "panorama" << endl;
    int center_frame = 150;

    vector<cv::Scalar> colors;
    yagi::setColor(&colors);

    //center_frame目をベースとする
    cv::Mat base = imList[center_frame].image;
    cv::Mat result;
    cv::Mat result_pano;

    //パノラマ画像に0フレーム目を貼り付け
    for (int x = 0; x < base.rows; x++) {
        for (int y = 0; y < base.cols; y++) {
            this->PanoramaImage.at<cv::Vec3b>(x, y) = base.at<cv::Vec3b>(x, y);
        }
    }

    //パノラマ画像生成
    cv::Mat mul_H = cv::Mat::zeros(3, 3, CV_64F);
    mul_H.at<double>(0, 0) = 1;
    mul_H.at<double>(1, 1) = 1;
    mul_H.at<double>(2, 2) = 1;

    int frame_counter = 0;
    int x_min, y_min, x_max, y_max;

    for (int i = center_frame; i > 0; i--) {

        ImageInfo im = imList[i];

        //ホモグラフィー掛け合わせ
        mul_H *= im.H.inv();

        //ホモグラフィーの更新
        cv::Mat mul_clone = mul_H.clone();
        imList[i].mulH = mul_clone;
    }

    cv::Mat mul_H2 = cv::Mat::zeros(3, 3, CV_64F);
    mul_H2.at<double>(0, 0) = 1;
    mul_H2.at<double>(1, 1) = 1;
    mul_H2.at<double>(2, 2) = 1;

    for (int i = center_frame; i < imList.size() - 1; i++) {

        ImageInfo im = imList[i];

        //ホモグラフィー掛け合わせ
        mul_H2 *= im.H;

        //ホモグラフィーの更新
        cv::Mat mul_clone = mul_H2.clone();
        imList[i].mulH = mul_clone;
    }

    //端の4点の斜影位置を求める
    vector<cv::Point2f> base_translation;
    vector<cv::Point2f> points;
    cv::Point2f pt1(0, 0);
    cv::Point2f pt2(imList[0].image.cols, 0);
    cv::Point2f pt3(0, imList[0].image.rows);
    cv::Point2f pt4(imList[0].image.cols, imList[0].image.rows);
    points.push_back(pt1);
    points.push_back(pt2);
    points.push_back(pt3);
    points.push_back(pt4);
    mycalcWarpedPoint(points, &base_translation, imList[1].mulH);

    //パノラマ画像の更新
    cv::Vec3b BLACK(0, 0, 0);
    x_min = (base_translation[0].x < base_translation[2].x ? base_translation[0].x : base_translation[2].x);
    x_max = (base_translation[1].x > base_translation[3].x ? base_translation[1].x : base_translation[3].x);
    y_min = (base_translation[0].y < base_translation[1].y ? base_translation[0].y : base_translation[1].y);
    y_max = (base_translation[2].y > base_translation[3].y ? base_translation[2].y : base_translation[3].y);
    cout << x_min << " " << x_max << " " << y_min << " " << y_max << endl;

    cv::Mat affine = cv::Mat::zeros(2, 3, CV_64F);
    affine.at<double>(0,0) = 1;
    affine.at<double>(1,1) = 1;
    affine.at<double>(0,2) = -base_translation[0].x;
    affine.at<double>(1,2) = -base_translation[0].y;

    cv::warpAffine(this->PanoramaImage, this->PanoramaImage, affine,
                   cv::Size(imList[0].image.cols - base_translation[0].x,
                            imList[0].image.rows - base_translation[0].y));
    cv::imshow("samll panorama", this->PanoramaImage);
    cv::waitKey(0);

    for (int i = 1; i < imList.size(); i++){

        ImageInfo im = imList[i];

        //端の4点の斜影位置を求める
        vector<cv::Point2f> edge_points;
        mycalcWarpedPoint(points, &edge_points, im.H);

        //パノラマ画像の更新
        cv::Vec3b BLACK(0, 0, 0);
        x_min = (edge_points[0].x < edge_points[2].x ? edge_points[0].x : edge_points[2].x);
        x_max = (edge_points[1].x > edge_points[3].x ? edge_points[1].x : edge_points[3].x);
        y_min = (edge_points[0].y < edge_points[1].y ? edge_points[0].y : edge_points[1].y);
        y_max = (edge_points[2].y > edge_points[3].y ? edge_points[2].y : edge_points[3].y);
        x_min -= base_translation[0].x;
        y_min -= base_translation[0].y;
        x_max -= base_translation[0].x;
        y_max -= base_translation[0].y;
        x_min = (x_min > 0 ? x_min : 0);
        y_min = (y_min > 0 ? y_min : 0);
        cout << x_min << " " << x_max << " " << y_min << " " << y_max << endl;

        //ワーピング
        cv::warpAffine(imList[++frame_counter].image, result_pano, affine, cv::Size(imList[0].image.cols - base_translation[0].x,
                                                                  imList[0].image.rows - base_translation[0].y));
        cv::warpPerspective( result_pano, result_pano, im.H,
                            cv::Size(x_max, y_max));

        cv::imshow("image", result_pano);
        cv::waitKey(0);
        this->imList[frame_counter].panorama_scale_im = result_pano;

        cout << frame_counter++ << " th frame is added to Panorama" << endl;
        for (int x = x_min; x < x_max; x++) {
            for (int y = y_min; y < y_max; y++) {
                if (PanoramaImage.at<cv::Vec3b>(y, x) == BLACK) {
                    PanoramaImage.at<cv::Vec3b>(y, x) = result_pano.at<cv::Vec3b>(y, x);
                }
            }
        }

        //パノラマの途中経過
        cv::Rect rect(0, 0, x_max, y_max);
        cv::Mat smallPanorama(PanoramaImage, rect);
        cv::imshow("samll panorama", smallPanorama);
        cv::waitKey(0);


        if (frame_counter == (imList.size() - 1)) {
            break;
        }
    }

    //最後のフレームの x_max, y_maxに合わせてパノラマ画像をクロップ

    cv::Rect rect(0, 0, x_max, y_max);
    cv::Mat smallPanorama(PanoramaImage, rect);

    //画面に表示できるようにリサイズ
    cv::imwrite("../panoramaImage/" + this->_video_name + ".jpg", smallPanorama);
    this->OriginalPanorama = smallPanorama.clone();
    this->Panorama_width = smallPanorama.cols;
    this->Panorama_height = smallPanorama.rows;
    this->Panorama_width = smallPanorama_width / smallPanorama.cols;
    this->Panorama_height = smallPanorama_height / smallPanorama.rows;
    cv::resize(smallPanorama, smallPanorama, cv::Size(), smallPanorama_width / smallPanorama.cols,
               smallPanorama_height / smallPanorama.rows);
    this->PanoramaImage = smallPanorama;

    cv::imshow("panorama", this->PanoramaImage);
    cv::waitKey();
    cv::destroyAllWindows();
}


void Panorama::getOverviewHomography(){

    vector<cv::Point2f> cornerPoints;
    vector<cv::Point2f> panoramaCornerPoints;

    //フィニッシュラインの点をパノラマに投影
    vector<cv::Point2f> panoramaFinishLineCornerPoints;
    yagi::mycalcWarpedPoint(this->finishLineCornerPoints,
                            &panoramaFinishLineCornerPoints,
                            imList[this->finalLineImageNum].mulH);

    cornerPoints.push_back(this->startLineCornerPoints[0]);
    cornerPoints.push_back(this->startLineCornerPoints[1]);
    cornerPoints.push_back(panoramaFinishLineCornerPoints[0]);
    cornerPoints.push_back(panoramaFinishLineCornerPoints[1]);

    for (int p = 0; p < 4; p++){
        cornerPoints[p].x = cornerPoints[p].x * (this->smallPanorama_width/this->Panorama_width);
        cornerPoints[p].y = cornerPoints[p].y * (this->smallPanorama_height/this->Panorama_height);
    }

    //俯瞰画像のコーナー4点
    vector<cv::Point2f> overviewCornerPoints;
    cv::Point2f pt1(0, 0);
    cv::Point2f pt2(0, OverView.rows);
    cv::Point2f pt3(OverView.cols, 0);
    cv::Point2f pt4(OverView.cols, OverView.rows);
    overviewCornerPoints.push_back(pt1);
    overviewCornerPoints.push_back(pt2);
    overviewCornerPoints.push_back(pt3);
    overviewCornerPoints.push_back(pt4);

    //パノラマ->俯瞰画像へのホモグラフィー
    cv::Mat H = cv::findHomography(cornerPoints, overviewCornerPoints);
    H.convertTo(H, CV_32F);
    this->overView_H = H;
    cout << H << endl;

    //俯瞰画像を表示
    cv::Mat overviewImage;
    cv::warpPerspective(this->PanoramaImage, overviewImage, this->overView_H, this->OverView.size());

    cv::resize(overviewImage, overviewImage, cv::Size(), 0.5, 0.5);
    cv::imshow("overview Image first", overviewImage);
    cv::imwrite("../images/" + this->_video_name + "/OverView.jpg", overviewImage);
    cv::waitKey();
    this->overviewPanorama = overviewImage;

}


void Panorama::getInverseOverviewHomography(){

    vector<cv::Point2f> cornerPoints;
    vector<cv::Point2f> panoramaCornerPoints;

    //スタートラインの点をパノラマに投影
    vector<cv::Point2f> panoramaStartLineCornerPoints;
    yagi::mycalcWarpedPoint(this->startLineCornerPoints,
                            &panoramaStartLineCornerPoints,
                            imList[0].mulInvH);

    cornerPoints.push_back(panoramaStartLineCornerPoints[0]);
    cornerPoints.push_back(panoramaStartLineCornerPoints[1]);
    cornerPoints.push_back(this->finishLineCornerPoints[0]);
    cornerPoints.push_back(this->finishLineCornerPoints[1]);

//    for (int p = 0; p < 4; p++){
//        cornerPoints[p].x = cornerPoints[p].x * this->Panorama_width;
//        cornerPoints[p].y = cornerPoints[p].y * this->Panorama_height;
//    }

    //俯瞰画像のコーナー4点
    vector<cv::Point2f> overviewCornerPoints;
    cv::Point2f pt1(0, 0);
    cv::Point2f pt2(0, OverView.rows);
    cv::Point2f pt3(OverView.cols, 0);
    cv::Point2f pt4(OverView.cols, OverView.rows);
    overviewCornerPoints.push_back(pt1);
    overviewCornerPoints.push_back(pt2);
    overviewCornerPoints.push_back(pt3);
    overviewCornerPoints.push_back(pt4);

    //パノラマ->俯瞰画像へのホモグラフィー
    cv::Mat H = cv::findHomography(cornerPoints, overviewCornerPoints);
    H.convertTo(H, CV_32F);
    this->inv_overView_H = H;

    //俯瞰画像を表示
    cv::Mat overviewImage;
//    cv::warpPerspective(this->imList[this->imList.size() - 2
//                        ].image, overviewImage, this->inv_overView_H, this->OverView.size());
    cv::Mat dummy;
    cv::warpPerspective(this->imList[0].image, dummy, this->imList[0].mulInvH , this->OverView.size());
    cv::warpPerspective(this->imList[0].image, overviewImage, this->inv_overView_H, this->OverView.size());


    cv::resize(overviewImage, overviewImage, cv::Size(), 0.5, 0.5);
    cv::imshow("overview Image", overviewImage);
//    cv::imwrite("../panoramaImage/OverView_" + this->_video_name + ".jpg", overviewImage);
    cv::waitKey();
//    this->overviewPanorama = overviewImage;

}


void Panorama::projectTrackLine() {

    //10m毎の線
    vector<cv::Point2f> inline10mPoints;
    vector<cv::Point2f> outline10mPoints;

    for (int i = 0; i <= 10; i++) {
        cv::Point2f pt1((this->OverView.cols / 10) * i, 0);
        cv::Point2f pt2((this->OverView.cols / 10) * i, this->OverView.rows);
        inline10mPoints.push_back(pt1);
        outline10mPoints.push_back(pt2);
    }

    //逆行列で元画像に変換
    cv::Scalar BLUE(255, 0, 0);
    vector<cv::Point2f> panoramaInline10mPoints;
    vector<cv::Point2f> panoramaOutline10mPoints;

    //パノラマへの投影
    mycalcfloatWarpedPoint(inline10mPoints, &panoramaInline10mPoints, this->overView_H.inv());
    mycalcfloatWarpedPoint(outline10mPoints, &panoramaOutline10mPoints, this->overView_H.inv());


    //resizeの分投影点の各座標をx倍する
    for (int pt = 0; pt < panoramaInline10mPoints.size(); pt++) {
        cv::circle(this->PanoramaImage, panoramaInline10mPoints[pt], 2, BLUE, 2);
        panoramaInline10mPoints[pt].x = panoramaInline10mPoints[pt].x * (1 / this->Panorama_width);
        panoramaInline10mPoints[pt].y = panoramaInline10mPoints[pt].y * (1 / this->Panorama_height);
        cv::circle(this->PanoramaImage, panoramaInline10mPoints[pt], 3, BLUE, 3);
    }

    for (int pt = 0; pt < panoramaOutline10mPoints.size(); pt++) {
        cv::circle(this->PanoramaImage, panoramaOutline10mPoints[pt], 2, BLUE, 2);
        panoramaOutline10mPoints[pt].x = panoramaOutline10mPoints[pt].x * (1 / this->Panorama_width);
        panoramaOutline10mPoints[pt].y = panoramaOutline10mPoints[pt].y * (1 / this->Panorama_height);
        cv::circle(this->PanoramaImage, panoramaOutline10mPoints[pt], 3, BLUE, 3);
    }

//    //俯瞰画像を表示
//    cv::Mat overviewImage;
//    cv::warpPerspective(this->PanoramaImage, overviewImage, this->overView_H, this->OverView.size());
//    cv::resize(overviewImage, overviewImage, cv::Size(), 0.5, 0.5);
//    cv::imshow("overview Image", overviewImage);
//    cv::imwrite("../panoramaImage/OverView_" + this->_video_name + ".jpg", overviewImage);
//    cv::waitKey();
//    this->overviewPanorama = overviewImage;

    //ImageInfoに格納
    this->panoramaInline10mPoints = panoramaInline10mPoints;
    this->panoramaOutline10mPoints = panoramaOutline10mPoints;


    //各フレームに投影
    for (int i = 0; i < imList.size() - 1; i++) {
        ImageInfo im = imList[i];
        vector<cv::Point2f> frameIn10mPoints;
        vector<cv::Point2f> frameOut10mPoints;

        mycalcWarpedPoint(panoramaInline10mPoints, &frameIn10mPoints, im.mulH.inv());
        mycalcWarpedPoint(panoramaOutline10mPoints, &frameOut10mPoints, im.mulH.inv());

        for (int pt_n = 0; pt_n < frameIn10mPoints.size(); pt_n++) {
            cv::Point2f inPt = frameIn10mPoints[pt_n];
            cv::Point2f outPt = frameOut10mPoints[pt_n];

            cv::circle(im.image, inPt, 2, BLUE, 2);
            cv::circle(im.image, outPt, 2, BLUE, 2);

            cv::line(im.image, inPt, outPt, BLUE);
            cv::putText(im.image, to_string( pt_n * 10), inPt,
                        cv::FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(0, 0, 200), 2, CV_AA);

            std::pair<cv::Point2f, cv::Point2f> line(inPt, outPt);
            imList[i].lines10m.push_back(line);
        }
//        cv::imshow("10m line", im.image);
//        cv::waitKey();
    }
}


void Panorama::projectInverseTrackLine() {

    //10m毎の線
    vector<cv::Point2f> inline10mPoints;
    vector<cv::Point2f> outline10mPoints;

    for (int i = 0; i <= 10; i++) {
        cv::Point2f pt1((this->OverView.cols / 10) * i, 0);
        cv::Point2f pt2((this->OverView.cols / 10) * i, this->OverView.rows);
        inline10mPoints.push_back(pt1);
        outline10mPoints.push_back(pt2);
    }

    //逆行列で元画像に変換
    cv::Scalar BLUE(255, 0, 0);
    vector<cv::Point2f> panoramaInline10mPoints;
    vector<cv::Point2f> panoramaOutline10mPoints;

    //パノラマへの投影
    mycalcfloatWarpedPoint(inline10mPoints, &panoramaInline10mPoints, this->inv_overView_H.inv());
    mycalcfloatWarpedPoint(outline10mPoints, &panoramaOutline10mPoints, this->inv_overView_H.inv());


    //resizeの分投影点の各座標をx倍する
    for (int pt = 0; pt < panoramaInline10mPoints.size(); pt++) {
        cout <<  panoramaInline10mPoints[pt] << endl;
        cv::circle(this->imList[imList.size() - 1].image, panoramaInline10mPoints[pt], 2, BLUE, 2);
    }

    for (int pt = 0; pt < panoramaOutline10mPoints.size(); pt++) {
        cv::circle(this->imList[imList.size() - 1].image, panoramaOutline10mPoints[pt], 2, BLUE, 2);
    }

    cv::imshow("last image", this->imList[imList.size() - 1].image);
    cv::waitKey();

//    //俯瞰画像を表示
//    cv::Mat overviewImage;
//    cv::warpPerspective(this->PanoramaImage, overviewImage, this->overView_H, this->OverView.size());
//    cv::resize(overviewImage, overviewImage, cv::Size(), 0.5, 0.5);
//    cv::imshow("overview Image", overviewImage);
//    cv::imwrite("../panoramaImage/OverView_" + this->_video_name + ".jpg", overviewImage);
//    cv::waitKey();
//    this->overviewPanorama = overviewImage;

    //ImageInfoに格納
    this->panoramaInline10mPoints = panoramaInline10mPoints;
    this->panoramaOutline10mPoints = panoramaOutline10mPoints;


    //各フレームに投影
    for (int i = 0; i < imList.size() - 1; i++) {
        ImageInfo im = imList[i];
        vector<cv::Point2f> frameIn10mPoints;
        vector<cv::Point2f> frameOut10mPoints;

        mycalcWarpedPoint(panoramaInline10mPoints, &frameIn10mPoints, im.mulInvH.inv());
        mycalcWarpedPoint(panoramaOutline10mPoints, &frameOut10mPoints, im.mulInvH.inv());

        for (int pt_n = 0; pt_n < frameIn10mPoints.size(); pt_n++) {
            cv::Point2f inPt = frameIn10mPoints[pt_n];
            cv::Point2f outPt = frameOut10mPoints[pt_n];

            cv::circle(im.image, inPt, 2, BLUE, 2);
            cv::circle(im.image, outPt, 2, BLUE, 2);

            cv::line(im.image, inPt, outPt, BLUE);
            cv::putText(im.image, to_string( pt_n * 10), inPt,
                        cv::FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(0, 0, 200), 2, CV_AA);

            std::pair<cv::Point2f, cv::Point2f> line(inPt, outPt);
            imList[i].lines10m.push_back(line);
        }
        cv::imshow("10m line", im.image);
        cv::waitKey();
    }
}


void Panorama::legLaneDist() {
    for(int i = 0; i < 2; i++){
        int frame_num = 0;
        if(i == 0) { //右足の処理
            for (ImageInfo im: imList) {
                imList[frame_num].RlineDist = yagi::distPoint2Line(im.originalRfoot, im.grads[this->TARGET_RUNNER_ID],
                                                                            im.segments[this->TARGET_RUNNER_ID]);
                frame_num ++;
            }
        }else if(i == 1){ // 左足の処理

            for (ImageInfo im: imList) {
                imList[frame_num].LlineDist = yagi::distPoint2Line(im.originalLfoot, im.grads[this->TARGET_RUNNER_ID],
                                                                            im.segments[this->TARGET_RUNNER_ID]);
                frame_num ++;
            }
        }
    }
}


void Panorama::candidateStepFrame() {
    string filename;
    for(int i = 0; i < 2; i++) {
        int frame_num = 0;
        if (i == 0) { //右足の処理
            filename = "../images/" + this->_video_name + "/Data/right_leg_dist.txt";
            ofstream saveText(filename);
            for (ImageInfo im: imList) {
                if (frame_num > 2) {
                    float dist = im.RlineDist;
                    cout << "step possition " << im.overviewRfoot << endl;
                    // 直線との距離が7フレームで最小
                    if (((dist < imList[frame_num + 1].RlineDist) &&
                         (dist < imList[frame_num + 2].RlineDist) &&
                         (dist < imList[frame_num + 3].RlineDist) &&
                         (dist < imList[frame_num - 1].RlineDist) &&
                         (dist < imList[frame_num - 2].RlineDist) &&
                         (dist < imList[frame_num - 3].RlineDist))) {
                        // 足位置が0m以上かつ100m以下
                        if ((im.overviewRfoot.x > 0) && (im.overviewRfoot.x < 1000)) {
                            imList[frame_num].Rstep = true;
                            saveText << frame_num << " " << im.overviewRfoot << " " <<imList[frame_num].RlineDist << " @" << endl;
                            frame_num++;
                            cout << "Right Step " << frame_num << endl;
                            continue;
                        }
                    }
                    saveText << frame_num << " " << im.overviewRfoot << " " << imList[frame_num].RlineDist << endl;
                }
                frame_num++;
            }
        } else if (i == 1) { // 左足の処理
            filename = "../images/" + this->_video_name + "/Data/left_leg_dist.txt";
            ofstream saveText(filename);
            for (ImageInfo im: imList) {
                if (frame_num > 2) {
                    float dist = im.LlineDist;
                    // 直線との距離が7フレームで最小
                    if (((dist < imList[frame_num + 1].LlineDist) &&
                         (dist < imList[frame_num + 2].LlineDist) &&
                         (dist < imList[frame_num + 3].LlineDist) &&
                         (dist < imList[frame_num - 1].LlineDist) &&
                         (dist < imList[frame_num - 2].LlineDist) &&
                         (dist < imList[frame_num - 3].LlineDist))) {
                        // 足位置が0m以上かつ100m以下
                        if ((im.overviewLfoot.x > 0) && (im.overviewLfoot.x < 1000)) {
                            imList[frame_num].Lstep = true;
                            saveText << frame_num << " " << imList[frame_num].LlineDist << " @" << endl;
                            frame_num++;
                            cout << "Left Step " << frame_num << endl;
                            continue;
                        }
                    }
                    saveText << frame_num << " " << imList[frame_num].LlineDist << endl;
                }
                frame_num++;
            }
        }
    }
}

void Panorama::insideLane() {
    cv::Point2f footCoord;
    for (int i = 0; i < imList.size(); i++) {
        ImageInfo im = imList[i];

        float outGrad = im.grads[this->TARGET_RUNNER_ID - 1];
        float inGrad = im.grads[this->TARGET_RUNNER_ID];
        float outSegment = im.segments[this->TARGET_RUNNER_ID - 1];
        float inSegment = im.segments[this->TARGET_RUNNER_ID];

        if (im.Rstep == true) {
            footCoord = im.originalRfoot;
            if (!(footCoord.y - 10 < (outGrad * footCoord.x + outSegment)) ||
                (footCoord.y + 10 > (inGrad * footCoord.x + inSegment))) {
                imList[i].Rstep = false;
            }
        }

        if (im.Lstep == true) {
            footCoord = im.originalLfoot;
            if (!(footCoord.y - 10 < (outGrad * footCoord.x + outSegment)) ||
                (footCoord.y + 10 > (inGrad * footCoord.x + inSegment))) {
                imList[i].Lstep = false;
            }
        }
    }
}


void Panorama::mergeStepID() {
    vector<int> mergedStepID;
    for (int i = 0; i < imList.size(); i++) {
        ImageInfo im = imList[i];

        if (im.Rstep == true) {
            mergedStepID.push_back(i);
            imList[i].stepPoint = true;
        }
    }

    for (int i = 0; i < imList.size(); i++) {
        ImageInfo im = imList[i];

        if (im.Lstep == true) {
            bool found = false;
            for(int id: mergedStepID) {
                if(abs(id - i) < 2){
                    found = true;
                    break;
                }else if(abs(id - i) == 2){
                    if (id > i){
                        imList[id - 1].stepPoint = true;
                        imList[id - 1].Rstep = true;
                    }else{
                        imList[id + 1].stepPoint = true;
                        imList[id + 1].Rstep = true;
                    }
                    found = true;
                    break;
                }
            }
            if(found == false){
                imList[i].stepPoint = true;
                imList[i].Lstep = true;
            }
        }
    }
}

void Panorama::visualizeSteps(){
    cv::Scalar White(255,255,255);
    cv::Mat overview_dummy = cv::Mat::zeros(cv::Size(1000, 200), CV_8UC3);

    //10mラインの描写
    for (int i = 1; i < 10; i++){
        cv::Point2f ptup(i * 100, 0);
        cv::Point2f ptdown(i * 100, 200);
        cv::Scalar color(255,255,255);
        cv::line(overview_dummy, ptup, ptdown, color, 1);
    }

    int frame_num = 0;
    int preStepFrame = 0;
    int text_position = 20;
    for(ImageInfo im: imList){
        if(im.stepPoint){
            text_position *= -1;
            if(this->stepPoints.size() > 0){
                averagePitch += (frame_num - preStepFrame);
            }
            preStepFrame = frame_num;
            cout << "Step Frame: " << frame_num << endl;
            if(im.Rstep) {
                cv::Point2f pt = im.overviewRfoot;
                pt.y = 100;
                cv::circle(overview_dummy, pt, 2, White, 2);

                pt.y += text_position;
                cv::putText(overview_dummy, to_string(frame_num), pt,
                            cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(255, 255, 255), 1, CV_AA);
                this->stepPoints.push_back(im.overviewRfoot);
            }else{
                cv::Point2f pt = im.overviewLfoot;
                pt.y = 100;
                cv::circle(overview_dummy, pt, 2, White, 2);

                pt.y += text_position;
                cv::putText(overview_dummy, to_string(frame_num), pt,
                            cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(255, 255, 255), 1, CV_AA);
                this->stepPoints.push_back(im.overviewLfoot);
            }

            //接地点情報をベクトルに追加
            Step newStep;
            newStep.frame = frame_num;
            newStep.pitch = frame_num - preStepFrame;
            if(im.Rstep)
                newStep.leg = "R";
            else
                newStep.leg = "L";
            newStep.rightPt = im.overviewRfoot;
            newStep.leftPt = im.overviewLfoot;

            this->steps.push_back(newStep);
        }
        frame_num++;
    }
    averagePitch /= (this->stepPoints.size() - 1);
    cout << averagePitch;
    cv::imshow("over view image", overview_dummy);
    cv::waitKey();
}

void Panorama::getAllScaleFootPosition(){
    int frame_num = 0;
    for (ImageInfo im: imList) {

        //ホモグラフィー型変換
        im.mulH.convertTo(imList[frame_num].mulH, CV_64F);
        this->overView_H.convertTo(this->overView_H, CV_64F);

        int humanID = 0;

        //設置点
        cv::Point2f originalRfoot(0,0);
        cv::Point2f originalLfoot(0,0);
        cv::Point2f panoramaRfoot(0,0);
        cv::Point2f panoramaLfoot(0,0);
        cv::Point2f overviewRfoot(0,0);
        cv::Point2f overviewLfoot(0,0);

        //トラッキング対象ランナーの探索
        for (OpenPoseBody hb : im.runnerCandidate) {

            if (hb.humanID == 1) {

                //ランナー関節点
                vector <cv::Point2f> coord = hb.getBodyCoord();
                vector <cv::Point2f> panoramaCoords;
                vector <cv::Point2f> overviewCoords;

                //1,各フレームから1フレーム目へのH(パノラマ)
                mycalcWarpedPoint(coord, &panoramaCoords, imList[frame_num].mulH);
                for(cv::Point2f pt : panoramaCoords){
                    cout << "panorama" << pt << endl;
                }

                //2,リサイズ分の変形
                for (int pt = 0; pt < panoramaCoords.size(); pt++) {
                    panoramaCoords[pt].x *= (this->smallPanorama_width/this->Panorama_width);
                    panoramaCoords[pt].y *= (this->smallPanorama_height/this->Panorama_height);
                }
                for(cv::Point2f pt : panoramaCoords){
                    cout << "small panorama" << pt << endl;
                }

                //3,overviewへの変形
                mycalcWarpedPoint(panoramaCoords, &overviewCoords, this->overView_H);

                for(cv::Point2f pt : overviewCoords){
                    cout << "over view " << pt << endl;
                }
                //右足、左足だけ投影
                overviewCoords[10].x = overviewCoords[10].x * 0.50;
                overviewCoords[10].y = overviewCoords[10].y * 0.50;
                overviewCoords[13].x = overviewCoords[13].x * 0.50;
                overviewCoords[13].y = overviewCoords[13].y * 0.50;

                // 接地点の保存
                originalRfoot = coord[10];
                originalLfoot = coord[13];
                panoramaRfoot = panoramaCoords[10];
                panoramaLfoot = panoramaCoords[13];
                overviewRfoot = overviewCoords[10];
                overviewLfoot = overviewCoords[13];

//                cv::circle(this->OverView, overviewLfoot, 2, cv::Scalar(255,0,0), 2);
//                cv::circle(this->OverView, overviewRfoot, 2, cv::Scalar(0,255,0), 2);
//                cv::imshow("overstep", this->OverView);
//                cv::waitKey();

            }
            humanID++;

        }
        imList[frame_num].originalRfoot = originalRfoot;
        imList[frame_num].originalLfoot = originalLfoot;
        imList[frame_num].panoramaRfoot = panoramaRfoot;
        imList[frame_num].panoramaLfoot = panoramaLfoot;
        imList[frame_num].overviewRfoot = overviewRfoot;
        imList[frame_num].overviewLfoot = overviewLfoot;

        frame_num++;
    }
}

void Panorama::averageCompletion(){
//    // ave dist よりも大幅に小さい箇所は削除
//    float deleteThreshold = aveDist * 0.5;
//    pre_x = 0;
//    stepNum = 0;
//
//    for (int i = 0; i < stepPoints.size(); i++){
//        stepPoint step = stepPoints[stepNum];
//        //１フレーム目は無視
//        bool deleted = false;
//
//        //最初の100mはavethreshold変化
//        if (step.step.x < 100) {
//            // aveDist よりも大幅に小さいものは削除
//            float stepLength = abs(step.step.x - pre_x);
//            if (stepNum == 0) {
//                stepLength = step.step.x;
//            }
//            if (stepLength < (aveDist * 0.5)) {
//                stepPoints.erase(stepPoints.begin() + stepNum);
//                deleted = true;
//
//            }
//        }else if (step.step.x > 100){
//            // aveDist よりも大幅に小さいものは削除
//            float stepLength = abs(step.step.x - pre_x);
//            if (stepLength < (aveDist * 0.75)) {
//                stepPoints.erase(stepPoints.begin() + stepNum);
//                deleted = true;
//
//            }
//        }
//
//        if (deleted == false) {
//            pre_x = step.step.x;
//            stepNum++;
//        }
//    }
//
//    // ave dist よりも大幅に大きい箇所は追加
//    float addThreshold = aveDist * 1.7;
//    stepNum = 0;
//    bool lastframe = false;
//    for (int i = 0; i <= stepPoints.size(); i++){
//        float stepLength = 0;
//        stepPoint step = stepPoints[stepNum];
//
//        //１フレーム目は無視
//
//        //最後のフレームなら
//        if (i == stepPoints.size()) {
//
//            //aveDistの更新
//            aveDist = 0;
//            float prex = 0;
//            int stepID = 0;
//            int stepNums = 0;
//            for (stepPoint steps: stepPoints){
//                if (steps.step.x > 100){
//                    stepPoints[stepID].dist = steps.step.x - prex;
//                    aveDist+=steps.dist;
//                    stepNums++;
//                }
//                prex = steps.step.x;
//                stepID++;
//            }
//            aveDist /= stepNums;
//            stepLength = abs(float(1000.0 - stepPoints[stepPoints.size()-1].step.x));
//
//            if (stepLength > (aveDist)) {
//                cv::Point2f addpt(stepPoints[stepPoints.size()-1].step.x + aveDist, step.step.y);
//                stepPoint addStep;
//                addStep.step = addpt;
//                addStep.dist = aveDist;
//                addStep.frame = stepPoints[stepPoints.size() - 1].frame + ((stepPoints[stepPoints.size() - 2].frame) - (stepPoints[stepPoints.size() - 3].frame));
//                stepPoints.insert(stepPoints.begin() + stepNum, addStep);
//                lastframe = true;
//
//            }
//        }else {
//
//            if (step.step.x < 100){
//                // aveDist よりも大幅に小さいものは削除
//                stepLength = abs(step.step.x - pre_x);
//                if (stepNum == 0){
//                    stepLength = step.step.x;
//                }
//                cout << "step length = " << stepLength << endl;
//
//                //最初の100mはthreshold変化
//                if (stepLength > (aveDist)) {
//
//                    cv::Point2f addpt;
//                    if (stepLength > (aveDist * 2)) {
//                        addpt.x = (stepPoints[i - 1].step.x + (aveDist * 0.8));
//                        addpt.y = step.step.y;
//                    }else {
//                        addpt.x = ((stepPoints[i - 1].step.x + stepPoints[i].step.x)/2);
//                        addpt.y = step.step.y;
//                    }
//                    stepPoint addStep;
//                    addStep.step = addpt;
//                    addStep.dist = step.dist / 2;
//
//
//                    addStep.frame = float(stepPoints[i - 1].frame + ((step.frame) - stepPoints[i - 1].frame)/2);
//                    if (stepNum == 0){
//                        addStep.frame = stepPoints[i].frame - (stepPoints[i + 1].frame - stepPoints[i].frame);
//                    }
//                    stepPoints.insert(stepPoints.begin() + stepNum, addStep);
//
//                }
//
//            }else if (step.step.x > 100){
//                // aveDist よりも大幅に小さいものは削除
//                stepLength = abs(step.step.x - pre_x);
//
//                if (stepLength > (addThreshold)) {
//                    cv::Point2f addpt;
//                    if (stepLength > (aveDist * 2)) {
//                        addpt.x = (stepPoints[i - 1].step.x + (aveDist));
//                        addpt.y = step.step.y;
//                    }else {
//                        addpt.x = ((stepPoints[i - 1].step.x + stepPoints[i].step.x)/2);
//                        addpt.y = step.step.y;
//                    }
//                    stepPoint addStep;
//                    addStep.step = addpt;
//                    addStep.dist = step.dist / 2;
//                    addStep.frame = float(stepPoints[i - 1].frame + ((step.frame) - stepPoints[i - 1].frame)/2);
//                    stepPoints.insert(stepPoints.begin() + stepNum, addStep);
//
//                }
//            }
//        }
//
//
//        pre_x = stepPoints[stepNum].step.x;
//        stepNum ++;
//    }
//
}

void Panorama::calculateStrideLength(){
    cv::Point2d firstPt(0, 0);
    cv::Point2d previousPt(0, 0);
    this->stepPoints.push_back(firstPt);
    for(ImageInfo im: imList){
        if(im.stepPoint == true){
            cv::Point2d addPoint(this->stridePoints.size(), 0);
            if(this->strideLength.size() == 0){
                if(im.Rstep) {
                    this->strideLength.push_back(im.overviewRfoot.x);
                    previousPt = im.overviewRfoot;
                }else {
                    this->strideLength.push_back(im.overviewLfoot.x);
                    previousPt = im.overviewLfoot;
                }
            }else{
                if(im.Rstep) {
                    this->strideLength.push_back(im.overviewRfoot.x - previousPt.x);
                    previousPt = im.overviewRfoot;
                }else {
                    this->strideLength.push_back(im.overviewLfoot.x - previousPt.x);
                    previousPt = im.overviewLfoot;
                }
            }
            addPoint.y = this->strideLength[this->strideLength.size() - 1];
            this->stridePoints.push_back(addPoint);
        }
    }
}

void Panorama::visualizeStride(){
    cv::Mat graph = cv::Mat::ones(1000, 1000, CV_8UC3);
    for(cv::Point2d stride: this->stridePoints){
        stride.x *= 10;
        cout << stride << endl;
        cv::circle(graph, stride, 2, cv::Scalar(255,255,255), 2);
    }
//    cv::imshow("stride graph", graph);
//    cv::waitKey();
}


void Panorama::pitchCompletion(){
//    int frame_num = 0;
//    int step_num = 0;
//    int pre_step_frame = 0;
//    for(ImageInfo im: imList){
//        if(im.stepPoint){
//            if(step_num > 0){
//
//            }else {
//                if (abs((frame_num - pre_step_frame) - averagePitch) > 1) {
//                    text_position *= -1;
//                    if (this->stepPoints.size() > 0) {
//                        averagePitch += (frame_num - preStepFrame);
//                    }
//                    preStepFrame = frame_num;
//                    cout << "Step Frame: " << frame_num << endl;
//                    if (im.Rstep) {
//                        cv::Point2f pt = im.overviewRfoot;
//                        pt.y = 100;
//                        cv::circle(overview_dummy, pt, 2, White, 2);
//
//                        pt.y += text_position;
//                        cv::putText(overview_dummy, to_string(frame_num), pt,
//                                    cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(255, 255, 255), 1, CV_AA);
//                        this->stepPoints.push_back(im.overviewRfoot);
//                    } else {
//                        cv::Point2f pt = im.overviewLfoot;
//                        pt.y = 100;
//                        cv::circle(overview_dummy, pt, 2, White, 2);
//
//                        pt.y += text_position;
//                        cv::putText(overview_dummy, to_string(frame_num), pt,
//                                    cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(255, 255, 255), 1, CV_AA);
//                        this->stepPoints.push_back(im.overviewLfoot);
//                    }
//                }
//            }
//            step_num++;
//            pre_step_frame = frame_num;
//        }
//        frame_num++;
//    }
}


void Panorama::projectOverview() {

    //色ベクトル
    vector<cv::Scalar> colors;
    setColor(&colors);

    //トラッキング対象ランナーの
    //１原画像での足位置,２パノラマスケールでの足位置,３俯瞰画像での足位置を保存
    getAllScaleFootPosition();

    // 右足左足と直線の距離出力
    legLaneDist();

    // 直線距離が近傍xフレームで最小になるフレームをチェック
    candidateStepFrame();

    // 足が対象レーン内に存在するかどうか確認
    insideLane();

    //設置候補点の左右統合
    mergeStepID();

    //接地点の可視化
    visualizeSteps();

    //ピッチを用いた補正
    pitchCompletion();

    //歩幅を求める
    calculateStrideLength();

    //歩幅の可視化
    visualizeStride();

    //平均歩幅を用いた補正
    //averageCompletion();



    //多項式近似による補正
    nDegreeApproximation Aproximate(4, this->stridePoints);


}


//void Panorama::projectOverview() {
//
//    //色ベクトル
//    vector<cv::Scalar> colors;
//    setColor(&colors);
//
//    // 足投影用画像群
//    cv::Mat panoramaLegImage = this->PanoramaImage.clone();
//    cv::Mat overviewLegImage = this->overviewPanorama.clone();
//
////    //歩幅出力用
//    ofstream rightDisttxt("../images/" + this->_video_name + "/right_leg_dist.txt");
//    ofstream leftDisttxt("../images/" + this->_video_name + "/left_leg_dist.txt");
//    ofstream overrightDisttxt("../images/" + this->_video_name + "/Over_right_leg_dist.txt");
//    ofstream overleftDisttxt("../images/" + this->_video_name + "/Over_left_leg_dist.txt");
//
//    // 右足左足と直線の距離出力
//    vector<float> rightDists = legLaneDist("R");
//    vector<float> leftDists = legLaneDist("L");
//
//    // 直線距離が近傍xフレームで最小になるフレームをリストへ
//    vector<int> RstepID = candidateStepFrame("R");
//    vector<int> LstepID = candidateStepFrame("L");
//
//    int distnum = 0;
//
//
//    //設置候補点のID右
//
//    vector<int> stepID;
//    cv::Mat dummy_OV = overviewLegImage.clone();
//    for (ImageInfo im: imList) {
//
//        //ホモグラフィー型変換
//        imList[frame_num].H.convertTo(imList[frame_num].H, CV_64F);
//        this->overView_H.convertTo(this->overView_H, CV_64F);
//
//        int humanID = 0;
//        rightDisttxt << frame_num << " ";
//        leftDisttxt << frame_num << " ";
//
//        for (OpenPoseBody hb : im.runnerCandidate) {
//
//            //トラッキング対象ランナーのIdは1
//            if (hb.humanID == 1) {
//
//
//                //ランナー関節点
//                vector<cv::Point2f> coord = hb.getBodyCoord();
//                float rightDist = yagi::distPoint2Line(coord[10], im.grads[this->TARGET_RUNNER_ID], im.segments[this->TARGET_RUNNER_ID]);
//                float leftDist = yagi::distPoint2Line(coord[13], im.grads[this->TARGET_RUNNER_ID], im.segments[this->TARGET_RUNNER_ID]);
//
//                rightDisttxt << rightDist;
//                leftDisttxt  << leftDist;
//
//
//
//                vector<cv::Point2f> panoramaCoords;
//
//                //1,元の関節点をパノラマ座標系へ変換
//                mycalcWarpedPoint(coord, &panoramaCoords, imList[frame_num].H);
//
//                //2,リサイズ分の変形
//                for (int pt = 0; pt < panoramaCoords.size(); pt++) {
//                    panoramaCoords[pt].x = panoramaCoords[pt].x * this->Panorama_width;
//                    panoramaCoords[pt].y = panoramaCoords[pt].y * this->Panorama_height;
//                }
//
//                //パノラマ設置点に格納
//                this->PanoramaLeftPoints.push_back(panoramaCoords[13]);
//                this->PanoramaRightPoints.push_back(panoramaCoords[10]);
//
//                vector<cv::Point2f> overviewCoords;
//                //3,接地点をパノラマ座標系からoverview座標系へ変形
//                mycalcWarpedPoint(panoramaCoords, &overviewCoords, this->overView_H);
//
//                //右足、左足だけ投影
//                overviewCoords[10].x = overviewCoords[10].x * 0.50;
//                overviewCoords[10].y = overviewCoords[10].y * 0.50;
//                overviewCoords[13].x = overviewCoords[13].x * 0.50;
//                overviewCoords[13].y = overviewCoords[13].y * 0.50;
//
//                this->overviewRightLegs.push_back(overviewCoords[10]);
//                this->overviewLeftLegs.push_back(overviewCoords[13]);
//
//
//
//                //対象トラックの傾き
//                float outerGrad = im.grads[trackNumber - 1];
//                float innerGrad = im.grads[trackNumber];
//                float outerSeg = im.segments[trackNumber - 1];
//                float innerSeg = im.segments[trackNumber];
//                cv::Point2f rightLeg = coord[10];
//                cv::Point2f leftLeg = coord[13];
//                float outerRightThreshold = outerGrad*rightLeg.x + outerSeg;
//                float innerRightThreshold = innerGrad*rightLeg.x + innerSeg;
//
//
//                cv::Mat dummy = im.image.clone();
//                cv::circle(dummy, rightLeg, 2, colors[0]);
//                cv::Point2f outerL(0, outerRightThreshold);
//                cv::Point2f outerR(800, outerRightThreshold);
//                cv::line(dummy, outerL, outerR, colors[1]);
//                outerL.y + 10;
//                outerR.y + 10;
//                cv::line(dummy, outerL, outerR, colors[3]);
//                outerL.y + 10;
//                cv::Point2f innerL(0, innerRightThreshold);
//                cv::Point2f innerR(800, innerRightThreshold);
//                innerL.y + 10;
//                innerR.y + 10;
//                cv::line(dummy, innerL, innerR, colors[3]);
//                cv::line(dummy, innerL, innerR, colors[1]);
//
//
//                if ((rightLeg.x > 0) && (rightLeg.x < 1000)) {
//                    if ((rightLeg.y + 15 > (outerRightThreshold)) &&
//                        (rightLeg.y + 5 < (innerRightThreshold))) {
//
//                        //直線との距離が最小値とったらマーク
//                        if (distnum > 3) {
//                            if ((rightDist < rightDists[distnum + 1]) &&
//                                (rightDist < rightDists[distnum + 2]) &&
//                                (rightDist < rightDists[distnum + 3]) &&
//                                (rightDist < rightDists[distnum - 1]) &&
//                                (rightDist < rightDists[distnum - 2]) &&
//                                (rightDist < rightDists[distnum - 3]) &&
//                                (overviewCoords[10].x > 0) && (overviewCoords[10].x < 1000)) {
//                                rightDisttxt << " && ";
//                                RstepID.push_back(distnum);
//                                cv::circle(dummy, rightLeg, 3, colors[1], 3);
//                            }
//                        }
//                    }
//                }
//
//                if ((leftLeg.x > 0) && (leftLeg.x < 1000)) {
//                    float outerLeftThreshold = outerGrad * leftLeg.x + outerSeg;
//                    float innerLeftThreshold = innerGrad * leftLeg.x + innerSeg;
//
//                    if ((leftLeg.y + 15> (outerLeftThreshold)) &&
//                        (leftLeg.y + 5< (innerLeftThreshold))) {
//                        if (distnum > 3) {
//                            if ((leftDist < leftDists[distnum + 1]) &&
//                                (leftDist < leftDists[distnum + 2]) &&
//                                (leftDist < leftDists[distnum + 3]) &&
//                                (leftDist < leftDists[distnum - 1]) &&
//                                (leftDist < leftDists[distnum - 2]) &&
//                                (leftDist < leftDists[distnum - 3]) &&
//                                (overviewCoords[13].x > 0) && (overviewCoords[13].x < 1000)) {
//                                leftDisttxt << " && ";
//                                LstepID.push_back(distnum);
//                                cv::circle(dummy, leftLeg, 3, colors[1], 3);
//                            }
//                        }
//                    }
//                }
//
//
////                overrightDisttxt << overviewCoords[10] << endl;
////                overleftDisttxt << overviewCoords[13] << endl;
//
//            }
//            humanID++;
//        }
//        rightDisttxt << endl;
//        leftDisttxt << endl;
//        frame_num++;
//        distnum++;
//    }
//
//    //設置候補点の中から抜粋
//    //右足の連続3フレーム最小値を格納
//    for (int i : RstepID) {
//        stepID.push_back(i);
//    }
//
//    //左足の接地点候補と
//    //1フレームズレなら右足優先
//    //２フレームズレなら平均取る
//    int id = 0;
//    for (int j : LstepID) {
//        bool found = false;
//        for (int i : stepID){
//            if(abs(i - j) <= 1){
//                found = true;
//                break;
//            }else if(abs(i - j) == 2){
//                if (i > j){
//                    stepID[id] = stepID[id] - 1;
//                    found = true;
//                    break;
//                }else{
//                    stepID[id] = stepID[id] + 1;
//                    found = true;
//                    break;
//                }
//            }
//        }
//        if (found == false)
//            stepID.push_back(j);
//    }
//
//    std::sort(stepID.begin(),stepID.end());
//
//    frame_num = 0;
//    ofstream result("../images/" + this->_video_name + "/step_result.txt");
//    cv::Mat track = cv::Mat::zeros(cv::Size(1000, 200), CV_8UC3);
//
//    int steptimes = 0;
//    float aveDist = 0;
//
////    １０mごとのせん
//    for (int i = 1; i < 10; i++){
//        cv::Point2f ptup(i * 100, 0);
//        cv::Point2f ptdown(i * 100, 200);
//        cv::Scalar color(255,255,255);
//        cv::line(track, ptup, ptdown, color, 1);
//    }
//
//    vector <cv::Point2f> overviewStepPoints;
//
//    float pre_x = 0;
//    int pre_frame = 0;
//    for (ImageInfo im: imList) {
//        bool steptrue = false;
//
//        //stepしたフレームかどうか
//        for (int id : stepID){
//            if (id == frame_num)
//                steptrue = true;
//        }
//        if(steptrue) {
//
//            //ホモグラフィー型変換
//            imList[frame_num].H.convertTo(imList[frame_num].H, CV_64F);
//            this->overView_H.convertTo(this->overView_H, CV_64F);
//
//            int humanID = 0;
//            for (OpenPoseBody hb : im.runnerCandidate) {
//
//                if (hb.humanID == 1) {
//
//                    //ランナー関節点
//                    vector <cv::Point2f> coord = hb.getBodyCoord();
//                    vector <cv::Point2f> panoramaCoords;
//                    vector <cv::Point2f> overviewCoords;
//
//                    //1,各フレームから1フレーム目へのH(パノラマ)
//                    mycalcWarpedPoint(coord, &panoramaCoords, imList[frame_num].H);
//
//                    //2,リサイズ分の変形
//                    for (int pt = 0; pt < panoramaCoords.size(); pt++) {
//                        panoramaCoords[pt].x = panoramaCoords[pt].x * this->Panorama_width;
//                        panoramaCoords[pt].y = panoramaCoords[pt].y * this->Panorama_height;
//                    }
//
//                    //3,overviewへの変形
//                    mycalcWarpedPoint(panoramaCoords, &overviewCoords, this->overView_H);
//
//                    //右足、左足だけ投影
//                    overviewCoords[10].x = overviewCoords[10].x * 0.50;
//                    overviewCoords[10].y = overviewCoords[10].y * 0.50;
//                    overviewCoords[13].x = overviewCoords[13].x * 0.50;
//                    overviewCoords[13].y = overviewCoords[13].y * 0.50;
//
//                    cv::Point2f pt(overviewCoords[10].x, 200);
//                    if (overviewCoords[13].y > overviewCoords[10].y){
//                        pt.x = overviewCoords[13].x;
//                    }
//
//                    //接地点
//                    stepPoint step;
//                    step.frame = frame_num;
//                    step.step = pt;
//                    step.dist = pt.x - pre_x;
//                    stepPoints.push_back(step);
//                    step.time = step.frame*0.4;
//                    pre_x = pt.x;
//                    aveDist+=step.dist;
//
//                    cv::Scalar red(0,0,255);
//
//                    cout << frame_num << " " << pt.x << endl;
//                    result << frame_num << " " << pt.x << endl;
//
//                }
//                humanID++;
//            }
//        }
//        frame_num++;
//    }
//
//    aveDist/=stepPoints.size();
//    int stepNum = 0;
//
////    補正前の足あと出力
//    for (stepPoint step : stepPoints){
//        cout << step.step.x;
//        cv::Scalar white(255, 255, 255);
//        cv::Point2f plotPoint(step.step.x, 100);
//        cv::circle(track, plotPoint, 2, white ,2);
//    }
//    cout << "step number: " << stepPoints.size() << endl;
//    cv::imshow("trackaaa", track);
//    cv::waitKey(0);
//    cv::imwrite("result_before.png", track);
////    cv::imwrite("../images/" + _video_name + "/resultImage/" + digitString(trackNumber,4) + "result_before.png", track);
//
//    // ave dist よりも大幅に小さい箇所は削除
//    float deleteThreshold = aveDist * 0.5;
//    pre_x = 0;
//    stepNum = 0;
//
//    for (int i = 0; i < stepPoints.size(); i++){
//        stepPoint step = stepPoints[stepNum];
//        //１フレーム目は無視
//        bool deleted = false;
//
//        //最初の100mはavethreshold変化
//        if (step.step.x < 100) {
//            // aveDist よりも大幅に小さいものは削除
//            float stepLength = abs(step.step.x - pre_x);
//            if (stepNum == 0) {
//                stepLength = step.step.x;
//            }
//            if (stepLength < (aveDist * 0.5)) {
//                stepPoints.erase(stepPoints.begin() + stepNum);
//                deleted = true;
//
//            }
//        }else if (step.step.x > 100){
//            // aveDist よりも大幅に小さいものは削除
//            float stepLength = abs(step.step.x - pre_x);
//            if (stepLength < (aveDist * 0.75)) {
//                stepPoints.erase(stepPoints.begin() + stepNum);
//                deleted = true;
//
//            }
//        }
//
//        if (deleted == false) {
//            pre_x = step.step.x;
//            stepNum++;
//        }
//    }
//
//    // ave dist よりも大幅に大きい箇所は追加
//    float addThreshold = aveDist * 1.7;
//    stepNum = 0;
//    bool lastframe = false;
//    for (int i = 0; i <= stepPoints.size(); i++){
//        float stepLength = 0;
//        stepPoint step = stepPoints[stepNum];
//
//        //１フレーム目は無視
//
//        //最後のフレームなら
//        if (i == stepPoints.size()) {
//
//            //aveDistの更新
//            aveDist = 0;
//            float prex = 0;
//            int stepID = 0;
//            int stepNums = 0;
//            for (stepPoint steps: stepPoints){
//                if (steps.step.x > 100){
//                    stepPoints[stepID].dist = steps.step.x - prex;
//                    aveDist+=steps.dist;
//                    stepNums++;
//                }
//                prex = steps.step.x;
//                stepID++;
//            }
//            aveDist /= stepNums;
//            stepLength = abs(float(1000.0 - stepPoints[stepPoints.size()-1].step.x));
//
//            if (stepLength > (aveDist)) {
//                cv::Point2f addpt(stepPoints[stepPoints.size()-1].step.x + aveDist, step.step.y);
//                stepPoint addStep;
//                addStep.step = addpt;
//                addStep.dist = aveDist;
//                addStep.frame = stepPoints[stepPoints.size() - 1].frame + ((stepPoints[stepPoints.size() - 2].frame) - (stepPoints[stepPoints.size() - 3].frame));
//                stepPoints.insert(stepPoints.begin() + stepNum, addStep);
//                lastframe = true;
//
//            }
//        }else {
//
//            if (step.step.x < 100){
//                // aveDist よりも大幅に小さいものは削除
//                stepLength = abs(step.step.x - pre_x);
//                if (stepNum == 0){
//                    stepLength = step.step.x;
//                }
//                cout << "step length = " << stepLength << endl;
//
//                //最初の100mはthreshold変化
//                if (stepLength > (aveDist)) {
//
//                    cv::Point2f addpt;
//                    if (stepLength > (aveDist * 2)) {
//                        addpt.x = (stepPoints[i - 1].step.x + (aveDist * 0.8));
//                        addpt.y = step.step.y;
//                    }else {
//                        addpt.x = ((stepPoints[i - 1].step.x + stepPoints[i].step.x)/2);
//                        addpt.y = step.step.y;
//                    }
//                    stepPoint addStep;
//                    addStep.step = addpt;
//                    addStep.dist = step.dist / 2;
//
//
//                    addStep.frame = float(stepPoints[i - 1].frame + ((step.frame) - stepPoints[i - 1].frame)/2);
//                    if (stepNum == 0){
//                        addStep.frame = stepPoints[i].frame - (stepPoints[i + 1].frame - stepPoints[i].frame);
//                    }
//                    stepPoints.insert(stepPoints.begin() + stepNum, addStep);
//
//                }
//
//            }else if (step.step.x > 100){
//                // aveDist よりも大幅に小さいものは削除
//                stepLength = abs(step.step.x - pre_x);
//
//                if (stepLength > (addThreshold)) {
//                    cv::Point2f addpt;
//                    if (stepLength > (aveDist * 2)) {
//                        addpt.x = (stepPoints[i - 1].step.x + (aveDist));
//                        addpt.y = step.step.y;
//                    }else {
//                        addpt.x = ((stepPoints[i - 1].step.x + stepPoints[i].step.x)/2);
//                        addpt.y = step.step.y;
//                    }
//                    stepPoint addStep;
//                    addStep.step = addpt;
//                    addStep.dist = step.dist / 2;
//                    addStep.frame = float(stepPoints[i - 1].frame + ((step.frame) - stepPoints[i - 1].frame)/2);
//                    stepPoints.insert(stepPoints.begin() + stepNum, addStep);
//
//                }
//            }
//        }
//
//
//        pre_x = stepPoints[stepNum].step.x;
//        stepNum ++;
//    }
//
//    ofstream outputfile("../images/" + _video_name + "/result/" + digitString(trackNumber,4) + "result.txt");
//
//    int stepid = 1;
//
//    float prelength = 0.0;
//    outputfile << "# frame distance stride time" << endl;
//    for (stepPoint step : stepPoints){
//
//        outputfile << " " << step.frame << " " << step.step.x - prelength << " " <<  step.step.x << " " << step.frame * 0.04 << endl;
//        stepid++;
//        prelength = step.step.x;
//        cout << step.step.x;
//        cv::Scalar white(255, 255, 255);
//        cv::Point2f plotPoint(step.step.x, 100);
//        cv::circle(track, plotPoint, 2, white ,2);
//    }
//    cout << "step number: " << stepPoints.size() << endl;
//    outputfile << "# stepnumber " << stepPoints.size() << endl;
//
//    cv::imshow("track", track);
//    cv::imwrite("result.png", track);
////    cv::imwrite("../images/" + _video_name + "/resultImage/" + digitString(trackNumber,4) + "result.png", track);
//    cv::waitKey();
//
//    cv::destroyAllWindows();
//
//
//}

void Panorama::makeStroboRangeImage(){

    //パノラマ画像生成
    cv::Mat mul_H = cv::Mat::zeros(3, 3, CV_64F);
    mul_H.at<double>(0, 0) = 1;
    mul_H.at<double>(1, 1) = 1;
    mul_H.at<double>(2, 2) = 1;
    imList[FIRST_IM_ID].stroboH = mul_H.clone();

    //FIRST_IM_IDフレーム目をベースとする
    cv::Mat base = imList[FIRST_IM_ID].image;
    cv::Mat result;
    cv::Mat result_pano;
    cv::Mat dense_mask_pano;

    //パノラマ画像に0フレーム目を貼り付け
    for (int x = 0; x < base.rows; x++) {
        for (int y = 0; y < base.cols; y++) {
            this->strobo_image.at<cv::Vec3b>(x, y) = base.at<cv::Vec3b>(x, y);
        }
    }

    int x_min, x_max, y_min, y_max = 0;
    int frame_counter = FIRST_IM_ID;
    imList[frame_counter].stroboH = mul_H.clone();
    for (frame_counter; frame_counter < imList.size(); frame_counter++) {

        ImageInfo im = imList[frame_counter];

        //ホモグラフィー掛け合わせ
        mul_H *= im.H;

        //ホモグラフィーの更新
        cv::Mat mul_clone = mul_H.clone();
        imList[frame_counter].stroboH = mul_clone;

        //端の4点の斜影位置を求める
        vector<cv::Point2f> edge_points;
        vector<cv::Point2f> points;
        cv::Point2f pt1(0, 0);
        cv::Point2f pt2(im.image.cols, 0);
        cv::Point2f pt3(0, im.image.rows);
        cv::Point2f pt4(im.image.cols, im.image.rows);
        points.push_back(pt1);
        points.push_back(pt2);
        points.push_back(pt3);
        points.push_back(pt4);
        mycalcWarpedPoint(points, &edge_points, imList[frame_counter].stroboH);

        //パノラマ画像の更新
        cv::Vec3b BLACK(0, 0, 0);
        x_min = int(edge_points[0].x < edge_points[2].x ? edge_points[0].x : edge_points[2].x);
        x_max = int(edge_points[1].x > edge_points[3].x ? edge_points[1].x : edge_points[3].x);
        y_min = int(edge_points[0].y < edge_points[1].y ? edge_points[0].y : edge_points[1].y);
        y_max = int(edge_points[2].y > edge_points[3].y ? edge_points[2].y : edge_points[3].y);
        x_min = (x_min > 0 ? x_min : 0);
        y_min = (y_min > 0 ? y_min : 0);

        //ワーピング
        cv::warpPerspective(imList[frame_counter].image, result_pano, imList[frame_counter].stroboH, cv::Size(x_max, y_max), CV_HAL_BORDER_CONSTANT);
        cv::warpPerspective(imList[frame_counter].denseMask, imList[frame_counter].denseMask, imList[frame_counter].stroboH, cv::Size(x_max, y_max), CV_HAL_BORDER_CONSTANT);

        cout << frame_counter << " th frame is added to Panorama" << endl;
        for (int x = x_min; x < x_max; x++) {
            for (int y = y_min; y < y_max; y++) {
                if (result_pano.at<cv::Vec3b>(y, x) != BLACK) {
                    this->strobo_image.at<cv::Vec3b>(y, x) = result_pano.at<cv::Vec3b>(y, x);
                }
            }
        }
        imList[frame_counter].strobo_scale_im = result_pano.clone();

        //パノラマの途中経過
        cv::Rect rect(0, 0, x_max, y_max);
        cv::Mat smallStrobo(this->strobo_image, rect);
        this->small_strobo = smallStrobo.clone();
    }

    for (int id = FIRST_IM_ID; id < LAST_IM_ID; id++) {
        if (id % PROJECTION_STEP == 0) {
            ImageInfo im = imList[id];
            for (OpenPoseBody hb: im.runnerCandidate) {
                if (hb.humanID == 1) {
                    //ワーピングのサイズを決定
                    vector<cv::Point2f> panorama_human_area_points;
                    vector<cv::Point2f> human_area_points;
                    cv::Point2f Hb_pt1(hb.mask_rect.tl().x + 30, hb.mask_rect.tl().y + 30);
                    cv::Point2f Hb_pt2(hb.mask_rect.tl().x + hb.mask_rect.width + 30, hb.mask_rect.y + 30);
                    cv::Point2f Hb_pt4(hb.mask_rect.tl().x + 30, hb.mask_rect.y + hb.mask_rect.height + 30);
                    cv::Point2f Hb_pt3(hb.mask_rect.tl().x + hb.mask_rect.width + 30,
                                       hb.mask_rect.y + hb.mask_rect.height + 30);
                    human_area_points.push_back(Hb_pt1);
                    human_area_points.push_back(Hb_pt2);
                    human_area_points.push_back(Hb_pt3);
                    human_area_points.push_back(Hb_pt4);
                    mycalcWarpedPoint(human_area_points, &panorama_human_area_points, im.stroboH);

                    x_min = int(panorama_human_area_points[0].x < panorama_human_area_points[2].x
                                ? panorama_human_area_points[0].x : panorama_human_area_points[2].x);
                    x_max = int(panorama_human_area_points[1].x > panorama_human_area_points[3].x
                                ? panorama_human_area_points[1].x : panorama_human_area_points[3].x);
                    y_min = int(panorama_human_area_points[0].y < panorama_human_area_points[1].y
                                ? panorama_human_area_points[0].y : panorama_human_area_points[1].y);
                    y_max = int(panorama_human_area_points[2].y > panorama_human_area_points[3].y
                                ? panorama_human_area_points[2].y : panorama_human_area_points[3].y);
                    x_min = (x_min > 0 ? x_min : 0);
                    y_min = (y_min > 0 ? y_min : 0);

                    //ワーピング
                    cv::Mat stroboRunner, stroboRunnerMask, mask;
                    cv::warpPerspective(hb.opMaskedImage, stroboRunner, im.stroboH, cv::Size(x_max, y_max));
                    cv::warpPerspective(hb.openPoseMask, stroboRunnerMask, im.stroboH, cv::Size(x_max, y_max));

                    for (int x = x_min; x < x_max; x++) {
                        for (int y = y_min; y < y_max; y++) {
                            if (stroboRunnerMask.at<unsigned char>(y, x) == 255) {
                                this->small_strobo.at<cv::Vec3b>(y, x) = stroboRunner.at<cv::Vec3b>(y, x);
                            }
                        }
                    }
                }
            }
            if(SHOW_STROBO_PROCESS) {
                cv::imshow("strobo_process", this->small_strobo);
                cv::waitKey();
            }
        }
    }
    cv::Rect rect(0, 0, x_max + STROBO_RESIZE_MARGIN, y_max + STROBO_RESIZE_MARGIN);
    cv::Mat croppedStrobo(small_strobo, rect);
    cv::resize(croppedStrobo, croppedStrobo, cv::Size(), 0.5, 0.5);
    cv::imshow("small strobo", croppedStrobo);
    cv::imwrite("../images/" + this->_video_name + "/result/strobo.jpg", croppedStrobo);
    cv::waitKey();
}

//void Panorama::makeStroboImage(){
//    int frame_num = 0;
//    int x_min, y_min, x_max, y_max;
//    this->strobo_image = this->OriginalPanorama.clone();
//    for (ImageInfo im: this->imList){
//        if (frame_num % 25 == 5) {
//            for (OpenPoseBody hb: im.runnerCandidate) {
//                if (hb.humanID == 1) {
//
//                    //人領域のマスクエリアをワーピング
//                    vector<cv::Point2f> panorama_human_area_points;
//                    vector<cv::Point2f> human_area_points;
//                    cv::Point2f Hb_pt1 = hb.mask_rect.tl();
//                    cv::Point2f Hb_pt2(Hb_pt1.x + hb.mask_rect.width, Hb_pt1.y);
//                    cv::Point2f Hb_pt3(Hb_pt1.x, Hb_pt1.y + hb.mask_rect.height);
//                    cv::Point2f Hb_pt4(Hb_pt1.x + hb.mask_rect.width, Hb_pt1.y + hb.mask_rect.height);
//                    human_area_points.push_back(Hb_pt1);
//                    human_area_points.push_back(Hb_pt2);
//                    human_area_points.push_back(Hb_pt3);
//                    human_area_points.push_back(Hb_pt4);
//                    mycalcWarpedPoint(human_area_points, &panorama_human_area_points, im.H);
//
//                    x_min = (panorama_human_area_points[0].x < panorama_human_area_points[2].x
//                             ? panorama_human_area_points[0].x : panorama_human_area_points[2].x);
//                    x_max = (panorama_human_area_points[1].x > panorama_human_area_points[3].x
//                             ? panorama_human_area_points[1].x : panorama_human_area_points[3].x);
//                    y_min = (panorama_human_area_points[0].y < panorama_human_area_points[1].y
//                             ? panorama_human_area_points[0].y : panorama_human_area_points[1].y);
//                    y_max = (panorama_human_area_points[2].y > panorama_human_area_points[3].y
//                             ? panorama_human_area_points[2].y : panorama_human_area_points[3].y);
//                    x_min = (x_min > 0 ? x_min : 0);
//                    y_min = (y_min > 0 ? y_min : 0);
//
////                cv::Mat panorama_human_area_im;
//
//                    for (int x = x_min; x < x_max; x++) {
//                        for (int y = y_min; y < y_max; y++) {
////                        if (panorama_human_area_im.at<cv::Vec3b>(y, x) != BLACK) {
//                            strobo_image.at<cv::Vec3b>(y, x) = im.panorama_scale_im.at<cv::Vec3b>(y, x);
////                        }
//                        }
//                    }
//                }
//            }
//        }
//        frame_num++;
//    }
//
//    //画像のリサイズ
//    cv::resize(strobo_image, strobo_image, cv::Size(),
//               smallPanorama_width / strobo_image.cols,
//               smallPanorama_height / strobo_image.rows);
//
//    cv::imshow("strobo_image", this->strobo_image);
//    cv::imwrite("strobo.jpg", this->strobo_image);
//
//    cv::Mat affine = cv::Mat::zeros(2, 3, CV_64F);
//    affine.at<double>(0,0) = 1;
//    affine.at<double>(1,1) = 1;
//    affine.at<double>(0,2) = 2000;
//    affine.at<double>(1,2) = 2000;
//
//
//    cv::warpPerspective( this->strobo_image, this->strobo_image, imList[230].H.inv(),
//                         cv::Size(10000, 5000));
////        cv::warpAffine(this->strobo_image, this->strobo_image, affine,
////                   cv::Size(10000, 5000));
//    cv::imshow("panorama", this->strobo_image);
//    cv::imwrite("a.jpg", this->strobo_image);
////    cv::waitKey();
//    cv::waitKey();
//
////    for (int i = 1; i < this->imList.size(); i++) {
////        cv::warpPerspective(strobo_image, strobo_image, this->imList[i].H.inv(), this->PanoramaImage.size());
////        cv::imshow("strobo_image", this->strobo_image);
////        cv::waitKey();
////    }
//
//}

float median(vector<float> v) {
    float size = v.size();
    vector<float> _v(v.size());
    copy(v.begin(), v.end(), back_inserter(_v));
    float tmp;
    for (int i = 0; i < size - 1; i++){
        for (int j = i + 1; j < size; j++) {
            if (_v[i] > _v[j]){
                tmp = _v[i];
                _v[i] = _v[j];
                _v[j] = tmp;
            }
        }
    }
    if ((int(size) % 2) == 1) {
        return _v[(size - 1) / 2];
    } else {
        return (_v[(size / 2) - 1] + _v[size / 2]) / 2;
    }
}

void Panorama::calcOpticalFlow(){

    VideoCapture cap("../originalVideos/00009.MTS");

    int Width=cap.get(CV_CAP_PROP_FRAME_WIDTH);
    int Height=cap.get(CV_CAP_PROP_FRAME_HEIGHT);
    int max_frame=cap.get(CV_CAP_PROP_FRAME_COUNT); //フレーム数


    Mat source(Height,Width,CV_8UC1);
    Mat HIS_source(Height,Width,CV_8UC1);
    cv::resize(HIS_source, HIS_source, cv::Size(), 0.5, 0.5);

    for(int frame=0; frame<max_frame; frame++){
        cap>>source;
        cv::resize(source, source, cv::Size(), 0.5, 0.5);
        Mat disp=source.clone();
        cvtColor(source, source, CV_BGR2GRAY);

        if(frame>80){

            vector<cv::Point2f> prev_pts;
            vector<cv::Point2f> next_pts;

            Size flowSize(25,25); //ベクトルの数
            Point2f center = cv::Point(source.cols/2., source.rows/2.);
            for(int i=0; i<flowSize.width; ++i) {
                for(int j=0; j<flowSize.width; ++j) {
                    Point2f p(i*float(source.cols)/(flowSize.width-1),
                              j*float(source.rows)/(flowSize.height-1));
                    prev_pts.push_back((p-center)*0.95f+center);
                }
            }

            Mat flow;
            vector<float> error;

            calcOpticalFlowFarneback(HIS_source, source, flow, 0.8, 10, 15, 3, 5, 1.1, 0);

            // オプティカルフローの表示
            vector<float> xList;
            vector<float> yList;
            std::vector<cv::Point2f>::const_iterator p = prev_pts.begin();
            for(; p!=prev_pts.end(); ++p) {
                const cv::Point2f& fxy = flow.at<cv::Point2f>(p->y, p->x);
                cv::line(disp, *p, *p+fxy, cv::Scalar(0),1);
                cout << *p << " " << fxy << endl;
                xList.push_back(fxy.x);
                yList.push_back(fxy.y);
//                imshow("source", HIS_source);

            }


            std::sort(xList.begin(),xList.end());
            std::sort(yList.begin(),yList.end());

            int xMid = xList[int(xList.size()/2)];
            int yMid = yList[int(yList.size()/2)];

            //TranslationのH求める
            vector<cv::Point2f> prevPts;
            prevPts.push_back(cv::Point2f(0,0));
            prevPts.push_back(cv::Point2f(HIS_source.cols,0));
            prevPts.push_back(cv::Point2f(0,HIS_source.rows));
            prevPts.push_back(cv::Point2f(HIS_source.cols,HIS_source.rows));

            vector<cv::Point2f> nextPts;
            nextPts.push_back(cv::Point2f(xMid,yMid));
            nextPts.push_back(cv::Point2f(HIS_source.cols + xMid,yMid));
            nextPts.push_back(cv::Point2f(xMid,HIS_source.rows + yMid));
            nextPts.push_back(cv::Point2f(HIS_source.cols + xMid,HIS_source.rows + yMid));

            cout << "mid " << xMid << " " << yMid << endl;

            if(xMid < 0)
                xMid = 0;
            if(yMid < 0)
                yMid = 0;
            cv::Mat transH = cv::findHomography(prevPts, nextPts);
            cv::Mat warpedSrc;
            cv::warpPerspective(source, warpedSrc, transH.inv(), cv::Size(HIS_source.cols + xMid, HIS_source.rows + yMid));
//            cv::resize(warpedSrc, transH, warpedSrc.size());


            cout << "mid " << xMid << " " << yMid << endl;
            for (int x = 0; x < HIS_source.cols ; x++) {
                for (int y = 0; y < HIS_source.rows ; y++) {
                    if (warpedSrc.at<unsigned char>(y, x) == 0) {
                        warpedSrc.at<unsigned char>(y, x) = HIS_source.at<unsigned char>(y, x);
                    }
                }
            }
//            cv::imshow("source", source);
//            cv::imshow("HISsource", HIS_source);
            cv::imshow("warped", warpedSrc);
//            cv::imshow("transH", transH);
//            cv::waitKey();

            HIS_source=source.clone();

            imshow("vector", disp);
            imshow("source", HIS_source);

            int c = waitKey(0);
        }

        cout<<frame<<endl;
        frame+=1;
    }
    waitKey(0);
}


void Panorama::getTranslationByOpticalFlow(){
//    for(int frameID=0; frameID<imList.size(); frameID++){
//        Mat source(Height,Width,CV_8UC1);
//        Mat HIS_source(Height,Width,CV_8UC1);
//        cv::resize(HIS_source, HIS_source, cv::Size(), 0.5, 0.5);
//
//        cv::resize(source, source, cv::Size(), 0.5, 0.5);
//        Mat disp=source.clone();
//        cvtColor(source, source, CV_BGR2GRAY);
//
//        if(frameID>80){
//
//            vector<cv::Point2f> prev_pts;
//            vector<cv::Point2f> next_pts;
//
//            Size flowSize(25,25); //ベクトルの数
//            Point2f center = cv::Point(source.cols/2., source.rows/2.);
//            for(int i=0; i<flowSize.width; ++i) {
//                for(int j=0; j<flowSize.width; ++j) {
//                    Point2f p(i*float(source.cols)/(flowSize.width-1),
//                              j*float(source.rows)/(flowSize.height-1));
//                    prev_pts.push_back((p-center)*0.95f+center);
//                }
//            }
//
//            Mat flow;
//            vector<float> error;
//
//            calcOpticalFlowFarneback(HIS_source, source, flow, 0.8, 10, 15, 3, 5, 1.1, 0);
//
//            // オプティカルフローの表示
//            vector<float> xList;
//            vector<float> yList;
//            std::vector<cv::Point2f>::const_iterator p = prev_pts.begin();
//            for(; p!=prev_pts.end(); ++p) {
//                const cv::Point2f& fxy = flow.at<cv::Point2f>(p->y, p->x);
//                cv::line(disp, *p, *p+fxy, cv::Scalar(0),1);
//                cout << *p << " " << fxy << endl;
//                xList.push_back(fxy.x);
//                yList.push_back(fxy.y);
////                imshow("source", HIS_source);
//
//            }
//
//
//            std::sort(xList.begin(),xList.end());
//            std::sort(yList.begin(),yList.end());
//
//            int xMid = xList[int(xList.size()/2)];
//            int yMid = yList[int(yList.size()/2)];
//
//            //TranslationのH求める
//            vector<cv::Point2f> prevPts;
//            prevPts.push_back(cv::Point2f(0,0));
//            prevPts.push_back(cv::Point2f(HIS_source.cols,0));
//            prevPts.push_back(cv::Point2f(0,HIS_source.rows));
//            prevPts.push_back(cv::Point2f(HIS_source.cols,HIS_source.rows));
//
//            vector<cv::Point2f> nextPts;
//            nextPts.push_back(cv::Point2f(xMid,yMid));
//            nextPts.push_back(cv::Point2f(HIS_source.cols + xMid,yMid));
//            nextPts.push_back(cv::Point2f(xMid,HIS_source.rows + yMid));
//            nextPts.push_back(cv::Point2f(HIS_source.cols + xMid,HIS_source.rows + yMid));
//
//            cout << "mid " << xMid << " " << yMid << endl;
//
//            if(xMid < 0)
//                xMid = 0;
//            if(yMid < 0)
//                yMid = 0;
//            cv::Mat transH = cv::findHomography(prevPts, nextPts);
//            cv::Mat warpedSrc;
//            cv::warpPerspective(source, warpedSrc, transH.inv(), cv::Size(HIS_source.cols + xMid, HIS_source.rows + yMid));
////            cv::resize(warpedSrc, transH, warpedSrc.size());
//
//
//            cout << "mid " << xMid << " " << yMid << endl;
//            for (int x = 0; x < HIS_source.cols ; x++) {
//                for (int y = 0; y < HIS_source.rows ; y++) {
//                    if (warpedSrc.at<unsigned char>(y, x) == 0) {
//                        warpedSrc.at<unsigned char>(y, x) = HIS_source.at<unsigned char>(y, x);
//                    }
//                }
//            }
////            cv::imshow("source", source);
////            cv::imshow("HISsource", HIS_source);
//            cv::imshow("warped", warpedSrc);
////            cv::imshow("transH", transH);
////            cv::waitKey();
//
//            HIS_source=source.clone();
//
//            imshow("vector", disp);
//            imshow("source", HIS_source);
//
//            int c = waitKey(0);
//        }
//
//        cout<<frameID<<endl;
//        frameID+=1;
//    }
//    waitKey(0);
}

void Panorama::translateImage() {
    cv::Mat im1, im2;
    int WIDTH, HEIGHT;
    for(int frameID=0; frameID<imList.size(); frameID++) {
//        cout << "Translation = " << imList[frameID].translation << endl;
        im1 = imList[frameID].image;
        if (frameID == 0) {
            HEIGHT = im1.rows;
            WIDTH = im1.cols;
        } else {
            for(int i = 0; i < imList[frameID].translationList.size(); i ++) {
//                cout << i << endl;
                int x = int(imList[frameID].translationList[i].x);
                int y = int(imList[frameID].translationList[i].y);

                //TranslationのH求める
                vector<cv::Point2f> prevPts;
                prevPts.push_back(cv::Point2f(0, 0));
                prevPts.push_back(cv::Point2f(WIDTH, 0));
                prevPts.push_back(cv::Point2f(0, HEIGHT));
                prevPts.push_back(cv::Point2f(WIDTH, HEIGHT));

                vector<cv::Point2f> nextPts;
                nextPts.push_back(cv::Point2f(x, y));
                nextPts.push_back(cv::Point2f(WIDTH + x, y));
                nextPts.push_back(cv::Point2f(x, HEIGHT + y));
                nextPts.push_back(cv::Point2f(WIDTH + x, HEIGHT + y));

                if (x < 0)
                    x = 0;
                if (y < 0)
                    y = 0;

                cv::Mat transH = cv::findHomography(prevPts, nextPts);
                cv::Mat warpedSrc;
                cv::warpPerspective(im1, warpedSrc, transH,
                                    cv::Size(WIDTH + x, HEIGHT + y));

                for (int x = 0; x < HEIGHT; x++) {
                    for (int y = 0; y < WIDTH; y++) {
                        if (warpedSrc.at<cv::Vec3b>(x, y) == cv::Vec3b(0, 0, 0)) {
                            warpedSrc.at<cv::Vec3b>(x, y) = im2.at<cv::Vec3b>(x, y);
                        }
                    }
                }
                cv::imshow("Translation Result", warpedSrc);
                cv::waitKey();
            }
        }

        im2 = im1;
    }
}

void Panorama::getTranslation(){
//    this->getTranslationByOpticalFlow();
//    this->featurePointFindHomography();
    this->getTranslationByTempMatching();

    if(SHOW_TRANSLATION)
        this->translateImage();
}

cv::Mat getResizeHomography(cv::Size panorama, cv::Size smallPanorama){
    vector<cv::Point2d> src1_panorama_corners;
    src1_panorama_corners.push_back(cv::Point2d(0,0));
    src1_panorama_corners.push_back(cv::Point2d(panorama.width,0));
    src1_panorama_corners.push_back(cv::Point2d(0,panorama.height));
    src1_panorama_corners.push_back(cv::Point2d(panorama.width,panorama.height));

    vector<cv::Point2d> src1_small_panorama_corners;
    src1_small_panorama_corners.push_back(cv::Point2d(0,0));
    src1_small_panorama_corners.push_back(cv::Point2d(smallPanorama.width,0));
    src1_small_panorama_corners.push_back(cv::Point2d(0,smallPanorama.height));
    src1_small_panorama_corners.push_back(cv::Point2d(smallPanorama.width,smallPanorama.height));

    cv::Mat resize_H = cv::findHomography(src1_panorama_corners, src1_small_panorama_corners, 1);
    return resize_H;
}



void Panorama::makeVirtualRaceImages(){
    virtualRace src;
    virtualRace tar;
    src.readSavedData(_txt_folder + "/virtualRaceData.txt");
    tar.readSavedData("../projects/" + VIRTUAL_TARGET_VIDEO + "/texts/virtualRaceData.txt");
    tar.loadImages(VIRTUAL_TARGET_VIDEO);

    //パノラマresize分のH求める
    cv::Mat resizeH1 = getResizeHomography(cv::Size(src.PanoramaWidth, src.PanoramaHeight),
                                           cv::Size(src.smallPanoramaWidth, src.smallPanoramaHeight));
    cv::Mat resizeH2 = getResizeHomography(cv::Size(tar.PanoramaWidth, tar.PanoramaHeight),
                                           cv::Size(tar.smallPanoramaWidth, tar.smallPanoramaHeight));

    //パノラマ画像間のホモグラフィーを求める
    cv::Mat videoTovideoH;
    vector<cv::Point2f> src1_small_corners;
    vector<cv::Point2f> src2_small_corners;
    yagi::mycalcWarpedPoint(src.cornerPoints, &src1_small_corners, resizeH1);
    yagi::mycalcWarpedPoint(tar.cornerPoints, &src2_small_corners, resizeH2);
    videoTovideoH = cv::findHomography(src1_small_corners, src2_small_corners);

    //人領域をワーピング
    int frame_num = 0;
    for (ImageInfo im: this->imList) {
        for (OpenPoseBody hb: im.runnerCandidate) {
            if (hb.humanID == 1) {
                //ホモグラフィー掛けあわせていく
                cv::Mat H_virtual = tar.HomographyList[frame_num].inv()*resizeH2.inv()*videoTovideoH*resizeH1*im.mulH;
                cv::Mat warpedRunner;
                cv::Mat warpedRunnerMask;
                cv::warpPerspective(hb.opMaskedImage, warpedRunner, H_virtual, im.image.size());
                cv::warpPerspective(hb.openPoseMask, warpedRunnerMask, H_virtual, im.image.size());

                //人領域のマスクエリアをワーピング
                vector<cv::Point2d> tar_area;
                vector<cv::Point2d> src1_human_area;
                cv::Point2d Hb_pt1 = hb.mask_rect.tl();
                cv::Point2d Hb_pt2(Hb_pt1.x + hb.mask_rect.width, Hb_pt1.y);
                cv::Point2d Hb_pt3(Hb_pt1.x, Hb_pt1.y + hb.mask_rect.height);
                cv::Point2d Hb_pt4(Hb_pt1.x + hb.mask_rect.width, Hb_pt1.y + hb.mask_rect.height);
                src1_human_area.push_back(Hb_pt1);
                src1_human_area.push_back(Hb_pt2);
                src1_human_area.push_back(Hb_pt3);
                src1_human_area.push_back(Hb_pt4);

                //src1のフレームからパノラマへ
//                vector<cv::Point2d> src1_panorama_points;
//                mycalcDoubleWarpedPoint(src1_human_area, &src1_panorama_points, im.mulH);
//                vector<cv::Point2d> src1_small_panorama_points;
//                mycalcDoubleWarpedPoint(src1_panorama_points, &src1_small_panorama_points, resizeH1);
//                vector<cv::Point2d> src2_small_panorama_points;
//                mycalcDoubleWarpedPoint(src1_small_panorama_points, &src2_small_panorama_points, videoTovideoH);
//                vector<cv::Point2d> src2_panorama_points;
//                mycalcDoubleWarpedPoint(src2_small_panorama_points, &src2_panorama_points, resizeH2.inv());
//                vector<cv::Point2d> src2_human_area;
//                mycalcDoubleWarpedPoint(src2_panorama_points, &src2_human_area, tar.HomographyList[frame_num].inv());


//                for (auto x = int(src2_human_area[0].x); x < int(src2_human_area[1].x); x++) {
//                    for (auto y = int(src2_human_area[0].y); y < int(src2_human_area[2].y); y++) {
//                        if(warpedRunnerMask.at<unsigned char>(y,x) != 255) {
//                            tar.srcImages[frame_num].at<cv::Vec3b>(y, x)[0] = warpedRunner.at<cv::Vec3b>(y, x)[0];
//                            tar.srcImages[frame_num].at<cv::Vec3b>(y, x)[1] = warpedRunner.at<cv::Vec3b>(y, x)[1];
//                            tar.srcImages[frame_num].at<cv::Vec3b>(y, x)[2] = warpedRunner.at<cv::Vec3b>(y, x)[2];
//                        }
//                    }
//                }

                cv::imshow("warpedRunnerMask", warpedRunnerMask);
                for (auto x = 0; x < IMG_HEIGHT; x++) {
                    for (auto y = 0; y < IMG_WIDTH; y++) {
//                        cv::circle(warpedRunner, cv::Point2f(y,x), 2, cv::Scalar(255,0,0), 2);
                        if(warpedRunnerMask.at<unsigned char>(x,y) != 0) {
                            tar.srcImages[frame_num].at<cv::Vec3b>(x,y)[0] = warpedRunner.at<cv::Vec3b>(x,y)[0];
                            tar.srcImages[frame_num].at<cv::Vec3b>(x,y)[1] = warpedRunner.at<cv::Vec3b>(x,y)[1];
                            tar.srcImages[frame_num].at<cv::Vec3b>(x,y)[2] = warpedRunner.at<cv::Vec3b>(x,y)[2];
                        }
                    }
                }
                cv::imshow("warpedRunner", warpedRunner);
                cv::imshow("virtual race", tar.srcImages[frame_num]);
                cv::waitKey();
            }
        }
    frame_num++;
    }
}


//トラックラインで位置合わせ
//                vector<cv::Point2f> src1EdgePoints = getTrackEdgePoints();
//                vector<cv::Point2f> src2EdgePoints = getTrackEdgePoints();
//                cv::imshow("PanoramaImage", PanoramaImage);
//    //デバッグ
//    cv::Mat tar_panorama = cv::imread("../images/" + VIRTUAL_TARGET_VIDEO + "/result/panorama.jpg");
//    vector<cv::Point2f> warpedCorners;
//    yagi::mycalcWarpedPoint(src1_small_corners, &warpedCorners, videoTovideoH);
//
//    for(cv::Point2d pt: warpedCorners){
//        cv::circle(tar_panorama, pt, 2, cv::Scalar(0,255,0), 2);
//    }
//    cv::imshow("tar_panorama", tar_panorama);
//    cv::waitKey();

void Panorama::saveData(){
    int frame_num = 0;
    ofstream outputfile(_txt_folder + "/virtualRaceData.txt");
    outputfile << "Corner_points_panorama: ";
    for(cv::Point2f pt: this->startLineCornerPoints){
        outputfile << pt.x << " " << pt.y << " ";
        cout << "start point: " << pt << endl;
    }

    vector<cv::Point2f> dummy;
    yagi::mycalcWarpedPoint(this->finishLineCornerPoints, &dummy, this->imList[this->finalLineImageNum].mulH);
    for(cv::Point2f pt: dummy){
        outputfile << pt.x << " " << pt.y << " ";
        cout << "finish point: " << pt << endl;
    }
    outputfile << endl;

    outputfile << "SmallSizePanorama: ";
    outputfile << this->smallPanorama_width << " " << this->smallPanorama_height << " ";
    outputfile << this->Panorama_width << " " << this->Panorama_height << endl;
    for(ImageInfo im: this->imList){
        outputfile << "Frame: " << frame_num << " ";
        outputfile << im.mulH.at<double>(0,0) << " "
                   << im.mulH.at<double>(0,1) << " " << im.mulH.at<double>(0,2) << " "
                   << im.mulH.at<double>(1,0) << " " << im.mulH.at<double>(1,1) << " "
                   << im.mulH.at<double>(1,2) << " " << im.mulH.at<double>(2,0) << " "
                   << im.mulH.at<double>(2,1) << " " << im.mulH.at<double>(2,2) << endl;
        frame_num++;
    }
    outputfile << "Last_frame_num " << this->imList.size() - 1 << endl;
    outputfile.close();

}

