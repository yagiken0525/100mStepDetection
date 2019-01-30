#pragma once

#include <./opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <math.h>
#include <string.h>
#include <string>
#include <sys/stat.h>
#include "videoToImage/videoToImage.h"
#include "videoToImage/trimVideo.h"



namespace yagi {

    class Panorama {

    private:
        // フォルダパス
        std::string _image_folder = "../images/";
        std::string _originalVideo_folder = "../originalVideos/";
        std::string _trimmedVideo_folder = "../videos/";
        std::string _plotData_folder = "../plotData/";

        // ファイルパス
        std::string _imagelist_name = "/imagelist.txt";
        std::string _poselist_name = "/human_pose_info.txt";

        // 変数名
        std::string _video_name;
        std::string _image_txt_path;
        std::string _human_area_list;


    public:

        int MASK_MARGIN;
        int MAX_TRANSLATION;
        bool SHOW_TRANSLATION;
        bool SHOW_HOMOGRAPHY;
        bool SHOW_TRACKLINES;
        bool SHOW_STROBO_PROCESS;
        bool SHOW_PANORAMA;
        bool SHOW_RUNNER_CANDIDATES;
        bool SHOW_MASK_REGIONS;
        int OP_MASK_RADIUS;
        int FIRST_IM_ID;
        int LAST_IM_ID;
        int PROJECTION_STEP;
        bool ESTIMATE_STEPS = true;
        bool GENERATE_STROBO = true;
        bool GENERATE_VIRTUALRACE = true;
        int RANSAC_LOOP_LIMIT;
        int RANSAC_INLIER_RANGE;
        int STROBO_RESIZE_MARGIN;
        bool REMOVE_OTHER_RUNNERS;
        std::string VIRTUAL_TARGET_VIDEO;

        Panorama() {};

        ~Panorama() {};

        //変数格納
        void setVariables(std::string video_name);


        void trackMask(cv::Mat& im);
        void videotoImage();
        cv::Point2f templateMatching(cv::Mat& im1, cv::Mat& im2, cv::Rect tempRect, cv::Mat& maskImage, cv::Point2f preTranslation,  const int frameID);
        void getTranslationByTempMatching();
        void getTranslationByBatchTempMatching();
        void getTranslationByOpticalFlow();
        void featurePointFindHomography();
        void maskingRunners();
        void transformPanoramaTopview();

        void loadingData();
        void loadImage();
        void masking();


        cv::Mat  calcOpenPoseMask(std::vector<cv::Point2f>& pts, cv::Size imSize);

        void selectTrack();


        void trackTracking();


        void selectRunnerCandidates();


        void makePanorama();

        void selectMaskArea();


        void trackDetection();

        void trackingRunner();

        void detectHumanArea();


        void generatePanorama();

        void estimateStepPoints();
        void maskHumanArea();
        void getOpenPoseMask();


        void setMaskArea(cv::Rect _mask_area, cv::Point2i _center);


        void getOverviewHomography();


        void projectOverview();


        void projectTrackLine();


        void trackTargetRunner();


        void startFinishLineSelect();



        void makeStroboImage();


        void generateNthFramePanorama();


        void makeVirtualRaceImages();


        void templateInverseMatchingFindHomography();


        void generateInversePanorama();

        void saveData();

        void getInverseOverviewHomography();

        void projectInverseTrackLine();

        void legLaneDist();

        void candidateStepFrame();

        void insideLane();

        void mergeStepID();

        void visualizeSteps();

        void getAllScaleFootPosition();

        void averageCompletion();

        void calculateStrideLength();

        void calcOpticalFlow();

        void visualizeStride();

        void getHomographyFromTranslation();

        void getTranslation();

        void translateImage();

        void makeStroboRangeImage();

        void pitchCompletion();

        cv::Rect getMaskRect(std::vector<cv::Point2f>& body);

        //
        bool INIT_PROCESSING;
        bool SHOW_LOADED_IMAGE;
        bool SHOW_ONLINE_POINTS;
        bool USE_LASTMASK = true;
        bool USE_LAST_TRACKLINE = true;
        bool USE_LAST_CORNERS = true;
        bool SELECT_TARGET_RANE;
        int IMG_WIDTH;
        int IMG_HEIGHT;
        //
        std::vector<std::string> img_names;

        //OpenPoseBody
        class OpenPoseBody {
        public:
            //Methodｓ
            OpenPoseBody() {};
            ~OpenPoseBody() {};
            void setBodyCoord(std::vector<std::string> coord);
            std::vector<cv::Point2f> getBodyCoord();
            void clearBodyCoord();

            //Variables
            cv::Rect mask_rect;
            int humanID = 100;
            std::vector<cv::Point2f> _body_parts_coord;
            std::vector<float> _confidenceMap;
            cv::Mat openPoseMask;
            cv::Mat opMaskedImage;
        };

        //Step
        class Step{
        public:
            std::string leg;
            cv::Point2f rightPt;
            cv::Point2f leftPt;
            float frame;
            float pitch;
            float stride;
        };

        std::vector<Step> steps;

        //マスクエリア
        struct MaskArea {
            cv::Rect _mask_area;
        };

        std::vector<cv::Point2d> stepPoints;


        //Panorama
        cv::Mat OverView = cv::Mat::zeros(540, 2000, CV_8UC3);
        cv::Mat PanoramaImage = cv::Mat::zeros(20000, 40000, CV_8UC3);
        cv::Mat smallPanoramaImage;
        cv::Mat OriginalPanorama;
        cv::Mat overviewPanorama;
        cv::Mat inv_overView_H;

        int finalLineImageNum;

        std::vector<cv::Point2f> PanoramaLeftPoints;
        std::vector<cv::Point2f> PanoramaRightPoints;

        std::vector<cv::Point2f> panoramaInline10mPoints;
        std::vector<cv::Point2f> panoramaOutline10mPoints;


        std::vector<cv::Point2f> startLineCornerPoints;
        std::vector<cv::Point2f> finishLineCornerPoints;

        std::vector<cv::Point2f> overviewRightLegs;
        std::vector<cv::Point2f> overviewLeftLegs;

        //多項式近似ように歩幅をcv::Point2dベクトルに収納
        std::vector<float> strideLength;
        std::vector<cv::Point2d> stridePoints;

        float averagePitch;


        cv::Mat strobo_image = cv::Mat::zeros(8000, 30000, CV_8UC3);
        cv::Mat small_strobo = cv::Mat::zeros(8000, 30000, CV_8UC3);

        int panorama_offset = 50;

        //?��p?��m?��?��?��}?��p
        class ImageInfo {
        public:

            //images
            cv::Mat image;
            cv::Mat gray_image;
            cv::Mat hsv_image;
            cv::Mat edge;
            cv::Mat panorama_scale_im;
            cv::Mat trackLineImage;

            //mask
            cv::Mat maskimage;
            cv::Mat maskedrunner;
            cv::Mat denseMask;
            std::vector<MaskArea> maskAreas;

            //Homography
            cv::Point2f translation;
            std::vector<cv::Point2f> translationList;
            cv::Mat H;
            cv::Mat mulH;
            cv::Mat inverseH;
            cv::Mat mulInvH;
            std::vector<cv::Point2f> onlinePointList;

            //Runners
            cv::Mat runnerCandidatesImage;
            std::vector<OpenPoseBody> runnerCandidate;
            std::vector<std::pair<cv::Point2f, cv::Point2f>> lines10m;
            std::vector<OpenPoseBody> Runners;

            //接地判定
            cv::Point2f originalRfoot;
            cv::Point2f originalLfoot;
            cv::Point2f panoramaRfoot;
            cv::Point2f panoramaLfoot;
            cv::Point2f overviewRfoot;
            cv::Point2f overviewLfoot;
            float RlineDist;
            float LlineDist;
            bool Rstep = false;
            bool Lstep = false;
            bool stepPoint = false;

            //Strobo
            cv::Mat stroboH;
            cv::Mat strobo_scale_im;


            //?��}?��b?��`?��?��?��O?��p
            std::vector<cv::Point2f> prev_keypoints;
            std::vector<cv::Point2f> this_keypoints;


            //?��g?��?��?��b?��N?��?��?��̃g?��?��?��b?��L?��?��?��O
            std::vector<std::pair<cv::Point2f, cv::Point2f>> track_lines;
            std::vector<cv::Mat> track_line_masks;
            cv::Mat trackAreaMask;
            std::vector<float> grads;
            std::vector<float> segments;

        };

        cv::Mat overView_H;

        std::vector<ImageInfo> imList;

        int TARGET_RUNNER_ID;

        void showOnlinePoints(ImageInfo &im);

        void obtainOnlinePointsAsIm1(ImageInfo& im);
        void obtainOnlinePointsAsIm2(ImageInfo& im);

    private:
        //Panorama
        float Panorama_width;
        float Panorama_height;

        //smallPanorama
        float smallPanorama_width = 1000.0;
        float smallPanorama_height = 500.0;

        std::vector<MaskArea> mask_areas;
        std::vector<std::vector<OpenPoseBody> > allRunners;

    };
}

class step{
public:
    cv::Point2f pt;
    std::string s;
    int frame;
};