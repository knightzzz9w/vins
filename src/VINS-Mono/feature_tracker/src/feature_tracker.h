#pragma once

#include <cstdio>
#include <iostream>
#include <queue>
#include <execinfo.h>
#include <csignal>

#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>

#include "camodocal/camera_models/CameraFactory.h"
#include "camodocal/camera_models/CataCamera.h"
#include "camodocal/camera_models/PinholeCamera.h"

#include "parameters.h"
#include "tic_toc.h"

using namespace std;
using namespace camodocal;
using namespace Eigen;

bool inBorder(const cv::Point2f &pt);

void reduceVector(vector<cv::Point2f> &v, vector<uchar> status);
void reduceVector(vector<int> &v, vector<uchar> status);

class FeatureTracker
{
  public:
    FeatureTracker();

    void readImage(const cv::Mat &_img,double _cur_time);

    void setMask();

    void setMask_f();

    void addPoints();

    void addPoints_f();

    bool updateID(unsigned int i);

    bool updateID_f(unsigned int i);

    void readIntrinsicParameter(const string &calib_file);

    void showUndistortion(const string &name);

    void rejectWithF();

    void rejectWithF_f();


    void undistortedPoints();

    void undistortedPoints_f();

    cv::Mat mask;
    cv::Mat mask_f; //多数特征点的mask
    cv::Mat fisheye_mask;
    cv::Mat prev_img, cur_img, forw_img;
    vector<cv::Point2f> n_pts;
    vector<cv::Point2f> n_pts_f;

    vector<cv::Point2f> prev_pts, cur_pts, forw_pts;
    vector<cv::Point2f> prev_un_pts, cur_un_pts;
    vector<cv::Point2f> pts_velocity;

    vector<cv::Point2f> prev_pts_f, cur_pts_f, forw_pts_f;  //f代表的是f 表示的是不参与优化的特征但是参与三角化的特征
    vector<cv::Point2f> prev_un_pts_f, cur_un_pts_f;  
    //vector<cv::Point2f> pts_velocity_f;

    vector<int> ids;
    vector<int> track_cnt;

    vector<int> ids_f;
    vector<int> track_cnt_f;  

    map<int, cv::Point2f> cur_un_pts_map;
    map<int, cv::Point2f> prev_un_pts_map;

    // map<int, cv::Point2f> cur_un_pts_map_f;   不需要速度信息了
    // map<int, cv::Point2f> prev_un_pts_map_f;
    camodocal::CameraPtr m_camera;
    double cur_time;
    double prev_time;

    static int n_id;
    static int n_id_f;
};
