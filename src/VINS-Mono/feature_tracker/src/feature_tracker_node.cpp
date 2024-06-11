#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/Imu.h>
#include <std_msgs/Bool.h>
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>

#include "feature_tracker.h"

#define SHOW_UNDISTORTION 0

vector<uchar> r_status;
vector<float> r_err;
queue<sensor_msgs::ImageConstPtr> img_buf;

ros::Publisher pub_img, pub_img_feature  , pub_match, pub_match_feature;
ros::Publisher pub_restart;

FeatureTracker trackerData[NUM_OF_CAM];
double first_image_time;
int pub_count = 1;
bool first_image_flag = true;
double last_image_time = 0;
bool init_pub = 0;

void img_callback(const sensor_msgs::ImageConstPtr &img_msg)   //想要得到img feature与img incre feature对齐的
{
    if(first_image_flag)
    {
        first_image_flag = false;
        first_image_time = img_msg->header.stamp.toSec();
        last_image_time = img_msg->header.stamp.toSec();
        return;
    }
    // detect unstable camera stream
    if (img_msg->header.stamp.toSec() - last_image_time > 1.0 || img_msg->header.stamp.toSec() < last_image_time)
    {
        ROS_WARN("image discontinue! reset the feature tracker!");
        first_image_flag = true; 
        last_image_time = 0;
        pub_count = 1;
        std_msgs::Bool restart_flag;
        restart_flag.data = true;
        pub_restart.publish(restart_flag);
        return;
    }
    last_image_time = img_msg->header.stamp.toSec();
    // frequency control
    //std::cout << "Image pub freq is " << FREQ << std::endl;
    if (round(1.0 * pub_count / (img_msg->header.stamp.toSec() - first_image_time)) <= FREQ)  //发布的频率是10 所以才有后面的频率
    {
        PUB_THIS_FRAME = true;
        // reset the frequency control
        if (abs(1.0 * pub_count / (img_msg->header.stamp.toSec() - first_image_time) - FREQ) < 0.01 * FREQ)
        {
            first_image_time = img_msg->header.stamp.toSec();
            pub_count = 0;
        }
    }
    else
        PUB_THIS_FRAME = false;

    cv_bridge::CvImageConstPtr ptr;
    cv_bridge::CvImageConstPtr ptr_feature;

    //std::cout << "Image encoding is " << img_msg->encoding  <<  std::endl;
    if (img_msg->encoding == "8UC1")
    {
        sensor_msgs::Image img;
        img.header = img_msg->header;
        img.height = img_msg->height;
        img.width = img_msg->width;
        img.is_bigendian = img_msg->is_bigendian;
        img.step = img_msg->step;
        img.data = img_msg->data;
        img.encoding = "mono8";
        ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO8);
        ptr_feature = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO8);
    }
    else
    {
        ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::MONO8);
        ptr_feature = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::MONO8);
    }


    cv::Mat show_img = ptr->image;
    TicToc t_r;
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        ROS_DEBUG("processing camera %d", i);
        if (i != 1 || !STEREO_TRACK)
            trackerData[i].readImage(ptr->image.rowRange(ROW * i, ROW * (i + 1)), img_msg->header.stamp.toSec());
        else
        {
            if (EQUALIZE)
            {
                cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
                clahe->apply(ptr->image.rowRange(ROW * i, ROW * (i + 1)), trackerData[i].cur_img);
            }
            else
                trackerData[i].cur_img = ptr->image.rowRange(ROW * i, ROW * (i + 1));
        }

#if SHOW_UNDISTORTION
        trackerData[i].showUndistortion("undistrotion_" + std::to_string(i));
#endif
    }

    for (unsigned int i = 0;; i++)
    {
        bool completed = false;
        for (int j = 0; j < NUM_OF_CAM; j++)
            if (j != 1 || !STEREO_TRACK)
                completed |= trackerData[j].updateID(i);
        if (!completed)
            break;
    }

    for (unsigned int i = 0;; i++)
    {
        bool completed = false;
        for (int j = 0; j < NUM_OF_CAM; j++)
            if (j != 1 || !STEREO_TRACK)
                completed |= trackerData[j].updateID_f(i);
        if (!completed)
            break;
    }
    //std::cout << "cur feature size is " << FeatureTracker::n_id << std::endl;
    //std::cout << "cur feature f size is " << FeatureTracker::n_id_f << std::endl;
    ROS_DEBUG("cur feature size is %d", FeatureTracker::n_id);
    ROS_DEBUG("cur feature f size is %d", FeatureTracker::n_id_f);
   if (PUB_THIS_FRAME)
   {
        pub_count++;   //这一帧发布的信息
        sensor_msgs::PointCloudPtr feature_points(new sensor_msgs::PointCloud);
        sensor_msgs::ChannelFloat32 id_of_point;
        sensor_msgs::ChannelFloat32 u_of_point;
        sensor_msgs::ChannelFloat32 v_of_point;
        sensor_msgs::ChannelFloat32 velocity_x_of_point;
        sensor_msgs::ChannelFloat32 velocity_y_of_point;

        feature_points->header = img_msg->header;
        feature_points->header.frame_id = "world";

        vector<set<int>> hash_ids(NUM_OF_CAM);
        for (int i = 0; i < NUM_OF_CAM; i++)
        {
            auto &un_pts = trackerData[i].cur_un_pts;  //这边是经过逆畸变处理 得到的没有畸变的归一化的点
            auto &cur_pts = trackerData[i].cur_pts; //这边是像素坐标系的点
            auto &ids = trackerData[i].ids;
            auto &pts_velocity = trackerData[i].pts_velocity;
            for (unsigned int j = 0; j < ids.size(); j++)
            {
                if (trackerData[i].track_cnt[j] > 1)
                {
                    int p_id = ids[j];
                    hash_ids[i].insert(p_id);
                    geometry_msgs::Point32 p;
                    p.x = un_pts[j].x;   //这边是经过逆畸变处理 得到的没有畸变的归一化的点
                    p.y = un_pts[j].y;
                    p.z = 1;

                    feature_points->points.push_back(p);
                    //std::cout << "Point " << p << std::endl;
                    id_of_point.values.push_back(p_id * NUM_OF_CAM + i);
                    u_of_point.values.push_back(cur_pts[j].x);
                    v_of_point.values.push_back(cur_pts[j].y);


                    //std::cout << "Point u v is  " << cur_pts[j].x <<  cur_pts[j].y  <<  std::endl;

                    velocity_x_of_point.values.push_back(pts_velocity[j].x);
                    velocity_y_of_point.values.push_back(pts_velocity[j].y);
                }
            }
        }
        feature_points->channels.push_back(id_of_point);
        feature_points->channels.push_back(u_of_point);
        feature_points->channels.push_back(v_of_point);
        feature_points->channels.push_back(velocity_x_of_point);
        feature_points->channels.push_back(velocity_y_of_point);
        ROS_DEBUG("publish %f, at %f", feature_points->header.stamp.toSec(), ros::Time::now().toSec());
        // skip the first image; since no optical speed on frist image

            




        sensor_msgs::PointCloudPtr feature_incre_points(new sensor_msgs::PointCloud);  //多用来三角化的像素点
        sensor_msgs::ChannelFloat32 id_of_point_f;         //三角化许多像素点的3D坐标
        sensor_msgs::ChannelFloat32 u_of_point_f;
        sensor_msgs::ChannelFloat32 v_of_point_f;
        sensor_msgs::ChannelFloat32 velocity_x_of_point_f;
        sensor_msgs::ChannelFloat32 velocity_y_of_point_f;

        feature_incre_points->header = img_msg->header;
        feature_incre_points->header.frame_id = "world";

        //vector<set<int>> hash_ids(NUM_OF_CAM);
        for (int i = 0; i < NUM_OF_CAM; i++)
        {
            auto &un_pts_f = trackerData[i].cur_un_pts_f;  //这边是经过逆畸变处理 得到的没有畸变的归一化的点
            auto &cur_pts_f = trackerData[i].cur_pts_f; //这边是像素坐标系的点
            auto &ids_f = trackerData[i].ids_f;
            //auto &pts_velocity_f = trackerData[i].pts_velocity_f;
            for (unsigned int j = 0; j < ids_f.size(); j++)
            {
                if (trackerData[i].track_cnt_f[j] > 1)
                {
                    int p_id = ids_f[j];
                    //hash_ids[i].insert(p_id);
                    geometry_msgs::Point32 p;
                    p.x = un_pts_f[j].x;   //这边是经过逆畸变处理 得到的没有畸变的归一化的点
                    p.y = un_pts_f[j].y;
                    p.z = 1;

                    feature_incre_points->points.push_back(p);
                    //std::cout << "Point " << p << std::endl;
                    id_of_point_f.values.push_back(p_id * NUM_OF_CAM + i);
                    u_of_point_f.values.push_back(cur_pts_f[j].x);
                    v_of_point_f.values.push_back(cur_pts_f[j].y);


                    //std::cout << "Point u v is  " << cur_pts[j].x <<  cur_pts[j].y  <<  std::endl;

                    velocity_x_of_point_f.values.push_back(0);
                    velocity_y_of_point_f.values.push_back(0);
                }
            }
        }
        feature_incre_points->channels.push_back(id_of_point_f);
        feature_incre_points->channels.push_back(u_of_point_f);
        feature_incre_points->channels.push_back(v_of_point_f);
        feature_incre_points->channels.push_back(velocity_x_of_point_f);
        feature_incre_points->channels.push_back(velocity_y_of_point_f);
        ROS_DEBUG("publish %f, at %f", feature_incre_points->header.stamp.toSec(), ros::Time::now().toSec());
        // skip the first image; since no optical speed on frist image
        if (!init_pub)
        {
            init_pub = 1;
        }
        else
        {
            
            pub_img.publish(feature_points);
            pub_img_feature.publish(feature_incre_points);   //增强过的数据点
            
        }
           






        if (SHOW_TRACK)
        {
            ptr = cv_bridge::cvtColor(ptr, sensor_msgs::image_encodings::BGR8);
            //cv::Mat stereo_img(ROW * NUM_OF_CAM, COL, CV_8UC3);
            ptr_feature = cv_bridge::cvtColor(ptr_feature, sensor_msgs::image_encodings::BGR8);
            cv::Mat stereo_img = ptr->image;
            cv::Mat stereo_img_f = ptr_feature->image;

            for (int i = 0; i < NUM_OF_CAM; i++)
            {
                cv::Mat tmp_img = stereo_img.rowRange(i * ROW, (i + 1) * ROW);
                cv::cvtColor(show_img, tmp_img, CV_GRAY2RGB);

                for (unsigned int j = 0; j < trackerData[i].cur_pts.size(); j++)
                {
                    double len = std::min(1.0, 1.0 * trackerData[i].track_cnt[j] / WINDOW_SIZE);
                    cv::circle(tmp_img, trackerData[i].cur_pts[j], 2, cv::Scalar(255 * (1 - len), 0, 255 * len), 2);
                    //draw speed line
                    /*
                    Vector2d tmp_cur_un_pts (trackerData[i].cur_un_pts[j].x, trackerData[i].cur_un_pts[j].y);
                    Vector2d tmp_pts_velocity (trackerData[i].pts_velocity[j].x, trackerData[i].pts_velocity[j].y);
                    Vector3d tmp_prev_un_pts;
                    tmp_prev_un_pts.head(2) = tmp_cur_un_pts - 0.10 * tmp_pts_velocity;
                    tmp_prev_un_pts.z() = 1;
                    Vector2d tmp_prev_uv;
                    trackerData[i].m_camera->spaceToPlane(tmp_prev_un_pts, tmp_prev_uv);
                    cv::line(tmp_img, trackerData[i].cur_pts[j], cv::Point2f(tmp_prev_uv.x(), tmp_prev_uv.y()), cv::Scalar(255 , 0, 0), 1 , 8, 0);
                    */
                    //char name[10];
                    //sprintf(name, "%d", trackerData[i].ids[j]);
                    //cv::putText(tmp_img, name, trackerData[i].cur_pts[j], cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
                }


                cv::Mat tmp_img_f = stereo_img_f.rowRange(i * ROW, (i + 1) * ROW);
                //cv::cvtColor(show_img, tmp_img, CV_GRAY2RGB);
                //std::cout << "publish feature pts num is " <<  trackerData[i].cur_pts_f.size()   << std::endl; 
                for (unsigned int j = 0; j < trackerData[i].cur_pts_f.size(); j++)
                {
                    double len = std::min(1.0, 1.0 * trackerData[i].track_cnt_f[j] / WINDOW_SIZE);
                    cv::circle(tmp_img_f, trackerData[i].cur_pts_f[j], 2, cv::Scalar(255 * (1 - len), 0, 255 * len), 2);

                }


            }


            //cv::imshow("vis", stereo_img);
            //cv::waitKey(5);
            pub_match.publish(ptr->toImageMsg());
            pub_match_feature.publish(ptr_feature->toImageMsg());
        }
    }
    ROS_INFO("whole feature tracker processing costs: %f", t_r.toc());
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "feature_tracker");
    ros::NodeHandle n("~");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);
    readParameters(n);

    for (int i = 0; i < NUM_OF_CAM; i++)
        trackerData[i].readIntrinsicParameter(CAM_NAMES[i]);

    if(FISHEYE)
    {
        for (int i = 0; i < NUM_OF_CAM; i++)
        {
            trackerData[i].fisheye_mask = cv::imread(FISHEYE_MASK, 0);
            if(!trackerData[i].fisheye_mask.data)
            {
                ROS_INFO("load mask fail");
                ROS_BREAK();
            }
            else
                ROS_INFO("load mask success");
        }
    }

    ros::Subscriber sub_img = n.subscribe(IMAGE_TOPIC, 100, img_callback);

    pub_img = n.advertise<sensor_msgs::PointCloud>("feature", 1000);
    pub_img_feature = n.advertise<sensor_msgs::PointCloud>("feature_incre", 1000);   //多出来的特征
    pub_match = n.advertise<sensor_msgs::Image>("feature_img",1000);
    pub_match_feature = n.advertise<sensor_msgs::Image>("feature_img_incre",1000);
    pub_restart = n.advertise<std_msgs::Bool>("restart",1000);
    /*
    if (SHOW_TRACK)
        cv::namedWindow("vis", cv::WINDOW_NORMAL);
    */
    ros::spin();
    return 0;
}


// new points velocity is 0, pub or not?
// track cnt > 1 pub?