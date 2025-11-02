#pragma once
#ifndef _UTILITY_LIDAR_ODOMETRY_H_
#define _UTILITY_LIDAR_ODOMETRY_H_
#define PCL_NO_PRECOMPILE

#include <ros/ros.h>

#include <std_msgs/Header.h>
#include <std_msgs/Float64MultiArray.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud_conversion.h>
#include <sensor_msgs/NavSatFix.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include <opencv/cv.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/impl/search.hpp>
#include <pcl/range_image/range_image.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/registration/icp.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/crop_box.h> 
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/gicp.h>

#include <tf/LinearMath/Quaternion.h>
#include <tf/transform_listener.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
 
#include <vector>
#include <cmath>
#include <algorithm>
#include <queue>
#include <deque>
#include <iostream>
#include <fstream>
#include <ctime>
#include <cfloat>
#include <iterator>
#include <sstream>
#include <string>
#include <limits>
#include <iomanip>
#include <array>
#include <thread>
#include <mutex>

#include "GpsImu7661/ivsensorgps.h"

#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <omp.h>

struct PointXYZRGBI
{
    PCL_ADD_POINT4D;
    float intensity;
    float diffRange;
    float b;
    float g;
    float r;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;
POINT_CLOUD_REGISTER_POINT_STRUCT (PointXYZRGBI,
                                   (float, x, x) (float, y, y) (float, z, z) (float, intensity, intensity)
                                   (float, diffRange, diffRange) (float, b, b) (float, g, g) (float, r, r)
)

using namespace std;

constexpr double DEG_TO_RAD = M_PI / 180.0;
constexpr double RAD_TO_DEG = 180.0 / M_PI;

double time_start = 0;
double time_end = 0;
int deal_num_start = 0;
int deal_num_end = 0;

typedef PointXYZRGBI PointType;
typedef pcl::PointXYZI PointTypeOrigin;

enum class SensorType { VELODYNE, OUSTER, LIVOX, ROBOSENSE};

class ParamServer
{
public:
    ros::NodeHandle nh;

    std::string robot_id;

    string pointCloudTopic;
    string imuTopic;
    string odomTopic;
    string gpsTopic;

    string lidarFrame;
    string baselinkFrame;
    string odometryFrame;
    string mapFrame;

    bool useImuHeadingInitialization;
    bool useGpsElevation;
    float gpsCovThreshold;
    float poseCovThreshold;
    float useGPSTimeGap;

    bool savePCD;
    string savePCDDirectory;
    string saveTraDirectory;

    SensorType sensor;
    int N_SCAN;
    int Horizon_SCAN;
    int downsampleRate;
    float lidarMinRange;
    float lidarMaxRange;
    int point_filter_num;

    float imuAccNoise;
    float imuGyrNoise;
    float imuAccBiasN;
    float imuGyrBiasN;
    float imuGravity;
    float imuRPYWeight;
    vector<double> extRotV;
    vector<double> extRPYV;
    vector<double> extTransV;
    Eigen::Matrix3d extRot;
    Eigen::Matrix3d extRPY;
    Eigen::Vector3d extTrans;
    Eigen::Quaterniond extQRPY;

    float edgeThreshold;
    float surfThreshold;
    int edgeFeatureMinValidNum;
    int surfFeatureMinValidNum;

    float odometrySurfLeafSize;
    float mappingCornerLeafSize;
    float mappingSurfLeafSize ;

    float z_tollerance;
    bool use_tollerance;
    float rotation_tollerance;
    float pitch_rotation_tollerance;

    int numberOfCores;
    double mappingProcessInterval;

    float surroundingkeyframeAddingDistThreshold; 
    float surroundingkeyframeAddingAngleThreshold; 
    float surroundingKeyframeDensity;
    float surroundingKeyframeSearchRadius;
    
    bool  loopClosureEnableFlag;
    float loopClosureFrequency;
    int   surroundingKeyframeSize;
    float historyKeyframeSearchRadius;
    float historyKeyframeSearchTimeDiff;
    int   historyKeyframeSearchNum;
    float historyKeyframeFitnessScore;

    float globalMapVisualizationSearchRadius;
    float globalMapVisualizationPoseDensity;
    float globalMapVisualizationLeafSize;

    bool  has_ring;
    float ang_bottom;
    float ang_res_y;
    float imu_rate;

    float initialSurfLeafSize;
    int   initialSurfPointsSize;
    float minSurfLeafSize;
    float maxSurfLeafSize;
    float degenerateBoundary;
    float adaptiveRate;

    bool useGPSFactor;
    double lidarHeight;

    ParamServer()
    {
        nh.param<std::string>("/robot_id", robot_id, "roboat");

        nh.param<std::string>("ad_lio/pointCloudTopic", pointCloudTopic, "points_raw");
        nh.param<std::string>("ad_lio/imuTopic", imuTopic, "imu_correct");
        nh.param<std::string>("ad_lio/odomTopic", odomTopic, "odometry/imu");
        nh.param<std::string>("ad_lio/gpsTopic", gpsTopic, "odometry/gps");
        
        nh.param<bool>("ad_lio/has_ring", has_ring, true);
        nh.param<float>("ad_lio/ang_bottom", ang_bottom, 15.0);
        nh.param<float>("ad_lio/ang_res_y", ang_res_y, 1.0);
        nh.param<float>("ad_lio/imuRate", imu_rate, 100);

        nh.param<std::string>("ad_lio/lidarFrame", lidarFrame, "base_link");
        nh.param<std::string>("ad_lio/baselinkFrame", baselinkFrame, "base_link");
        nh.param<std::string>("ad_lio/odometryFrame", odometryFrame, "odom");
        nh.param<std::string>("ad_lio/mapFrame", mapFrame, "map");

        nh.param<bool>("ad_lio/useImuHeadingInitialization", useImuHeadingInitialization, false);
        nh.param<bool>("ad_lio/useGpsElevation", useGpsElevation, false);
        nh.param<float>("ad_lio/gpsCovThreshold", gpsCovThreshold, 2.0);
        nh.param<float>("ad_lio/useGPSTimeGap", useGPSTimeGap, 0.5);
        nh.param<float>("ad_lio/poseCovThreshold", poseCovThreshold, 25.0);
        nh.param<bool>("ad_lio/useGPSFactor", useGPSFactor, false);
        nh.param<double>("ad_lio/lidarHeight", lidarHeight, 0.0);

        nh.param<bool>("ad_lio/savePCD", savePCD, false);
        nh.param<std::string>("ad_lio/savePCDDirectory", savePCDDirectory, "/home/whq/");
        nh.param<std::string>("ad_lio/saveTraDirectory", saveTraDirectory, "/home/whq/trajectory.txt");

        std::string sensorStr;
        nh.param<std::string>("ad_lio/sensor", sensorStr, "");
        if (sensorStr == "velodyne")
        {
            sensor = SensorType::VELODYNE;
        }
        else if (sensorStr == "ouster")
        {
            sensor = SensorType::OUSTER;
        }
        else if (sensorStr == "livox")
        {
            sensor = SensorType::LIVOX;
        }
        else if (sensorStr == "robosense")
        {
            sensor = SensorType::ROBOSENSE;
        }
        else
        {
            ROS_ERROR_STREAM(
                "Invalid sensor type (must be either 'velodyne' or 'ouster' or 'livox' or 'robosense'): " << sensorStr);
            ros::shutdown();
        }

        nh.param<int>("ad_lio/N_SCAN", N_SCAN, 16);
        nh.param<int>("ad_lio/Horizon_SCAN", Horizon_SCAN, 1800);
        nh.param<int>("ad_lio/downsampleRate", downsampleRate, 1);
        nh.param<float>("ad_lio/lidarMinRange", lidarMinRange, 1.0);
        nh.param<float>("ad_lio/lidarMaxRange", lidarMaxRange, 1000.0);
        nh.param<int>("ad_lio/point_filter_num", point_filter_num, 1);

        nh.param<float>("ad_lio/imuAccNoise", imuAccNoise, 0.01);
        nh.param<float>("ad_lio/imuGyrNoise", imuGyrNoise, 0.001);
        nh.param<float>("ad_lio/imuAccBiasN", imuAccBiasN, 0.0002);
        nh.param<float>("ad_lio/imuGyrBiasN", imuGyrBiasN, 0.00003);
        nh.param<float>("ad_lio/imuGravity", imuGravity, 9.80511);
        nh.param<float>("ad_lio/imuRPYWeight", imuRPYWeight, 0.01);
        nh.param<vector<double>>("ad_lio/extrinsicRot", extRotV, vector<double>());
        nh.param<vector<double>>("ad_lio/extrinsicRPY", extRPYV, vector<double>());
        nh.param<vector<double>>("ad_lio/extrinsicTrans", extTransV, vector<double>());
        extRot = Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(extRotV.data(), 3, 3);
        extRPY = Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(extRPYV.data(), 3, 3);
        extTrans = Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(extTransV.data(), 3, 1);
        extQRPY = Eigen::Quaterniond(extRPY).inverse();

        nh.param<float>("ad_lio/edgeThreshold", edgeThreshold, 0.1);
        nh.param<float>("ad_lio/surfThreshold", surfThreshold, 0.1);
        nh.param<int>("ad_lio/edgeFeatureMinValidNum", edgeFeatureMinValidNum, 10);
        nh.param<int>("ad_lio/surfFeatureMinValidNum", surfFeatureMinValidNum, 100);

        nh.param<float>("ad_lio/odometrySurfLeafSize", odometrySurfLeafSize, 0.2);
        nh.param<float>("ad_lio/mappingCornerLeafSize", mappingCornerLeafSize, 0.2);
        nh.param<float>("ad_lio/mappingSurfLeafSize", mappingSurfLeafSize, 0.2);

        nh.param<float>("ad_lio/initialSurfLeafSize", initialSurfLeafSize, 0.5);
        nh.param<int>("ad_lio/initialSurfPointsSize", initialSurfPointsSize, 3000);
        nh.param<float>("ad_lio/minSurfLeafSize", minSurfLeafSize, 0.2);
        nh.param<float>("ad_lio/maxSurfLeafSize", maxSurfLeafSize, 1.2);
        nh.param<float>("ad_lio/degenerateBoundary", degenerateBoundary, 0.8);
        nh.param<float>("ad_lio/adaptiveRate", adaptiveRate, 0.75);

        nh.param<float>("ad_lio/z_tollerance", z_tollerance, FLT_MAX);
        nh.param<bool>("ad_lio/use_tollerance", use_tollerance, false);
        nh.param<float>("ad_lio/rotation_tollerance", rotation_tollerance, FLT_MAX);
        nh.param<float>("ad_lio/pitch_rotation"
                        "_tollerance", pitch_rotation_tollerance, FLT_MAX);

        nh.param<int>("ad_lio/numberOfCores", numberOfCores, 2);
        nh.param<double>("ad_lio/mappingProcessInterval", mappingProcessInterval, 0.15);

        nh.param<float>("ad_lio/surroundingkeyframeAddingDistThreshold", surroundingkeyframeAddingDistThreshold, 1.0);
        nh.param<float>("ad_lio/surroundingkeyframeAddingAngleThreshold", surroundingkeyframeAddingAngleThreshold, 0.2);
        nh.param<float>("ad_lio/surroundingKeyframeDensity", surroundingKeyframeDensity, 1.0);
        nh.param<float>("ad_lio/surroundingKeyframeSearchRadius", surroundingKeyframeSearchRadius, 50.0);

        nh.param<bool>("ad_lio/loopClosureEnableFlag", loopClosureEnableFlag, false);
        nh.param<float>("ad_lio/loopClosureFrequency", loopClosureFrequency, 1.0);
        nh.param<int>("ad_lio/surroundingKeyframeSize", surroundingKeyframeSize, 50);
        nh.param<float>("ad_lio/historyKeyframeSearchRadius", historyKeyframeSearchRadius, 10.0);
        nh.param<float>("ad_lio/historyKeyframeSearchTimeDiff", historyKeyframeSearchTimeDiff, 30.0);
        nh.param<int>("ad_lio/historyKeyframeSearchNum", historyKeyframeSearchNum, 25);
        nh.param<float>("ad_lio/historyKeyframeFitnessScore", historyKeyframeFitnessScore, 0.3);

        nh.param<float>("ad_lio/globalMapVisualizationSearchRadius", globalMapVisualizationSearchRadius, 1e3);
        nh.param<float>("ad_lio/globalMapVisualizationPoseDensity", globalMapVisualizationPoseDensity, 10.0);
        nh.param<float>("ad_lio/globalMapVisualizationLeafSize", globalMapVisualizationLeafSize, 1.0);

        usleep(100);
    }

    sensor_msgs::Imu imuConverterKPQ(GpsImu7661::ivsensorgps gpsimu)
    {
        sensor_msgs::Imu imu_out;

        Eigen::Vector3d acc(gpsimu.accx, gpsimu.accy, gpsimu.accz);
        acc = extRot * acc;
        imu_out.linear_acceleration.x = acc.x();
        imu_out.linear_acceleration.y = acc.y();
        imu_out.linear_acceleration.z = acc.z();

        Eigen::Vector3d gyr(gpsimu.angx * DEG_TO_RAD, gpsimu.angy * DEG_TO_RAD, gpsimu.yaw * DEG_TO_RAD);
        gyr = extRot * gyr;
        imu_out.angular_velocity.x = gyr.x();
        imu_out.angular_velocity.y = gyr.y();
        imu_out.angular_velocity.z = gyr.z();

        Eigen::AngleAxisd rollAngle(Eigen::AngleAxisd(gpsimu.roll * DEG_TO_RAD,Eigen::Vector3d::UnitX()));
        Eigen::AngleAxisd pitchAngle(Eigen::AngleAxisd(gpsimu.pitch * DEG_TO_RAD,Eigen::Vector3d::UnitY()));
        Eigen::AngleAxisd yawAngle(Eigen::AngleAxisd((-gpsimu.heading + 360) * DEG_TO_RAD,Eigen::Vector3d::UnitZ()));
        Eigen::Quaterniond q_from;
        q_from = yawAngle * pitchAngle * rollAngle;

        Eigen::Quaterniond q_final = q_from * extQRPY;
        imu_out.orientation.x = q_final.x();
        imu_out.orientation.y = q_final.y();
        imu_out.orientation.z = q_final.z();
        imu_out.orientation.w = q_final.w();
        imu_out.header = gpsimu.header;

        if (sqrt(q_final.x()*q_final.x() + q_final.y()*q_final.y() + q_final.z()*q_final.z() + q_final.w()*q_final.w()) < 0.1)
        {
            ROS_ERROR("Invalid quaternion, please use a 9-axis IMU!");
            ros::shutdown();
        }

        return imu_out;
    }
};

template<typename T>
sensor_msgs::PointCloud2 publishCloud(const ros::Publisher& thisPub, const T& thisCloud, ros::Time thisStamp, std::string thisFrame)
{
    sensor_msgs::PointCloud2 tempCloud;
    pcl::toROSMsg(*thisCloud, tempCloud);
    tempCloud.header.stamp = thisStamp;
    tempCloud.header.frame_id = thisFrame;
    if (thisPub.getNumSubscribers() != 0)
        thisPub.publish(tempCloud);
    return tempCloud;
}

template<typename T>
double ROS_TIME(T msg)
{
    return msg->header.stamp.toSec();
}


template<typename T>
void imuAngular2rosAngular(sensor_msgs::Imu *thisImuMsg, T *angular_x, T *angular_y, T *angular_z)
{
    *angular_x = thisImuMsg->angular_velocity.x;
    *angular_y = thisImuMsg->angular_velocity.y;
    *angular_z = thisImuMsg->angular_velocity.z;
}


template<typename T>
void imuAccel2rosAccel(sensor_msgs::Imu *thisImuMsg, T *acc_x, T *acc_y, T *acc_z)
{
    *acc_x = thisImuMsg->linear_acceleration.x;
    *acc_y = thisImuMsg->linear_acceleration.y;
    *acc_z = thisImuMsg->linear_acceleration.z;
}


template<typename T>
void imuRPY2rosRPY(sensor_msgs::Imu *thisImuMsg, T *rosRoll, T *rosPitch, T *rosYaw)
{
    double imuRoll, imuPitch, imuYaw;
    tf::Quaternion orientation;
    tf::quaternionMsgToTF(thisImuMsg->orientation, orientation);
    tf::Matrix3x3(orientation).getRPY(imuRoll, imuPitch, imuYaw);

    *rosRoll = imuRoll;
    *rosPitch = imuPitch;
    *rosYaw = imuYaw;
}


float pointDistance(PointType p)
{
    return sqrt(p.x*p.x + p.y*p.y + p.z*p.z);
}


float pointDistance(PointType p1, PointType p2)
{
    return sqrt((p1.x-p2.x)*(p1.x-p2.x) + (p1.y-p2.y)*(p1.y-p2.y) + (p1.z-p2.z)*(p1.z-p2.z));
}

#endif
