#include "utility.h"
#include "ad_lio/cloud_info.h"

struct VelodynePointXYZIRT
{
    PCL_ADD_POINT4D
    PCL_ADD_INTENSITY;
    uint16_t ring;
    float time;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;
POINT_CLOUD_REGISTER_POINT_STRUCT (VelodynePointXYZIRT,
    (float, x, x) (float, y, y) (float, z, z) (float, intensity, intensity)
    (uint16_t, ring, ring) (float, time, time)
)

struct OusterPointXYZIRT {
    PCL_ADD_POINT4D;
    float intensity;
    uint32_t t;
    uint16_t reflectivity;
    uint8_t ring;
    uint16_t noise;
    uint32_t range;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;
POINT_CLOUD_REGISTER_POINT_STRUCT(OusterPointXYZIRT,
    (float, x, x) (float, y, y) (float, z, z) (float, intensity, intensity)
    (uint32_t, t, t) (uint16_t, reflectivity, reflectivity)
    (uint8_t, ring, ring) (uint16_t, noise, noise) (uint32_t, range, range)
)

struct RoboSensePointXYZIRT
{
    PCL_ADD_POINT4D
    uint8_t intensity;
    double timestamp;
    uint16_t ring;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;
POINT_CLOUD_REGISTER_POINT_STRUCT (RoboSensePointXYZIRT,
    (float, x, x) (float, y, y) (float, z, z) (uint8_t, intensity, intensity)
    (uint16_t, ring, ring) (double, timestamp, timestamp)
)


using PointXYZIRT = VelodynePointXYZIRT;
const int queueLength = 2000;

class ImageProjection : public ParamServer
{
private:

    std::mutex imuLock;
    std::mutex odoLock;

    ros::Subscriber subLaserCloud;
    ros::Publisher  pubLaserCloud;
    
    ros::Publisher pubExtractedCloud;
    ros::Publisher pubLaserCloudInfo;

    ros::Subscriber subImu;
    std::deque<sensor_msgs::Imu> imuQueue;

    ros::Subscriber subOdom;
    std::deque<nav_msgs::Odometry> odomQueue;

    ros::Subscriber subCam;

    std::deque<sensor_msgs::PointCloud2> cloudQueue;
    sensor_msgs::PointCloud2 currentCloudMsg;


    double *imuTime = new double[queueLength];
    double *imuRotX = new double[queueLength];
    double *imuRotY = new double[queueLength];
    double *imuRotZ = new double[queueLength];

    int imuPointerCur;
    bool firstPointFlag;
    Eigen::Affine3f transStartInverse;

    pcl::PointCloud<PointXYZIRT>::Ptr laserCloudIn;
    pcl::PointCloud<OusterPointXYZIRT>::Ptr tmpOusterCloudIn;
    pcl::PointCloud<PointType>::Ptr   fullCloud;
    pcl::PointCloud<PointType>::Ptr   extractedCloud;

    int deskewFlag;
    cv::Mat rangeMat;

    bool odomDeskewFlag;
    float odomIncreX;
    float odomIncreY;
    float odomIncreZ;

    ad_lio::cloud_info cloudInfo;
    double timeScanCur;
    double timeScanEnd;
    std_msgs::Header cloudHeader;

    vector<int> columnIdnCountVec;

    // add by sx
    pcl::PointCloud<RoboSensePointXYZIRT>::Ptr tmpRobosenseCloudIn;

    deque<cv::Mat> LastImage;

public:
    ImageProjection():
    deskewFlag(0)
    {
        subImu        = nh.subscribe<GpsImu7661::ivsensorgps>(imuTopic, 2000, &ImageProjection::imuHandler, this, ros::TransportHints().tcpNoDelay());
        subOdom       = nh.subscribe<nav_msgs::Odometry>(odomTopic+"_incremental", 2000, &ImageProjection::odometryHandler, this, ros::TransportHints().tcpNoDelay());
        subLaserCloud = nh.subscribe<sensor_msgs::PointCloud>(pointCloudTopic, 5, &ImageProjection::cloudHandler, this, ros::TransportHints().tcpNoDelay());

        pubExtractedCloud = nh.advertise<sensor_msgs::PointCloud2> ("ad_lio/deskew/cloud_deskewed", 1);
        pubLaserCloudInfo = nh.advertise<ad_lio::cloud_info> ("ad_lio/deskew/cloud_info", 1);

        allocateMemory();
        resetParameters();

        pcl::console::setVerbosityLevel(pcl::console::L_ERROR);
    }

    void allocateMemory()
    {
        laserCloudIn.reset(new pcl::PointCloud<PointXYZIRT>());
        tmpOusterCloudIn.reset(new pcl::PointCloud<OusterPointXYZIRT>());
        fullCloud.reset(new pcl::PointCloud<PointType>());
        extractedCloud.reset(new pcl::PointCloud<PointType>());
        tmpRobosenseCloudIn.reset(new pcl::PointCloud<RoboSensePointXYZIRT>());

        fullCloud->points.resize(N_SCAN*Horizon_SCAN);

        cloudInfo.startRingIndex.assign(N_SCAN, 0);
        cloudInfo.endRingIndex.assign(N_SCAN, 0);

        cloudInfo.pointColInd.assign(N_SCAN*Horizon_SCAN, 0);
        cloudInfo.pointRange.assign(N_SCAN*Horizon_SCAN, 0);

        resetParameters();
    }

    void resetParameters()
    {
        laserCloudIn->clear();
        extractedCloud->clear();
        rangeMat = cv::Mat(N_SCAN, Horizon_SCAN, CV_32F, cv::Scalar::all(FLT_MAX));

        imuPointerCur = 0;
        firstPointFlag = true;
        odomDeskewFlag = false;

        for (int i = 0; i < queueLength; ++i)
        {
            imuTime[i] = 0;
            imuRotX[i] = 0;
            imuRotY[i] = 0;
            imuRotZ[i] = 0;
        }

        columnIdnCountVec.assign(N_SCAN, 0);
    }

    ~ImageProjection(){}

    void imuHandler(GpsImu7661::ivsensorgps imuMsg)
    {
        sensor_msgs::Imu thisImu = imuConverterKPQ(imuMsg);

        std::lock_guard<std::mutex> lock1(imuLock);
        imuQueue.push_back(thisImu);
    }

    void odometryHandler(const nav_msgs::Odometry::ConstPtr& odometryMsg)
    {
        std::lock_guard<std::mutex> lock2(odoLock);
        odomQueue.push_back(*odometryMsg);
    }

    void cloudHandler(const sensor_msgs::PointCloudConstPtr& laserCloudMsgPC)
    {
        sensor_msgs::PointCloud2 laserCloudMsg;
        sensor_msgs::convertPointCloudToPointCloud2(*laserCloudMsgPC, laserCloudMsg);
        laserCloudMsg.header.stamp = laserCloudMsgPC->header.stamp;
        laserCloudMsg.header.frame_id = lidarFrame;

        vector<float> cloudIntensity;
        cloudIntensity.resize(laserCloudMsgPC->points.size());
        for (int i = 0; i < laserCloudMsgPC->points.size(); i++)
        {
            cloudIntensity[i] = laserCloudMsgPC->channels[0].values[i];
        }

        if (!cachePointCloud(laserCloudMsg, cloudIntensity))
            return;

        if (!deskewInfo())
            return;

        projectPointCloud();

        publishClouds();

        resetParameters();
    }

    bool cachePointCloud(const sensor_msgs::PointCloud2 laserCloudMsg, vector<float> cloudIntensity)
    {
        cloudQueue.push_back(laserCloudMsg);
        if (cloudQueue.size() <= 2)
            return false;

        currentCloudMsg = std::move(cloudQueue.front());
        cloudQueue.pop_front();
        if (sensor == SensorType::ROBOSENSE)
        {
            tmpRobosenseCloudIn->clear();
            pcl::moveFromROSMsg(currentCloudMsg, *tmpRobosenseCloudIn);
            laserCloudIn->clear();
            laserCloudIn->points.resize(tmpRobosenseCloudIn->size());
            laserCloudIn->is_dense = tmpRobosenseCloudIn->is_dense ? tmpRobosenseCloudIn->is_dense : true;
            double start_time = currentCloudMsg.header.stamp.toSec();
            for (size_t i = 0; i < tmpRobosenseCloudIn->size(); i++)
            {
                auto &src = tmpRobosenseCloudIn->points[i];
                auto &dst = laserCloudIn->points[i];
                dst.x = src.x;
                dst.y = src.y;
                dst.z = src.z + lidarHeight;
                if(dst.x >= 0.5 || dst.x <= -0.5 || dst.y >= 0.5 || dst.y <= -0.5)
                {
                        if(cloudIntensity[i] > 10)
                        {
                            dst.intensity = 10;
                        }
                        else
                        {
                            dst.intensity = cloudIntensity[i];
                        }
                        dst.time = src.timestamp - start_time;
                }
            }
        }
        else
        {
            ROS_ERROR_STREAM("Unknown sensor type: " << int(sensor));
            ros::shutdown();
        }

        cloudHeader = currentCloudMsg.header;
        timeScanCur = cloudHeader.stamp.toSec();
        timeScanEnd = timeScanCur + laserCloudIn->points.back().time;

        if (laserCloudIn->is_dense == false)
        {
            ROS_ERROR("Point cloud is not in dense format, please remove NaN points first!");
            ros::shutdown();
        }
        return true;
    }

    bool deskewInfo()
    {
        std::lock_guard<std::mutex> lock1(imuLock);
        std::lock_guard<std::mutex> lock2(odoLock);

        if (imuQueue.empty() || imuQueue.front().header.stamp.toSec() > timeScanCur || imuQueue.back().header.stamp.toSec() < timeScanEnd)
        {
            ROS_DEBUG("Waiting for IMU data ...");
            return false;
        }

        imuDeskewInfo();

        odomDeskewInfo();

        return true;
    }

    void imuDeskewInfo()
    {
        cloudInfo.imuAvailable = false;

        while (!imuQueue.empty())
        {
            if (imuQueue.front().header.stamp.toSec() < timeScanCur - 0.01)
                imuQueue.pop_front();
            else
                break;
        }

        if (imuQueue.empty())
            return;

        imuPointerCur = 0;

        for (int i = 0; i < (int)imuQueue.size(); ++i)
        {
            sensor_msgs::Imu thisImuMsg = imuQueue[i];
            double currentImuTime = thisImuMsg.header.stamp.toSec();

            if (currentImuTime <= timeScanCur)
                imuRPY2rosRPY(&thisImuMsg, &cloudInfo.imuRollInit, &cloudInfo.imuPitchInit, &cloudInfo.imuYawInit);

            if (currentImuTime > timeScanEnd + 0.01)
                break;

            if (imuPointerCur == 0)
            {
                imuRotX[0] = 0;
                imuRotY[0] = 0;
                imuRotZ[0] = 0;
                imuTime[0] = currentImuTime;
                ++imuPointerCur;
                continue;
            }

            double angular_x, angular_y, angular_z;
            imuAngular2rosAngular(&thisImuMsg, &angular_x, &angular_y, &angular_z);

            double timeDiff = currentImuTime - imuTime[imuPointerCur-1];
            imuRotX[imuPointerCur] = imuRotX[imuPointerCur-1] + angular_x * timeDiff;
            imuRotY[imuPointerCur] = imuRotY[imuPointerCur-1] + angular_y * timeDiff;
            imuRotZ[imuPointerCur] = imuRotZ[imuPointerCur-1] + angular_z * timeDiff;
            imuTime[imuPointerCur] = currentImuTime;
            ++imuPointerCur;
        }

        --imuPointerCur;

        if (imuPointerCur <= 0)
            return;

        cloudInfo.imuAvailable = true;
    }

    void odomDeskewInfo()
    {
        static float delay_time = imu_rate > 300 ? 0.01 : 0.20;

        cloudInfo.odomAvailable = false;

        while (!odomQueue.empty())
        {
            if (odomQueue.front().header.stamp.toSec() < timeScanCur - delay_time)
                odomQueue.pop_front();
            else
                break;
        }

        if (odomQueue.empty())
            return;

        if (odomQueue.front().header.stamp.toSec() > timeScanCur)
            return;

        nav_msgs::Odometry startOdomMsg;

        for (int i = 0; i < (int)odomQueue.size(); ++i)
        {
            startOdomMsg = odomQueue[i];

            if (ROS_TIME(&startOdomMsg) < timeScanCur)
                continue;
            else
                break;
        }

        tf::Quaternion orientation;
        tf::quaternionMsgToTF(startOdomMsg.pose.pose.orientation, orientation);

        double roll, pitch, yaw;
        tf::Matrix3x3(orientation).getRPY(roll, pitch, yaw);

        cloudInfo.initialGuessX = startOdomMsg.pose.pose.position.x;
        cloudInfo.initialGuessY = startOdomMsg.pose.pose.position.y;
        cloudInfo.initialGuessZ = startOdomMsg.pose.pose.position.z;
        cloudInfo.initialGuessRoll  = roll;
        cloudInfo.initialGuessPitch = pitch;
        cloudInfo.initialGuessYaw   = yaw;

        cloudInfo.odomAvailable = true;

        odomDeskewFlag = false;

        if (odomQueue.back().header.stamp.toSec() < timeScanEnd)
            return;

        nav_msgs::Odometry endOdomMsg;

        for (int i = 0; i < (int)odomQueue.size(); ++i)
        {
            endOdomMsg = odomQueue[i];

            if (ROS_TIME(&endOdomMsg) < timeScanEnd)
                continue;
            else
                break;
        }

        if (int(round(startOdomMsg.pose.covariance[0])) != int(round(endOdomMsg.pose.covariance[0])))
            return;

        Eigen::Affine3f transBegin = pcl::getTransformation(startOdomMsg.pose.pose.position.x, startOdomMsg.pose.pose.position.y, startOdomMsg.pose.pose.position.z, roll, pitch, yaw);

        tf::quaternionMsgToTF(endOdomMsg.pose.pose.orientation, orientation);
        tf::Matrix3x3(orientation).getRPY(roll, pitch, yaw);
        Eigen::Affine3f transEnd = pcl::getTransformation(endOdomMsg.pose.pose.position.x, endOdomMsg.pose.pose.position.y, endOdomMsg.pose.pose.position.z, roll, pitch, yaw);

        Eigen::Affine3f transBt = transBegin.inverse() * transEnd;

        float rollIncre, pitchIncre, yawIncre;
        pcl::getTranslationAndEulerAngles(transBt, odomIncreX, odomIncreY, odomIncreZ, rollIncre, pitchIncre, yawIncre);

        odomDeskewFlag = true;
    }

    void projectPointCloud()
    {
        fullCloud->clear();
        fullCloud->points.resize(N_SCAN*Horizon_SCAN);
        int cloudSize = laserCloudIn->points.size();

        for (int i = 0; i < cloudSize; ++i)
        {
            if( i % point_filter_num == 0)
            {
                int currentCircle = i / 1800;
                if(currentCircle % downsampleRate == 0)
                {
                    PointType thisPoint;
                    thisPoint.x = laserCloudIn->points[i].x;
                    thisPoint.y = laserCloudIn->points[i].y;
                    thisPoint.z = laserCloudIn->points[i].z;
                    thisPoint.intensity = laserCloudIn->points[i].intensity;

                    float range = pointDistance(thisPoint);
                    if (range < lidarMinRange || range > lidarMaxRange)
                        continue;

                    fullCloud->points[i] = thisPoint;
                }
            }
        }
    }
    
    void publishClouds()
    {
        cloudInfo.header = cloudHeader;
        cloudInfo.cloud_deskewed  = publishCloud(pubExtractedCloud, fullCloud, cloudHeader.stamp, lidarFrame);
        pubLaserCloudInfo.publish(cloudInfo);
    }
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "ad_lio");

    ImageProjection IP;
    
    ROS_INFO("\033[1;32m----> Image Projection Started.\033[0m");

    ros::MultiThreadedSpinner spinner(3);
    spinner.spin();
    
    return 0;
}
