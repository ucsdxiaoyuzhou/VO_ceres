#include "utils.h"

using namespace std;
using namespace cv;
using namespace pcl;

vector<string> getPCDFileName(string &strPathPCD)
{

    string cpstrPathPCD = strPathPCD;

    vector<string> vstrPCD;

    DIR *dir;
    class dirent *ent;
    class stat st;

    dir = opendir(cpstrPathPCD.c_str());

    while ((ent = readdir(dir)) != NULL) {
        const string file_name = ent->d_name;
        const string full_file_name_bin = cpstrPathPCD + "/" + file_name;
        int nameLen = file_name.length();

        if (file_name[0] == '.')
            continue;

        if (stat(full_file_name_bin.c_str(), &st) == -1)
            continue;

        if(file_name.substr(nameLen-3,3) != "pcd")
            continue;
        const bool is_directory = (st.st_mode & S_IFDIR) != 0;

        if (is_directory)
            continue;

        vstrPCD.push_back(full_file_name_bin);
    }
    closedir(dir);
    sort(vstrPCD.begin(), vstrPCD.end());

    return vstrPCD;
}


vector<string> getIMUFileName(string &strPathIMU)
{

    string cpstrPathIMU = strPathIMU;

    vector<string> vstrIMU;

    DIR *dir;
    class dirent *ent;
    class stat st;

    dir = opendir(cpstrPathIMU.c_str());

    while ((ent = readdir(dir)) != NULL) {
        const string file_name = ent->d_name;
        const string full_file_name_imu = cpstrPathIMU + "/" + file_name;
        int len = file_name.length();
        if (file_name[0] == '.')
            continue;
        
        if(file_name.substr(len-3,3) != "txt")
            continue;

        if (stat(full_file_name_imu.c_str(), &st) == -1)
            continue;

        const bool is_directory = (st.st_mode & S_IFDIR) != 0;

        if (is_directory)
            continue;

        vstrIMU.push_back(full_file_name_imu);
    }
    closedir(dir);
    sort(vstrIMU.begin(), vstrIMU.end());

    return vstrIMU;
}

vector<string> getImgFileName(string &strPathImg)
{

    string cpstrPathImg = strPathImg;

    vector<string> vstrImage;

    DIR *dir;
    class dirent *ent;
    class stat st;

    dir = opendir(cpstrPathImg.c_str());

    while ((ent = readdir(dir)) != NULL) {
        const string file_name = ent->d_name;
        const string full_file_name = cpstrPathImg + "/" + file_name;

        if (file_name[0] == '.')
            continue;

        if (stat(full_file_name.c_str(), &st) == -1)
            continue;

        const bool is_directory = (st.st_mode & S_IFDIR) != 0;

        if (is_directory)
            continue;

        vstrImage.push_back(full_file_name);
    }
    closedir(dir);
    sort(vstrImage.begin(), vstrImage.end());

    return vstrImage;
}

void getRectificationParams(Mat K1, Mat K2, 
                            Mat R, Mat t,
                            Mat distort1,
                            Mat distort2,
                            STEREO_RECTIFY_PARAMS& srp)
{
    Rect validRoi[2];

    stereoRectify(K1,distort1, K2, distort2,
                  srp.imageSize, R, t,
                  srp.R1, srp.R2,
                  srp.P1, srp.P2,
                  srp.Q,
                  CALIB_ZERO_DISPARITY, 0, srp.imageSize,
                  &validRoi[0], &validRoi[1]);
    // cout << srp.Q << endl;
    initUndistortRectifyMap(K1, distort1, srp.R1, srp.P1,
                            srp.imageSize, CV_32F, 
                            srp.map11, srp.map12);

    initUndistortRectifyMap(K2, distort2, srp.R2, srp.P2,
                            srp.imageSize, CV_32F,
                            srp.map21, srp.map22);    
}


vector<Eigen::Affine3d> loadGPSIMU(const vector<string>& strPathIMU)
{
    vector<Eigen::Affine3d> transformation;
    bool getfirst = true;
    Eigen::Affine3d firstPos;
    for(int n = 0; n < strPathIMU.size(); n++){
        FILE *fp = fopen(strPathIMU[n].c_str(),"r");
        if (!fp)
            cout << "Could not open transformation file:  "<<strPathIMU[n] << endl;
        exit;
        while (!feof(fp)) {
            double T[6];
            if (fscanf(fp, "%lf %lf %lf %lf %lf %lf\n",
                       &T[0], &T[1], &T[2], &T[3], &T[4], &T[5])==6) {
                //latitude, longtitude, altitude, roll, pitch, yaw
                double x, y, z;
                convertGPScoordinateToTranslation(T[0],T[1],T[2],x,y,z);

                Eigen::Affine3d trans;
                GetTransformationFromRollPitchYawAndTranslation(trans, T[3], T[4], T[5],x,y,z);

                    // cout << trans.matrix() << endl<<endl<<endl;
                if(getfirst == true){
                    firstPos = trans;
                    getfirst = false;
                    Eigen::Affine3d temp = Eigen::Affine3d::Identity();
                    transformation.push_back(temp);
                }
                else{
                    trans = firstPos.inverse() * trans;

                    transformation.push_back(trans);
                }


            }
            else{
                cout << "format wrong! "<<endl;
            }
        }
        fclose(fp);
    }
    return transformation;
}

vector<Eigen::Affine3d> getTransformFromGPSIMUinCamera(vector<string> IMUFileName){

    vector<Eigen::Affine3d> transformGPSIMU = loadGPSIMU(IMUFileName);
    Eigen::Affine3d fromLiDARtoRAW = Eigen::Affine3d::Identity();
    fromLiDARtoRAW(0,2) = 1;
    fromLiDARtoRAW(1,0) = 1;
    fromLiDARtoRAW(2,1) = 1;
    fromLiDARtoRAW(3,3) = 1;

    fromLiDARtoRAW(0,0) = 0;
    fromLiDARtoRAW(1,1) = 0;
    fromLiDARtoRAW(2,2) = 0;
    Eigen::Affine3d fromCam1toRaw = Eigen::Affine3d::Identity();
    fromCam1toRaw(0,0) = 2.04979302e-02;
    fromCam1toRaw(0,1) = 9.78461317e-03;
    fromCam1toRaw(0,2) = 9.99741993e-01;
    fromCam1toRaw(1,0) = -9.99789535e-01;
    fromCam1toRaw(1,1) = 9.62882439e-04; 
    fromCam1toRaw(1,2) = 2.04895366e-02;
    fromCam1toRaw(2,0) = -7.62148293e-04;
    fromCam1toRaw(2,1) = -9.99951667e-01;
    fromCam1toRaw(2,2) = 9.80219750e-03;
    fromCam1toRaw(0,3) = 7.76330490e-01;
    fromCam1toRaw(1,3) = 5.36925340e-01;
    fromCam1toRaw(2,3) = -1.91739741e-01;

    double p[6];
    Eigen::Affine3d fromLiDARToGPSIMU = Eigen::Affine3d::Identity();
    p[0] = 1.51;
    p[1] =  -0.0181572762417 ;
    p[2] = 3.16936854475 ;
    p[3] = -0.0173696789743;
    p[4] = 1.33480107981 ;
    p[5] = 1.24589891942;
    GetTransformationFromEulerAngleAndTranslation(fromLiDARToGPSIMU, p[0], p[1], p[2], p[3], p[4], p[5]);

    Eigen::Affine3d fromRawToCam1 = fromCam1toRaw.inverse();
    // Eigen::Affine3d fromLiDARtoCam1 = fromLiDARtoRAW*fromRawToCam1;

    Eigen::Affine3d fromLiDARtoCam1 = fromRawToCam1*fromLiDARtoRAW;

    vector<Eigen::Affine3d> result;


    for(int n = 0; n < IMUFileName.size()-1; n++){
        Eigen::Affine3d T1to2Image = Eigen::Affine3d::Identity();

        Eigen::Affine3d T1 = transformGPSIMU[n];//imu
        Eigen::Affine3d T2 = transformGPSIMU[n+1];

        Eigen::Affine3d T1to2IMU = T2.inverse() * T1;

        T1to2Image = fromLiDARtoCam1 * fromLiDARToGPSIMU.inverse() * T1to2IMU * fromLiDARToGPSIMU*fromLiDARtoCam1.inverse();


        // Eigen::Affine3d T1to2IMU = T1.inverse() * T2;


        // T1to2Image = fromLiDARtoCam1.inverse() *fromLiDARToGPSIMU*T1to2IMU *fromLiDARToGPSIMU.inverse()* fromLiDARtoCam1;
        // cout << T1to2Image.matrix() << endl<<endl;
        result.push_back(T1to2Image);

    }
    return result;

}


void GetTransformationFromEulerAngleAndTranslation(Eigen::Affine3d& Transformation,
                                                   double rx, double ry, double rz, 
                                                   double tx, double ty, double tz)
{

    Eigen::Affine3d Rx = Eigen::Affine3d::Identity();
    Eigen::Affine3d Ry = Eigen::Affine3d::Identity();
    Eigen::Affine3d Rz = Eigen::Affine3d::Identity();

    Rx(0,0) = 1;
    Rx(0,1) = 0;
    Rx(0,2) = 0;
    Rx(1,0) = 0;
    Rx(1,1) = std::cos(rx);
    Rx(1,2) = -std::sin(rx);
    Rx(2,0) = 0;
    Rx(2,1) = std::sin(rx);
    Rx(2,2) = std::cos(rx);

    Ry(0,0) = std::cos(ry);
    Ry(0,1) = 0;
    Ry(0,2) = std::sin(ry);
    Ry(1,0) = 0;
    Ry(1,1) = 1;
    Ry(1,2) = 0;
    Ry(2,0) = -std::sin(ry);
    Ry(2,1) = 0;
    Ry(2,2) = std::cos(ry);

    Rz(0,0) = std::cos(rz);
    Rz(0,1) = -std::sin(rz);
    Rz(0,2) = 0;
    Rz(1,0) = std::sin(rz);
    Rz(1,1) = std::cos(rz);
    Rz(1,2) = 0;
    Rz(2,0) = 0;
    Rz(2,1) = 0;
    Rz(2,2) = 1;

    Eigen::Affine3d R = Rz*Ry*Rx;

    // Set rotation
    Transformation(0,0) = R(0,0);
    Transformation(0,1) = R(0,1);
    Transformation(0,2) = R(0,2);
    Transformation(1,0) = R(1,0);
    Transformation(1,1) = R(1,1);
    Transformation(1,2) = R(1,2);
    Transformation(2,0) = R(2,0);
    Transformation(2,1) = R(2,1);
    Transformation(2,2) = R(2,2);

    // Set translation
    Transformation(0, 3) = tx;
    Transformation(1, 3) = ty;
    Transformation(2, 3) = tz;
}

void transformationToVector(Eigen::Affine3d transformMatrix, Mat& rvec, Mat& tvec)
{
    Mat R = Mat::zeros(3,3,CV_64F)  ;
    for(int r = 0; r < 3; r++){
        for(int c = 0; c < 3; c++){
            R.at<double>(r,c) = transformMatrix(r,c);
        }
    }
    Rodrigues(R, rvec);
    tvec = Mat::zeros(3,1,CV_64F);
    tvec.at<double>(0,0) = transformMatrix(0,3);
    tvec.at<double>(1,0) = transformMatrix(1,3);
    tvec.at<double>(2,0) = transformMatrix(2,3);
}

void convertTransformMatrixToVector(vector<Eigen::Affine3d> TransformaMatrix,
                               	    vector<Mat>& Rvec,
                                    vector<Mat>& Tvec)
{

    int numTrans = TransformaMatrix.size();
    Rvec.clear();
    Tvec.clear();

    for(int n = 0; n < numTrans; n++){
        Mat rvec;
        Mat tvec;
        transformationToVector(TransformaMatrix[n], rvec, tvec);
        
        Rvec.push_back(rvec);
        Tvec.push_back(tvec);
    }
}

void convertGPScoordinateToTranslation(double lati, double lon, double alt,
                                       double& tx, double& ty, double& tz){
    const static double startLat = 32.77284305159999888701;
    const static double startLon = -117.12791566949999833014;
    const static double er = 6378137.0;

    double scale = std::cos(startLat * M_PI / 180.);
    tx = scale * lon * M_PI * er / 180.0;
    ty = scale * er * std::log(std::tan((90.0 + lati) * M_PI / 360.0));
    tz = alt;

}


void GetTransformationFromRollPitchYawAndTranslation(Eigen::Affine3d& Transformation,
                                                     double roll, double pitch, double yaw, 
                                                     double tx, double ty, double tz){

    double c_roll = std::cos(roll);
    double c_pitch = std::cos(pitch);
    double c_yaw = std::cos(yaw);
    double s_roll = std::sin(roll);
    double s_pitch = std::sin(pitch);
    double s_yaw = std::sin(yaw);

    // Set rotation
    Transformation(0, 0) = c_yaw*c_roll - s_yaw*s_pitch*s_roll;
    Transformation(1, 0) = s_yaw*c_roll + c_yaw*s_pitch*s_roll;
    Transformation(2, 0) = -c_pitch*s_roll;

    Transformation(0, 1) = -s_yaw*c_pitch;
    Transformation(1, 1) = c_yaw*c_pitch;
    Transformation(2, 1) = s_pitch;

    Transformation(0, 2) = c_yaw*s_roll + s_yaw*s_pitch*c_roll;
    Transformation(1, 2) = s_yaw*s_roll - c_yaw*s_pitch*c_roll;
    Transformation(2, 2) = c_pitch*c_roll;

    // Set translation
    Transformation(0, 3) = tx;
    Transformation(1, 3) = ty;
    Transformation(2, 3) = tz;
}

void getAccumulateMotion(const vector<Mat>& Rvec,
                         const vector<Mat>& Tvec,
                         int startIdx, int endIdx,
                         Mat& accumRvec, Mat& accumTvec){
    accumRvec = Mat::zeros(3,1,CV_64F);
    accumTvec = Mat::zeros(3,1,CV_64F);
    
    Eigen::Affine3d accumMatrix = vectorToTransformation(accumRvec, accumTvec);
    for(int n = startIdx; n < endIdx; n++){
        Mat tempRvec, tempTvec;
        tempRvec = Rvec[n].clone();
        tempTvec = Tvec[n].clone();

        Eigen::Affine3d tempMatrix = vectorToTransformation(tempRvec, tempTvec);

        accumMatrix = tempMatrix * accumMatrix;
    }
    Mat R = Mat::zeros(3,3,CV_64F);
    for(int r = 0; r < 3; r++){
        for(int c = 0; c < 3; c++){
            R.at<double>(r,c) = accumMatrix(r,c);
        }
    }
    Rodrigues(R, accumRvec);
    accumTvec.at<double>(0,0) = accumMatrix(0,3);
    accumTvec.at<double>(1,0) = accumMatrix(1,3);
    accumTvec.at<double>(2,0) = accumMatrix(2,3);
}

Eigen::Affine3d vectorToTransformation(Mat rvec, Mat tvec){
	Mat R;
	Eigen::Affine3d result;
	Rodrigues(rvec, R);
	for(int r = 0; r < 3; r++){
		for(int c = 0; c < 3; c++){
			result(r,c) = R.at<double>(r,c);
		}
	}
	result(0, 3) = tvec.at<double>(0, 0);
	result(1, 3) = tvec.at<double>(1, 0);
	result(2, 3) = tvec.at<double>(2, 0);

	return result;
}

vector<Point3f> transformPoints(Eigen::Affine3d trans, vector<Point3f> obj_pts){
	vector<Point3f> result;
	PointCloud<PointXYZ> tempCloud = Point3ftoPointCloud(obj_pts);
	Eigen::Affine3d invTrans = trans.inverse();
	transformPointCloud(tempCloud, tempCloud, invTrans.matrix());
	result = PointCloudtoPoint3f(tempCloud);
	return result;
}

vector<Point3f> PointCloudtoPoint3f(PointCloud<PointXYZ> ptcloud){
    PointXYZ ptxyz;
    Point3f pt;
    vector<Point3f> result;
    for(int i = 0; i < ptcloud.points.size(); i++){
        ptxyz = ptcloud.points[i];
        pt.x = ptxyz.x;
        pt.y = ptxyz.y;
        pt.z = ptxyz.z;

        result.push_back(pt);
    }
    return result;
}
vector<Point3f> PointCloudtoPoint3f(PointCloud<PointXYZI> ptcloud){
    PointXYZI ptxyzi;
    Point3f pt;
    vector<Point3f> result;
    for(int i = 0; i < ptcloud.points.size(); i++){
        ptxyzi = ptcloud.points[i];
        pt.x = ptxyzi.x;
        pt.y = ptxyzi.y;
        pt.z = ptxyzi.z;

        result.push_back(pt);
    }
    return result;
}
PointCloud<PointXYZ> Point3ftoPointCloud(vector<Point3f> pts){
    PointXYZ ptxyz;
    Point3f pt;
    PointCloud<PointXYZ> result;
    for(int i = 0; i < pts.size(); i++){
        pt = pts[i];
        ptxyz.x = pt.x;
        ptxyz.y = pt.y;
        ptxyz.z = pt.z;

        result.points.push_back(ptxyz);
    }
    return result;
}

vector<Point3f> transformRawPointstoCameraCoor(vector<Point3f> RawPoints, Eigen::Affine3d trans){
    PointCloud<PointXYZ> obj_pts = Point3ftoPointCloud(RawPoints);
    transformPointCloud(obj_pts, obj_pts, trans.matrix());

    return PointCloudtoPoint3f(obj_pts);
}

vector<Point3f> transformLiDARptstoRawCoor(vector<Point3f> LiDARpoints){
    Point3f pointbefore, pointafter;
    vector<Point3f> result;

    for(int i = 0; i < LiDARpoints.size(); i++){
        pointbefore = LiDARpoints[i];
        pointafter.x = pointbefore.z;  
        pointafter.y = pointbefore.x;
        pointafter.z = pointbefore.y;
        result.push_back(pointafter);
    }
    return  result;
}
vector<Point3f> readLiDARandConvertToWorldCoor(string PCDPath, 
                                               Mat accumRvec, 
                                               Mat accumTvec){

    Eigen::Affine3d fromCam1toRaw = Eigen::Affine3d::Identity();
    fromCam1toRaw(0,0) = 2.04979302e-02;
    fromCam1toRaw(0,1) = 9.78461317e-03;
    fromCam1toRaw(0,2) = 9.99741993e-01;
    fromCam1toRaw(1,0) = -9.99789535e-01;
    fromCam1toRaw(1,1) = 9.62882439e-04; 
    fromCam1toRaw(1,2) = 2.04895366e-02;
    fromCam1toRaw(2,0) = -7.62148293e-04;
    fromCam1toRaw(2,1) = -9.99951667e-01;
    fromCam1toRaw(2,2) = 9.80219750e-03;
    fromCam1toRaw(0,3) = 7.76330490e-01;
    fromCam1toRaw(1,3) = 5.36925340e-01;
    fromCam1toRaw(2,3) = -1.91739741e-01;

    Eigen::Affine3d fromRawToCam1 = fromCam1toRaw.inverse();

    PointCloud<PointXYZI>::Ptr cloud(new PointCloud<PointXYZI>);
    if (pcl::io::loadPCDFile<pcl::PointXYZI> (PCDPath, *cloud) == -1)
    {
        PCL_ERROR ("Couldn't read file .pcd \n");
        exit;
    }
    vector<Point3f> LiDARpoints = PointCloudtoPoint3f(*cloud);
    LiDARpoints = transformLiDARptstoRawCoor(LiDARpoints);
    LiDARpoints = transformRawPointstoCameraCoor(LiDARpoints, fromRawToCam1);
    Eigen::Affine3d accumTrans =  vectorToTransformation(accumRvec, accumTvec);

    vector<Point3f> result = transformPoints(accumTrans, LiDARpoints);

    return result;
}