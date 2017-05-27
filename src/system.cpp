#include "system.h"

using namespace std;
using namespace pcl;
using namespace cv;

void SLAMsystem(string commonPath, string yamlPath){

    STEREO_RECTIFY_PARAMS srp; // used to store
	//================obtain file name================================
	string leftImgFilePath, rightImgFilePath;
    leftImgFilePath = commonPath + "/image_0";
    rightImgFilePath = commonPath + "/image_1";

    vector<string> leftImgName = getImgFileName(leftImgFilePath);
    vector<string> rightImgName = getImgFileName(rightImgFilePath);

    //================read settings===================================
    cv::FileStorage fsSettings(yamlPath, cv::FileStorage::READ);
    if(!fsSettings.isOpened()){
        cerr << "ERROR: Wrong path to settings" << endl;
        return;
    }
    string savePath = fsSettings["trajectory file"];
    fsSettings["LEFT.P"] >> srp.P1;
    fsSettings["RIGHT.P"] >> srp.P2;

    srp.imageSize.height = fsSettings["height"];
    srp.imageSize.width = fsSettings["width"];
    //================threshold======================================
    int localTimes = fsSettings["local optimization times"];
    int dataLength = fsSettings["data length"];
    int startIdx = fsSettings["start index"];

    //================set up data length ============================
    if(dataLength == 0){
        dataLength = (int)leftImgName.size();
    }
    else{
        vector<string>::const_iterator first = leftImgName.begin()+ startIdx;
        vector<string>::const_iterator last = leftImgName.begin()+ startIdx + dataLength;
        vector<string> tempLeft(first, last);
        leftImgName = tempLeft;

        first = rightImgName.begin()+ startIdx;
        last = rightImgName.begin() + startIdx+ dataLength;
        vector<string> tempRight(first, last);
        rightImgName = tempRight;
    }
    if(srp.P1.empty() || srp.P2.empty()){
        cerr << "ERROR: Calibration parameters to rectify stereo are missing!" << endl;
        return;
    }

    //=============== initialize system ============================================
    vector<Mat> relativeRvec_cam, relativeTvec_cam;
    Map map;
    Mapviewer mapviewer;
    vector<Frame* > allFrame;

    int deleteFrame = -20;

    Problem globalBAProblem;
    Optimizer ba(true, globalBAProblem);

    pcl::visualization::CloudViewer viewer("Cloud Viewer");
    PointCloud<PointXYZRGB>::Ptr cloud (new PointCloud<PointXYZRGB>);

    Mat accumRvec = Mat::zeros(3,1,CV_64F);
    Mat accumTvec = Mat::zeros(3,1,CV_64F);

    Frame* prevFrame = new Frame(leftImgName[0], rightImgName[0], srp, 0, &map);
    prevFrame->scenePtsinWorld = prevFrame->scenePts;

    Frame* refFrame = new Frame(leftImgName[0], rightImgName[0], srp, 0, &map);
    refFrame = prevFrame;

    allFrame.push_back(prevFrame);

    for(int n = 1; n < dataLength; n++){
        cout << "current frame number: " << n << endl;
        Frame* currFrame = new Frame(leftImgName[n], rightImgName[n], srp, n, &map);
        prevFrame->matchFrame(currFrame);
        //=================== accumlate motion ====================
        relativeRvec_cam.push_back(currFrame->rvec);
        relativeTvec_cam.push_back(currFrame->tvec);

        getAccumulateMotion(relativeRvec_cam, relativeTvec_cam,
                            0, n, accumRvec, accumTvec);

        currFrame->setWrdTransVectorAndTransScenePts(accumRvec, accumTvec);
        prevFrame->manageMapPoints(currFrame);
        allFrame.push_back(currFrame);
        //=================== perform more matching ====================
        for(int t = localTimes-1; t > 0; t--){
//            cout <<"matching from: " << MAX(0, (int)allFrame.size() - localTimes-1) << " to: " << MAX(0, n-t)  << endl;
//            allFrame[MAX(0, n-t)]->matchFrame(currFrame);
//            allFrame[MAX(0, n-t)]->manageMapPoints(currFrame);
            refFrame->matchFrame(allFrame[MAX(0, n-t)]);
            refFrame->manageMapPoints(allFrame[MAX(0, n-t)]);
        }
        ba.localBundleAdjustment(allFrame, MAX(0, (int)allFrame.size()-localTimes), localTimes);
        allFrame[MAX(0, (int)allFrame.size()-localTimes)]->judgeBadPoints();

        //================= test: count bad point number =================
//        int badPointNum = 0;
//        for(auto mappoint : map.allMapPoints){
//            if(mappoint->isBad){badPointNum++;}
//        }
//        cout << "bad point number: " << badPointNum << endl;
        //=================== show result ================================
//        Mat showTvec, showRvec;
//        getAccumulateMotion(relativeRvec_cam, relativeTvec_cam,
//                            0, MAX(0, n - localTimes),
//                            accumRvec, accumTvec);
//        cout <<endl<<endl<< n<<"  "<<accumTvec.at<double>(0,0)<<"  "<<accumTvec.at<double>(1,0)<<"  "<<accumTvec.at<double>(2,0)<<endl;
//
//        showRvec = allFrame[MAX(0, n - localTimes)]->worldRvec.clone();
//        showTvec = allFrame[MAX(0, n - localTimes)]->worldTvec.clone();
//        cout <<"  "<<showTvec.at<double>(0,0)<<"  "<<showTvec.at<double>(1,0)<<"  "<<showTvec.at<double>(2,0)<<endl<<endl;
        //============== update accumulative transformation ================
        for(int n = MAX(0, (int)allFrame.size()-localTimes)+1; n < allFrame.size(); n++){

            getRelativeMotion(allFrame[n-1]->worldRvec, allFrame[n-1]->worldTvec,
                              allFrame[n]->worldRvec,   allFrame[n]->worldTvec,
                              relativeRvec_cam[n-1],
                              relativeTvec_cam[n-1]);
        }
        Mat tempAccumRvec, tempAccumTvec;
        for(int n = MAX(0, (int)allFrame.size()-localTimes); n < allFrame.size(); n++) {

            getAccumulateMotion(relativeRvec_cam, relativeTvec_cam,
                                0, n, tempAccumRvec, tempAccumTvec);
            allFrame[n]->setWrdTransVectorAndTransScenePts(tempAccumRvec,tempAccumTvec);
        }

        //=============draw map=============================================
        Eigen::Affine3d curTrans =  Eigen::Affine3d::Identity();
        Eigen::Affine3d curCam = vectorToTransformation(tempAccumRvec, tempAccumTvec);
        vector<Point3f> allPoints = map.getAllMapPoints();

        mapviewer.addCamera(curCam);
        mapviewer.jointToMap(mapviewer.pointToPointCloud(allPoints), curTrans);

        *cloud = mapviewer.entireMap;

        viewer.showCloud(cloud);
        if(waitKey(5) == 27){};

        refFrame = allFrame[MAX(0, (int)allFrame.size()-localTimes-1 )];

        prevFrame = allFrame.back();

        deleteFrame++;
        if(deleteFrame > 0) {
            allFrame[deleteFrame]->releaseMemory();
        }
    }

//============== end of main loop =========================================================================


    ofstream outfile;
    outfile.open("../beforeBA.txt");
    //evaluate before global BA
    for(int n = 0; n < dataLength; n++){
        // Mat accumRvec, accumTvec;
        getAccumulateMotion(relativeRvec_cam, relativeTvec_cam,
                            0, n,
                            accumRvec, accumTvec);

        double diff1 = allFrame[n]->worldRvec.at<double>(0,0) - accumRvec.at<double>(0,0);
        cout << setprecision(15) <<diff1/accumRvec.at<double>(0,0) * 100.0 <<"\%" <<"  ";

        double diff2 = allFrame[n]->worldRvec.at<double>(1,0) - accumRvec.at<double>(1,0);
        cout << setprecision(15)<< diff2/accumRvec.at<double>(1,0) * 100.0 <<"\%" <<"  ";

        double diff3 = allFrame[n]->worldRvec.at<double>(2,0) - accumRvec.at<double>(2,0);
        cout << setprecision(15)<< diff3/accumRvec.at<double>(2,0) * 100.0 <<"\%" <<"  ";

        double diffx = allFrame[n]->worldTvec.at<double>(0,0) - accumTvec.at<double>(0,0);
        cout << setprecision(15)<< diffx/accumTvec.at<double>(0,0) * 100.0 <<"\%" <<"  ";

        double diffy = allFrame[n]->worldTvec.at<double>(1,0) - accumTvec.at<double>(1,0);
        cout << setprecision(15)<< diffy/accumTvec.at<double>(1,0) * 100.0 <<"\%" <<"  ";

        double diffz = allFrame[n]->worldTvec.at<double>(2,0) - accumTvec.at<double>(2,0);
        cout << setprecision(15)<< diffz/accumTvec.at<double>(2,0) * 100.0 <<"\%" <<endl;

        Eigen::Affine3d curPose = vectorToTransformation(accumRvec, accumTvec);
        Eigen::Affine3d invPose = curPose.inverse();

        for(int r = 0; r < 3; r++){
            for(int c = 0; c < 4; c++){
                outfile << invPose(r,c) << " ";
            }
        }
        outfile << endl;
    }
    outfile.close();

    outfile.open(savePath);
    ba.globalBundleAdjustment(&map, allFrame);
    //============== update accumulative transformation ================
    for(int n = 1; n < allFrame.size(); n++){

        getRelativeMotion(allFrame[n-1]->worldRvec, allFrame[n-1]->worldTvec,
                          allFrame[n]->worldRvec,   allFrame[n]->worldTvec,
                          relativeRvec_cam[n-1],
                          relativeTvec_cam[n-1]);
    }
    Mat tempAccumRvec, tempAccumTvec;
    for(int n = 0; n < allFrame.size(); n++) {

        getAccumulateMotion(relativeRvec_cam, relativeTvec_cam,
                            0, n, tempAccumRvec, tempAccumTvec);
        allFrame[n]->setWrdTransVectorAndTransScenePts(tempAccumRvec,tempAccumTvec);
    }
    //evaluate after global BA
    for(int n = 0; n < dataLength; n++){
        // Mat accumRvec, accumTvec;
        getAccumulateMotion(relativeRvec_cam, relativeTvec_cam,
                            0, n,
                            accumRvec, accumTvec);

        Eigen::Affine3d curPose = vectorToTransformation(accumRvec, accumTvec);
        Eigen::Affine3d invPose = curPose.inverse();

        mapviewer.addCamera(curPose);

        for(int r = 0; r < 3; r++){
            for(int c = 0; c < 4; c++){
                outfile << invPose(r,c) << " ";
            }
        }
        outfile << endl;
    }
    outfile.close();

    //=========== draw points after BA ================================
    long unsigned int totalNumPoints = map.allMapPointNumber();
    // convert to Point3f
    Eigen::Affine3d curTrans =  Eigen::Affine3d::Identity();
    
    vector<Point3f> afterBAPoints3f;

    for(set<MapPoint*>::iterator pIt = map.allMapPoints.begin();
                                 pIt!= map.allMapPoints.end();
                                 pIt++){
        if((*pIt)->isBad == false){
            Point3f pt;
            pt.x = (*pIt)->pos.x;
            pt.y = (*pIt)->pos.y;
            pt.z = (*pIt)->pos.z;

            afterBAPoints3f.push_back(pt);
        }
    }
    
    mapviewer.addMorePoints(mapviewer.pointToPointCloud(afterBAPoints3f,0,255,0), curTrans);
    *cloud = mapviewer.entireMap;
    viewer.showCloud(cloud);
    if(waitKey(5) == 27){};

    waitKey(0);
}