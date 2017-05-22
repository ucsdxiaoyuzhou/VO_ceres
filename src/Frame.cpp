#include "Frame.h"
#include "utils.h"

using namespace cv;
using namespace std;

Frame::Frame(string leftImgFile, string rightImgFile, 
             STEREO_RECTIFY_PARAMS _srp, int _id, Map* _map):
			 frameID(_id), map(_map), srp(_srp){

    imgL = imread(leftImgFile);
    imgR = imread(rightImgFile);

    if(!imgL.data || ! imgR.data){
        cout << "image does not exist..." << endl;
        exit;
    }

    fx = _srp.P1.at<double>(0,0);
    fy = _srp.P1.at<double>(1,1);
    cx = _srp.P1.at<double>(0,2);
    cy = _srp.P1.at<double>(1,2);
    b = _srp.P2.at<double>(0,3)/fx;

    // Mat imgLgray, imgRgray;
    // cvtColor(imgL, imgLgray, CV_BGR2GRAY);
    // cvtColor(imgR, imgRgray, CV_BGR2GRAY);

    // ORB_SLAM2::ORBextractor* detector = new ORB_SLAM2::ORBextractor(10000,1.1,8,10,5);
    // (*detector)(imgLgray, cv::Mat(), keypointL, despL);
    // (*detector)(imgRgray, cv::Mat(), keypointR, despR);

    SurfFeatureDetector detector(2);
    SurfDescriptorExtractor descriptor;
    
    detector.detect(imgL, keypointL);
    detector.detect(imgR, keypointR);
    
    descriptor.compute(imgL, keypointL, despL);
    descriptor.compute(imgR, keypointR, despR);

    rvec = Mat::zeros(3, 1, CV_64F);
    tvec = Mat::zeros(3, 1, CV_64F);

    worldRvec = Mat::zeros(3, 1, CV_64F);
    worldTvec = Mat::zeros(3, 1, CV_64F);
    // drawFeature(imgL, keypointL, "features");
    vector<KeyPoint> stereoKeypointLeft, stereoKeypointRight;
    vector<DMatch> stereoMatches;

    //compute stereo matches
    matchFeatureKNN(despL, despR, keypointL, keypointR,
    				stereoKeypointLeft, stereoKeypointRight,
    				stereoMatches, 0.8);

    compute3Dpoints(stereoKeypointLeft, stereoKeypointRight,
    				keypointL, keypointR);

    descriptor.compute(imgL, keypointL, despL);

    mappoints = vector<MapPoint*>(keypointL.size(), static_cast<MapPoint*>(NULL));
    originality = vector<bool>(keypointL.size(), false);
//    drawMatch(imgL, keypointL, keypointR, 1, "stereo");
//    waitKey(1000);
}

bool idx_comparator(const DMatch& m1, const DMatch& m2){
	return m1.queryIdx < m2.queryIdx;
}

void Frame::matchFrame(Frame* frame){
    matchesBetweenFrame.clear();
	//first step, match using features
    vector<KeyPoint> matchedPrev, matchedCurr;
    vector<DMatch> matches;
    matchFeatureKNN(despL, frame->despL, 
                    keypointL, frame->keypointL,
                    matchedPrev, matchedCurr,
                    matches, 0.7);

    // obtain obj_pts, img_pts and matchedIdx
    vector<Point3f> obj_pts;
    vector<Point2f> img_pts;

    for(auto m : matches){
        // frame.matchedIdx.push_back(m.queryIdx);
        obj_pts.push_back(scenePts[m.queryIdx]);
        img_pts.push_back(frame->keypointL[m.trainIdx].pt);
    }
    Mat inliers;
    PnP(obj_pts, img_pts, inliers);

    for(int n = 0; n < inliers.rows; n++){
        matchesBetweenFrame.push_back(matches[inliers.at<int>(n,0)]);
    }

    frame->rvec = rvec.clone();
    frame->tvec = tvec.clone();

    sort(matchesBetweenFrame.begin(), matchesBetweenFrame.end(), idx_comparator);

    // for(int n = 0; n < matchesBetweenFrame.size(); n++){
    //  cout << matchesBetweenFrame[n].queryIdx << "  "<< matchesBetweenFrame[n].trainIdx<<endl;
    // }

    vector<Point2f> finalP1, finalP2;
    for(int n = 0; n < matchesBetweenFrame.size(); n++){
        finalP1.push_back(keypointL[matchesBetweenFrame[n].queryIdx].pt);
        finalP2.push_back(frame->keypointL[matchesBetweenFrame[n].trainIdx].pt);
    }

    drawMatch(imgL, finalP1, finalP2, 1, "pnp inliers");
}

void Frame::setWrdTransVectorAndTransScenePts(cv::Mat _worldRvec, cv::Mat _worldTvec) {
    worldRvec = _worldRvec.clone();
    worldTvec = _worldTvec.clone();
    transformScenePtsToWorldCoordinate();
}

void Frame::transformScenePtsToWorldCoordinate(){
	Eigen::Affine3d worldTrans = vectorToTransformation(worldRvec, worldTvec);
	scenePtsinWorld = transformPoints(worldTrans, scenePts);
}

void Frame::manageMapPoints(Frame* frame){    
    int newMappointCount = 0;
    int newObservationCount = 0;
    // cout << "final match size: " << finalMatches.size() << endl;
    for(int n = 0; n < matchesBetweenFrame.size(); n++){

    	unsigned int qIdx = (unsigned int)matchesBetweenFrame[n].queryIdx;
    	unsigned int tIdx = (unsigned int)matchesBetweenFrame[n].trainIdx;
    	// cout << qIdx << "   " << tIdx << endl;
    	if(mappoints[qIdx] == NULL){
    		//create new mappoint in current frame
    		//create a pointer to this mappoint in the matched frame
    		//also need to add an observation
            originality[qIdx] = true; 
    		mappoints[qIdx] = createNewMapPoint(qIdx);
            pointToExistingMapPoint(frame, mappoints[qIdx], tIdx);

    		mappoints[qIdx]->addObservation(frame, tIdx);
    		mappoints[qIdx]->addObservation(this, qIdx);

    		map->addMapPoint(mappoints[qIdx]);
    		newMappointCount++;
    	}
    	else{
    		//add observation to existing mappoint
    		mappoints[qIdx]->addObservation(frame, tIdx);
            pointToExistingMapPoint(frame, mappoints[qIdx], tIdx);
    		newObservationCount++;
    	}
    }
//     cout << "new mappoints number: " << newMappointCount <<endl;
//     cout << "new observation number: " << newObservationCount << endl;
    // cout << "between frame: "<<frameID <<" and frame: " << frame->frameID << endl<<endl;
}


MapPoint* Frame::createNewMapPoint(unsigned int pointIdx){
	MapPoint* ptrMp = new MapPoint(scenePtsinWorld[pointIdx], frameID, pointIdx);
	return ptrMp;
}

void Frame::pointToExistingMapPoint(Frame* frame, MapPoint* mp, unsigned int currIdx){
	frame->mappoints[currIdx] = mp;
}


void Frame::PnP(vector<Point3f> obj_pts, 
                vector<Point2f> img_pts,
                Mat& inliers){

     //solve PnP
    Mat K = srp.P2.colRange(0,3).clone();
    // Mat temprvec, temptvec;
    if(obj_pts.size() == 0){
        cout << "points for PnP is 0, cannot solve." << endl;
        // success = false;
        return;
    }
    solvePnPRansac(obj_pts, img_pts, K, Mat(),
                   rvec, tvec, false, 2000,3.0, 300, inliers);
    // matchedNumWithCurrentFrame = inliers.rows;

    // cout << "PnP inliers: " << matchedNumWithCurrentFrame << endl;
}

void Frame::releaseMemory(){
    imgL.release();
    imgR.release();
    despL.release();
    despR.release();

}



void Frame::matchFeatureKNN(const Mat& desp1, const Mat& desp2, 
                            const vector<KeyPoint>& keypoint1, 
                            const vector<KeyPoint>& keypoint2,
                            vector<KeyPoint>& matchedKeypoint1,
                            vector<KeyPoint>& matchedKeypoint2,
                            vector<DMatch>& matches,
                            double knn_match_ratio){

    matchedKeypoint1.clear();
    matchedKeypoint2.clear();
    matches.clear();

    float imgThres = 0.15 * std::sqrt(std::pow(imgL.rows, 2)+std::pow(imgL.cols, 2));

    cv::Ptr<cv::DescriptorMatcher>  matcher = cv::DescriptorMatcher::create("BruteForce");

    vector< vector<cv::DMatch> > matches_knn;
    matcher->knnMatch( desp1, desp2, matches_knn, 2 );
    vector< cv::DMatch > tMatches;

    for ( size_t i=0; i<matches_knn.size(); i++ )
    {
        if (matches_knn[i][0].distance < knn_match_ratio * matches_knn[i][1].distance )
            tMatches.push_back( matches_knn[i][0] );
    }
    
    if (tMatches.size() <= 20) //too few matches
        return;

    vector<KeyPoint> tMatchedKeypoint1, tMatchedKeypoint2;
    for ( auto m:tMatches )
    {
        Point2f pt1, pt2;
        pt1 = keypoint1[m.queryIdx].pt;
        pt2 = keypoint2[m.trainIdx].pt;
        float ptdist = std::sqrt(std::pow(pt1.x-pt2.x, 2) + std::pow(pt1.y-pt2.y, 2));
        if(ptdist < imgThres){
            matchedKeypoint1.push_back(keypoint1[m.queryIdx]);
            matchedKeypoint2.push_back(keypoint2[m.trainIdx]);
            matches.push_back(m);
        } 
    }  
}

void Frame::compute3Dpoints(vector<KeyPoint>& kl, 
					 	    vector<KeyPoint>& kr,
					 		vector<KeyPoint>& trikl,
					 		vector<KeyPoint>& trikr){

	vector<KeyPoint> copy_kl, copy_kr;
	copy_kl = kl;
	copy_kr = kr;

	scenePts.clear();
	trikl.clear();
	trikr.clear();

	double thres = 60*b;

	for(int n = 0; n < copy_kl.size(); n++){
		Point3f pd;
		double d = fabs(copy_kl[n].pt.x - copy_kr[n].pt.x);
		pd.x = (float)(b*(copy_kl[n].pt.x - cx)/d);
		pd.y = (float)(b*(copy_kl[n].pt.y - cy)/d);
		pd.z = (float)(b*fx/d);
		if(pd.z < thres &&
		   fabs(copy_kl[n].pt.y - copy_kr[n].pt.y) < 10){
			scenePts.push_back(pd);
			trikl.push_back(copy_kl[n]);
			trikr.push_back(copy_kr[n]);
		}
	}

}