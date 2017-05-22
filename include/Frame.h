#ifndef FRAME_H
#define FRAME_H

#include "MapPoint.h"
#include "Map.h"
#include "utils.h"

#include <vector>
#include <stdio.h>

#include "opencv2/core/core.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include <opencv2/features2d/features2d.hpp>


class Map;
class MapPoint;
class Frame{
public:
	cv::Mat imgL, imgR;
	cv::Mat despL, despR;

	cv::Mat rvec, tvec;
	cv::Mat worldRvec, worldTvec;

	std::vector<cv::KeyPoint> keypointL, keypointR;

	std::vector<cv::Point3f> scenePts;
	std::vector<cv::Point3f> scenePtsinWorld;
	
	std::vector<cv::DMatch> matchesBetweenFrame;

	unsigned int frameID;

	STEREO_RECTIFY_PARAMS srp;
	cv::Mat K;
	double fx;
	double fy;
	double cx;
	double cy;
	double b;

	std::vector<MapPoint*> mappoints; // 3d points in world coordinate
	std::vector<bool> originality;
	Map* map;


	Frame(string leftImgFile, string rightImgFile,
		  STEREO_RECTIFY_PARAMS _srp, int _id, Map* _map);

	void setWrdTransVectorAndTransScenePts(cv::Mat _worldRvec, cv::Mat _worldTvec);

	void matchFrame(Frame* frame);
	void manageMapPoints(Frame* frame);

	void transformScenePtsToWorldCoordinate(Eigen::Affine3d accumTrans);
	void transformScenePtsToWorldCoordinate();
    
    void matchFeatureKNN(const cv::Mat& desp1, const cv::Mat& desp2, 
                            const std::vector<cv::KeyPoint>& keypoint1, 
                            const std::vector<cv::KeyPoint>& keypoint2,
                            std::vector<cv::KeyPoint>& matchedKeypoint1,
                            std::vector<cv::KeyPoint>& matchedKeypoint2,
                            std::vector<cv::DMatch>& matches,
                            double knn_match_ratio = 0.8);

    void compute3Dpoints(std::vector<cv::KeyPoint>& kl, 
					 	 std::vector<cv::KeyPoint>& kr,
					 	 std::vector<cv::KeyPoint>& trikl,
					 	 std::vector<cv::KeyPoint>& trikr);

	void PnP(std::vector<cv::Point3f> obj_pts, 
	                std::vector<cv::Point2f> img_pts,
	                cv::Mat& inliers);

	MapPoint* createNewMapPoint(unsigned int pointIdx);
	void pointToExistingMapPoint(Frame* frame, MapPoint* mp, unsigned int currIdx);
	


    void releaseMemory();
};
	

#endif