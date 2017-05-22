#ifndef MAPPOINT_H
#define MAPPOINT_H

#include <opencv2/core/core.hpp>
#include <vector>
#include <set>
#include <map>
#include <stdio.h>

#include "Frame.h" 

class Frame;

class MapPoint{

public:
	MapPoint(cv::Point3f _pos, 
			 unsigned int _frameID, 
			 unsigned int _keypointIdx);
	void addObservation(Frame* frame, unsigned int pointIdx);

public:
	cv::Point3f pos; //pose in the first visited frame, in world coordinate
	unsigned int firstVisitFrameID;
	unsigned int firstVisitKeyPointIdx;
	bool isBad = false;

	std::map<Frame*, unsigned int> observations;


};

#endif