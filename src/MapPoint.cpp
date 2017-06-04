#include "MapPoint.h"

using namespace cv;
using namespace std;

MapPoint::MapPoint(Point3f _pos, 
				   unsigned int _frameID, 
				   unsigned int _keypointIdx):
				   pos(_pos), firstVisitFrameID(_frameID),firstVisitKeyPointIdx(_keypointIdx) {}

void MapPoint::addObservation(Frame* frame, unsigned int pointIdx) { 
	observations.insert(pair<Frame*, unsigned int>(frame, pointIdx)); 
}

Point3f MapPoint::getPositionInCameraCoordinate(Mat rvec, Mat tvec){

	Eigen::Affine3d curTrans = vectorToTransformation(rvec, tvec);
	Eigen::Affine3d curTransInv = curTrans.inverse();

	return transformPoint(curTransInv, pos);;
}

void MapPoint::getColor(unsigned char _r, unsigned char _g, unsigned char _b){
	r = _r;
	g = _g;
	b = _b;
}