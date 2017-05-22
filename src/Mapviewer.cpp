#include "Mapviewer.h"

using namespace pcl;
using namespace std;

Mapviewer::Mapviewer(){
	// viewer = new pcl::visualization::CloudViewer(n);
	// cloud = new pcl::PointCloud<pcl::PointXYZ>();
}


pcl::PointCloud<pcl::PointXYZRGB> Mapviewer::pointToPointCloud(std::vector<cv::Point3f> scenePts, 
														    int R, int G, int B){
	int ptsNum = scenePts.size();
	PointCloud<PointXYZRGB> result;
	for(int n = 0; n < ptsNum; n++){
		PointXYZRGB pt;
		pt.x = scenePts[n].x;
		pt.y = scenePts[n].y;
		pt.z = scenePts[n].z;
		pt.r = R;
		pt.g = G;
		pt.b = B;
		result.points.push_back(pt);
	}

	result.height = 1;
	result.width = result.points.size();

	return result;
}


void Mapviewer::jointToMap(PointCloud<PointXYZRGB> frameMap, Eigen::Affine3d& trans){

	if(false == initialized){
		initialized = true;
		cout << "map initializing! " << endl;
		entireMap = frameMap;
		cout << "map initialized!" << endl;
	}
	entireMap = frameMap;


}

void Mapviewer::addMorePoints(PointCloud<PointXYZRGB> frameMap, Eigen::Affine3d& trans, bool downsample){
	if(false == initialized){
		initialized = true;
		cout << "map initializing! " << endl;
		entireMap = frameMap;
		cout << "map initialized!" << endl;
	}
	if(downsample == true){
		PointCloud<PointXYZRGB>::Ptr cloud  (new PointCloud<PointXYZRGB>);
		*cloud = frameMap;
		pcl::VoxelGrid<pcl::PointXYZRGB> sor;
        sor.setInputCloud (cloud);
        sor.setLeafSize (0.3f, 0.3f, 0.3f);
        sor.filter (*cloud);
        frameMap = *cloud;
	}
	entireMap += frameMap;
}