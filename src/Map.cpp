#include "Map.h"

Map::Map():pointIdx(0){
}

void Map::addMapPoint(MapPoint* mappoint){
	allMapPoints.insert(mappoint);
	pointIdx++;
}

std::vector<cv::Point3f> Map::getAllMapPoints(){
	std::vector<cv::Point3f> result;

	for(auto mp : allMapPoints){
		if(!mp->isBad){ result.push_back(mp->pos); }
	}

	return result;
}

void Map::getAllColors(vector<unsigned char>& R, 
					   vector<unsigned char>& G,
					   vector<unsigned char>& B){
	R.clear();
	G.clear();
	B.clear();

	for(auto mp : allMapPoints){
		if(!mp->isBad){ 
			R.push_back(mp->r);
			G.push_back(mp->g);
			B.push_back(mp->b);
		 }
	}

}

long unsigned int Map::allMapPointNumber(){
	return (long unsigned int)allMapPoints.size();
}




