#include "optimizer.hpp"

Optimizer::Optimizer(bool printOut, Problem& prob):globalBAProblem(prob){
    // BAProblem = problem;
    options.linear_solver_type = SPARSE_NORMAL_CHOLESKY;
    options.minimizer_progress_to_stdout = printOut;
    options.max_num_iterations = 5;
}

void Optimizer::globalBundleAdjustment(Map* slammap, vector<Frame*> frames){
    // set camera poses
    double* cameraParameter_ = new double[6*frames.size()];
    cout << "frame size: " << frames.size() << endl;
    for(int n = 0; n < frames.size(); n++){
        cameraParameter_[cameraBlkSize*n + 0] = frames[n]->worldRvec.at<double>(0,0);
        cameraParameter_[cameraBlkSize*n + 1] = frames[n]->worldRvec.at<double>(1,0);
        cameraParameter_[cameraBlkSize*n + 2] = frames[n]->worldRvec.at<double>(2,0);

        cameraParameter_[cameraBlkSize*n + 3] = frames[n]->worldTvec.at<double>(0,0);
        cameraParameter_[cameraBlkSize*n + 4] = frames[n]->worldTvec.at<double>(1,0);
        cameraParameter_[cameraBlkSize*n + 5] = frames[n]->worldTvec.at<double>(2,0);
    }

    long unsigned int numPointsParameters = pointBlkSize*slammap->allMapPointNumber();
    double* pointParameters_;
    pointParameters_ = new double[numPointsParameters];
    //convert container of mappoints to vector
    vector<MapPoint*> vallMapPoints;
    //add all map points into parameter
    cout << "adding mappoints into problem...";
    int n = 0;
    for(set<MapPoint*>::iterator it = slammap->allMapPoints.begin(); 
                                 it!= slammap->allMapPoints.end(); 
                                 it++){
        vallMapPoints.push_back(*it);

        pointParameters_[pointBlkSize*n + 0] = (double)(*it)->pos.x;
        pointParameters_[pointBlkSize*n + 1] = (double)(*it)->pos.y;
        pointParameters_[pointBlkSize*n + 2] = (double)(*it)->pos.z;

        n++;
    }
    cout <<" done!" << endl;
    double* points = pointParameters_;
    double* cameras = cameraParameter_;
    double* point = points;
    double* camera = cameras;

    //add mappoints
    cout << "adding observations into problem...";
    long unsigned int count = 0;//used to count mappoints
    for(vector<MapPoint*>::iterator it = vallMapPoints.begin();
                                    it!= vallMapPoints.end();
                                    it++){
        //iterate all observations of a mappoint
        for(map<Frame*, unsigned int>::iterator pIt = (*it)->observations.begin();
                                                pIt!= (*it)->observations.end();
                                                pIt++){
            Point2f observedPt;
            observedPt.x = pIt->first->keypointL[pIt->second].pt.x;
            observedPt.y = pIt->first->keypointL[pIt->second].pt.y;
            camera = cameras + cameraBlkSize*pIt->first->frameID;
            point = points + pointBlkSize*count;
            //if first camera, fix
            if(pIt->first->frameID == 0){
                CostFunction* costFunc = SnavelyReprojectionOnlyError::Create((double)observedPt.x, 
                                                                              (double)observedPt.y,
                                                                               pIt->first->fx,
                                                                               pIt->first->cx,
                                                                               pIt->first->cy,
                                                                               camera);
                globalBAProblem.AddResidualBlock(costFunc, NULL, point);
            }
            //else do not fix
            else{
                CostFunction* costFunc = SnavelyReprojectionError::Create((double)observedPt.x, 
                                                                          (double)observedPt.y,
                                                                          pIt->first->fx,
                                                                          pIt->first->cx,
                                                                          pIt->first->cy);
                globalBAProblem.AddResidualBlock(costFunc, NULL, camera, point);
            }
        }
        count++;
    }
    cout << "  done!" << endl;
    cout << "solving problem..." <<endl;
    Solve(options, &globalBAProblem, &summary);

    //update mappoints;
    n = 0;
    for(set<MapPoint*>::iterator it = slammap->allMapPoints.begin(); 
                                 it!= slammap->allMapPoints.end(); 
                                 it++){
        (*it)->pos.x = (float)pointParameters_[pointBlkSize*n + 0];
        (*it)->pos.y = (float)pointParameters_[pointBlkSize*n + 1];
        (*it)->pos.z = (float)pointParameters_[pointBlkSize*n + 2];
        if((*it)->pos.z > 200){(*it)->isBad = true;}

        n++;
    }
    for(int n = 0; n < frames.size(); n++){
        frames[n]->worldRvec.at<double>(0,0) = cameraParameter_[cameraBlkSize*n + 0];
        frames[n]->worldRvec.at<double>(1,0) = cameraParameter_[cameraBlkSize*n + 1];
        frames[n]->worldRvec.at<double>(2,0) = cameraParameter_[cameraBlkSize*n + 2];
        frames[n]->worldTvec.at<double>(0,0) = cameraParameter_[cameraBlkSize*n + 3];
        frames[n]->worldTvec.at<double>(1,0) = cameraParameter_[cameraBlkSize*n + 4];
        frames[n]->worldTvec.at<double>(2,0) = cameraParameter_[cameraBlkSize*n + 5];
    }

}


void Optimizer::localBundleAdjustment(vector<Frame*> frames, int startIdx, int length){
    //construct camera parameters
    double* cameraParameter_ = new double[6*frames.size()];
    cout << "frame size: " << frames.size() << endl;
    for(int n = 0; n < frames.size(); n++){
        cameraParameter_[cameraBlkSize*n + 0] = frames[n]->worldRvec.at<double>(0,0);
        cameraParameter_[cameraBlkSize*n + 1] = frames[n]->worldRvec.at<double>(1,0);
        cameraParameter_[cameraBlkSize*n + 2] = frames[n]->worldRvec.at<double>(2,0);

        cameraParameter_[cameraBlkSize*n + 3] = frames[n]->worldTvec.at<double>(0,0);
        cameraParameter_[cameraBlkSize*n + 4] = frames[n]->worldTvec.at<double>(1,0);
        cameraParameter_[cameraBlkSize*n + 5] = frames[n]->worldTvec.at<double>(2,0);
    }

    // build problem
    Problem localProblem;

    vector<MapPoint*> localMapPoints;
    vector<Point3f> localScenePoints;
    vector<int> localScenePointsIdx2;
    vector<int> localScenePointsIdx3;

    vector<Point2f> localKeypoints;
    vector<Point2f> localKeypoints2;
    vector<Point2f> localKeypoints3;

    double* cameras = cameraParameter_;
    double* camera = cameras;

    int offset = 0;
    //first iteration to create vector of point parameters
    for(int p = 0; p < frames[startIdx]->keypointL.size(); p++){
        //set all mappoints in the first frame as parameter potins, temporarily store into a vector
        if(frames[startIdx]->mappoints[p] != NULL){
            localMapPoints.push_back(frames[startIdx]->mappoints[p]);
            localScenePoints.push_back(frames[startIdx]->mappoints[p]->pos);
        }
    }

    for(int n = 0; n < frames[MAX(0,MIN(startIdx+offset,frames.size()-1))]->keypointL.size(); n++){
        if(frames[MAX(0,MIN(startIdx+offset,frames.size()-1))]->mappoints[n] != NULL){
            localKeypoints.push_back(frames[MAX(0,MIN(startIdx+offset,frames.size()-1))]->keypointL[n].pt);
        }
    }

    //create point parameters
    long unsigned int numPointsParameters = pointBlkSize*localMapPoints.size();
    double* pointParameters_;
    pointParameters_ = new double[numPointsParameters];

    //add all mappoints into problem
    for(int n = 0; n < localMapPoints.size(); n++){
        pointParameters_[pointBlkSize*n + 0] = (double)localMapPoints[n]->pos.x;
        pointParameters_[pointBlkSize*n + 1] = (double)localMapPoints[n]->pos.y;
        pointParameters_[pointBlkSize*n + 2] = (double)localMapPoints[n]->pos.z;
    }

    //add observations
    unsigned int minFrameID = frames[startIdx]->frameID;
    unsigned int maxFrameID = frames[MIN(startIdx+length-1, frames.size()-1)]->frameID;
    double* points = pointParameters_;
    double* point = points;
    cout << "min id: " << minFrameID<<endl;
    cout << "max id: " << maxFrameID<<endl;
    //====test==============================================================================
    Mat Rvec = Mat::zeros(3,1,CV_64F);
    Mat Tvec = Mat::zeros(3,1,CV_64F);  

    Rvec.at<double>(0,0) = cameraParameter_[cameraBlkSize*(MAX(0,startIdx+offset)) + 0];
    Rvec.at<double>(1,0) = cameraParameter_[cameraBlkSize*(MAX(0,startIdx+offset)) + 1];
    Rvec.at<double>(2,0) = cameraParameter_[cameraBlkSize*(MAX(0,startIdx+offset)) + 2];

    Tvec.at<double>(0,0) = cameraParameter_[cameraBlkSize*(MAX(0,startIdx+offset)) + 3];
    Tvec.at<double>(1,0) = cameraParameter_[cameraBlkSize*(MAX(0,startIdx+offset)) + 4];
    Tvec.at<double>(2,0) = cameraParameter_[cameraBlkSize*(MAX(0,startIdx+offset)) + 5];

    vector<Point2f> projImgPts;
    Mat K = frames[0]->srp.P1.colRange(0,3).clone();
    projectPoints(localScenePoints, Rvec, Tvec, K, Mat(), projImgPts);
    drawFarandCloseFeatures(frames[MIN(startIdx+offset,frames.size()-1)]->imgL, 
                            localKeypoints, 
                            projImgPts,"before");
    //====test end===========================================================================
    vector<Point2f> observedPts;

    for(int n = 0; n < localMapPoints.size(); n++){
        for(map<Frame*, unsigned int>::iterator pIt = localMapPoints[n]->observations.begin();
                                                pIt!= localMapPoints[n]->observations.end();
                                                pIt++){
            Point2f observedPt;

            observedPt.x = pIt->first->keypointL[pIt->second].pt.x;
            observedPt.y = pIt->first->keypointL[pIt->second].pt.y;

            camera = cameras + cameraBlkSize*pIt->first->frameID;
            point = points + pointBlkSize*n;

            //only fix the first frame in the entire sequence
            if(pIt->first->frameID == 0){
                CostFunction* costFunc = SnavelyReprojectionOnlyError::Create((double)observedPt.x, 
                                                                              (double)observedPt.y,
                                                                               pIt->first->fx,
                                                                               pIt->first->cx,
                                                                               pIt->first->cy,
                                                                               camera);
                localProblem.AddResidualBlock(costFunc, new HuberLoss(2), point);

            }
            //else do not fix
            else if(pIt->first->frameID >= minFrameID &&
                    pIt->first->frameID <= maxFrameID){
                // ==== this part is to extract keypoint in the corresponding frame
                // ==== only for evaluation
                if(pIt->first->frameID == minFrameID + 1){
                    localScenePointsIdx2.push_back(n);
                    localKeypoints2.push_back(observedPt);
                }
                if(pIt->first->frameID == minFrameID + 2){
                    localScenePointsIdx3.push_back(n);
                    localKeypoints3.push_back(observedPt);
                }

                CostFunction* costFunc = SnavelyReprojectionError::Create((double)observedPt.x, 
                                                                          (double)observedPt.y,
                                                                          pIt->first->fx,
                                                                          pIt->first->cx,
                                                                          pIt->first->cy);
                localProblem.AddResidualBlock(costFunc, new HuberLoss(2), camera, point);
            }
        }
    }
    Solve(options, &localProblem, &summary);

    if(options.minimizer_progress_to_stdout){
        cout << summary.FullReport() << endl;
    }

    //update points and pose
    unsigned int pointCount = 0;
    for(int p = 0; p < frames[startIdx]->keypointL.size(); p++){
         if(frames[startIdx]->mappoints[p] != NULL){
            frames[startIdx]->mappoints[p]->pos.x = (float)pointParameters_[pointBlkSize*pointCount + 0];
            frames[startIdx]->mappoints[p]->pos.y = (float)pointParameters_[pointBlkSize*pointCount + 1];
            frames[startIdx]->mappoints[p]->pos.z = (float)pointParameters_[pointBlkSize*pointCount + 2];

            if(frames[startIdx]->mappoints[p]->pos.z > 300){frames[startIdx]->mappoints[p]->isBad = true;}

            pointCount++;
        }
    }
    for(int n = 0; n < frames.size(); n++){
        frames[n]->worldRvec.at<double>(0,0) = cameraParameter_[cameraBlkSize*n + 0];
        frames[n]->worldRvec.at<double>(1,0) = cameraParameter_[cameraBlkSize*n + 1];
        frames[n]->worldRvec.at<double>(2,0) = cameraParameter_[cameraBlkSize*n + 2];
        frames[n]->worldTvec.at<double>(0,0) = cameraParameter_[cameraBlkSize*n + 3];
        frames[n]->worldTvec.at<double>(1,0) = cameraParameter_[cameraBlkSize*n + 4];
        frames[n]->worldTvec.at<double>(2,0) = cameraParameter_[cameraBlkSize*n + 5];
    }

    
//================================== display for evaluation ========================================
    Rvec.at<double>(0,0) = cameras[cameraBlkSize*(startIdx+offset) + 0];
    Rvec.at<double>(1,0) = cameras[cameraBlkSize*(startIdx+offset) + 1];
    Rvec.at<double>(2,0) = cameras[cameraBlkSize*(startIdx+offset) + 2];

    Tvec.at<double>(0,0) = cameras[cameraBlkSize*(startIdx+offset) + 3];
    Tvec.at<double>(1,0) = cameras[cameraBlkSize*(startIdx+offset) + 4];
    Tvec.at<double>(2,0) = cameras[cameraBlkSize*(startIdx+offset) + 5];

    vector<Point3f> adjustedPts;
    for(int n = 0; n < localMapPoints.size(); n++){
        Point3f pt;        

        pt.x = (float)pointParameters_[pointBlkSize*n + 0];
        pt.y = (float)pointParameters_[pointBlkSize*n + 1];
        pt.z = (float)pointParameters_[pointBlkSize*n + 2];
        
        adjustedPts.push_back(pt);
    }
    vector<Point2f> newProjPts;
    projectPoints(adjustedPts, Rvec, Tvec, K, Mat(), newProjPts);

    float totalErr = 0;
    for(int n = 0; n < newProjPts.size(); n++){
        float dx = newProjPts[n].x - localKeypoints[n].x;
        float dy = newProjPts[n].y - localKeypoints[n].y;
        float tempErr = sqrt(dx*dx + dy*dy);
        totalErr += tempErr;

    }
//    cout << "total error: " << totalErr << endl;
    drawFarandCloseFeatures(frames[MIN(startIdx+(unsigned int long)offset,frames.size()-1)]->imgL,
                            localKeypoints, 
                            newProjPts,"after");

    //======================== 1 =============================
    offset = 1;

    Rvec.at<double>(0,0) = cameras[cameraBlkSize*(startIdx+offset) + 0];
    Rvec.at<double>(1,0) = cameras[cameraBlkSize*(startIdx+offset) + 1];
    Rvec.at<double>(2,0) = cameras[cameraBlkSize*(startIdx+offset) + 2];

    Tvec.at<double>(0,0) = cameras[cameraBlkSize*(startIdx+offset) + 3];
    Tvec.at<double>(1,0) = cameras[cameraBlkSize*(startIdx+offset) + 4];
    Tvec.at<double>(2,0) = cameras[cameraBlkSize*(startIdx+offset) + 5];

    adjustedPts.clear();
    for(int n = 0; n < localScenePointsIdx2.size(); n++){
        Point3f pt;
        pt.x = (float)pointParameters_[pointBlkSize*localScenePointsIdx2[n] + 0];
        pt.y = (float)pointParameters_[pointBlkSize*localScenePointsIdx2[n] + 1];
        pt.z = (float)pointParameters_[pointBlkSize*localScenePointsIdx2[n] + 2];
        
        adjustedPts.push_back(pt);
    }
    newProjPts.clear();
    projectPoints(adjustedPts, Rvec, Tvec, K, Mat(), newProjPts);

    totalErr = 0;
    for(int n = 0; n < newProjPts.size(); n++){
        float dx = newProjPts[n].x - localKeypoints2[n].x;
        float dy = newProjPts[n].y - localKeypoints2[n].y;
        float tempErr = sqrt(dx*dx + dy*dy);
        totalErr += tempErr;

    }
//    cout << "total error: " << totalErr << endl;
    if(localKeypoints2.size() > 0){
        drawFarandCloseFeatures(frames[MIN(startIdx+offset,frames.size()-1)]->imgL, 
                                localKeypoints2, 
                                newProjPts,"1");
    }

    //======================== 2 =============================
    offset=2;

    Rvec.at<double>(0,0) = cameras[cameraBlkSize*(startIdx+offset) + 0];
    Rvec.at<double>(1,0) = cameras[cameraBlkSize*(startIdx+offset) + 1];
    Rvec.at<double>(2,0) = cameras[cameraBlkSize*(startIdx+offset) + 2];

    Tvec.at<double>(0,0) = cameras[cameraBlkSize*(startIdx+offset) + 3];
    Tvec.at<double>(1,0) = cameras[cameraBlkSize*(startIdx+offset) + 4];
    Tvec.at<double>(2,0) = cameras[cameraBlkSize*(startIdx+offset) + 5];

    adjustedPts.clear();

    for(int n = 0; n < localScenePointsIdx3.size(); n++){
        Point3f pt;

        pt.x = (float)pointParameters_[pointBlkSize*localScenePointsIdx3[n] + 0];
        pt.y = (float)pointParameters_[pointBlkSize*localScenePointsIdx3[n] + 1];
        pt.z = (float)pointParameters_[pointBlkSize*localScenePointsIdx3[n] + 2];
        
        adjustedPts.push_back(pt);
    }
    newProjPts.clear();
    if(adjustedPts.size() > 0){
        projectPoints(adjustedPts, Rvec, Tvec, K, Mat(), newProjPts);
    
        totalErr = 0;
        for(int n = 0; n < newProjPts.size(); n++){
            float dx = newProjPts[n].x - localKeypoints3[n].x;
            float dy = newProjPts[n].y - localKeypoints3[n].y;
            float tempErr = sqrt(dx*dx + dy*dy);
            totalErr += tempErr;

        }

//        cout << "total error: " << totalErr << endl;
        drawFarandCloseFeatures(frames[MIN(startIdx+offset,frames.size()-1)]->imgL, 
                                localKeypoints3, 
                                newProjPts,"2");
    }

}

void Optimizer::solveProblem(Problem& pb){
    cout << "solving problem..." <<endl;
    Solve(options, &pb, &summary);

    // if(options.minimizer_progress_to_stdout){
    //     cout << summary.FullReport() << endl;
    //     for(int n = 0; n < numCam; n++){
    //         cout << cameraParameter_[n*cameraBlkSize + 0] << "  ";
    //         cout << cameraParameter_[n*cameraBlkSize + 1] << "  ";
    //         cout << cameraParameter_[n*cameraBlkSize + 2] << "  ";
    //         cout << cameraParameter_[n*cameraBlkSize + 3] << "  ";
    //         cout << cameraParameter_[n*cameraBlkSize + 4] << "  ";
    //         cout << cameraParameter_[n*cameraBlkSize + 5] << endl;
    //     }
    // }

}

