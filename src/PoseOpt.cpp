#include "PoseOpt.h"

// using namespace std;
using namespace g2o;

PoseOpt::PoseOpt(){
	SlamLinearSolver* linearSolver = new SlamLinearSolver();
    linearSolver->setBlockOrdering( false );
    SlamBlockSolver* blockSolver = new SlamBlockSolver( linearSolver );
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg( blockSolver );
    //==============================================
    optimizer.setAlgorithm( solver ); //the one will be used all the time
    optimizer.setVerbose( true );

    //add the first vertex
    g2o::VertexSE3* v = new g2o::VertexSE3();
    v->setId(0);
    v->setEstimate(Eigen::Isometry3d::Identity());
    v->setFixed( true ); //fix the first vertex
	optimizer.addVertex(v);
}

void PoseOpt::addNode(const vector<Frame*>& frames, int idx){
	if(0 == idx){return;}

	VertexSE3 *v = new VertexSE3();

	Eigen::Affine3d T = vectorToTransformation(frames[idx]->worldRvec, frames[idx]->worldTvec);
	Eigen::Affine3d invT = T.inverse();

	Eigen::Isometry3d _invT;
	for(int r = 0; r < 4; r++){
		for(int c = 0; c < 4; c++){
			_invT(r,c) = invT(r,c);
		}
	}


	v->setId(frames[idx]->frameID);
	v->setEstimate(_invT);
	optimizer.addVertex(v);
}

void PoseOpt::addEdge(const vector<Frame*>& frames, int fromIdx){

	for(edgeConstrain::iterator eIt = frames[fromIdx]->relativePose.begin();
								eIt!= frames[fromIdx]->relativePose.end();
								eIt++) {
		Eigen::Affine3d T = vectorToTransformation((*eIt).second.first, (*eIt).second.second);
		Eigen::Isometry3d _T;

		for(int r = 0; r < 4; r++){
			for(int c = 0; c < 4; c++){
				_T(r,c) = T(r,c);
			}
		}

		EdgeSE3* edge = new EdgeSE3();
		edge->vertices()[0] = optimizer.vertex(frames[fromIdx]->frameID);
		edge->vertices()[1] = optimizer.vertex((*eIt).first);

		// edge->setRobustKernel(robustKernel);

		//information matrix
	    Eigen::Matrix<double, 6, 6> information = Eigen::Matrix< double, 6,6 >::Identity();
	    // 信息矩阵是协方差矩阵的逆，表示我们对边的精度的预先估计
	    // 因为pose为6D的，信息矩阵是6*6的阵，假设位置和角度的估计精度均为0.1且互相独立
	    // 那么协方差则为对角为0.01的矩阵，信息阵则为100的矩阵
	    information(0,0) = information(1,1) = information(2,2) = 80;
	    information(3,3) = information(4,4) = information(5,5) = 110;
	    // 也可以将角度设大一些，表示对角度的估计更加准确
		edge->setInformation(information);
		edge->setMeasurement(_T.inverse());
		optimizer.addEdge(edge);
		cout << "added new edge from " << frames[fromIdx]->frameID << " to " << (*eIt).first << endl;
	} 

}

void PoseOpt::solve() {
	cout << "optimizing pose graph, vertice numbers: " << optimizer.vertices().size() << endl;
	optimizer.save("../pose_graph/before_full_optimize.g2o");    
	optimizer.computeActiveErrors();
    cout << "Initial chi2 = " << optimizer.chi2() << endl;
	optimizer.initializeOptimization();
	optimizer.optimize(25);
	optimizer.save("../pose_graph/after_full_optimize.g2o");
	cout<<"Optimization done."<<endl;
	//update pose

}







