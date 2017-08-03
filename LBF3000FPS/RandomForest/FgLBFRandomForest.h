#pragma once
#include "../FgLBFUtil.h"
#include "FgLBFNode.h"

class FgLBFRandomForest
{
	friend std::ofstream & operator << (std::ofstream &Out, FgLBFRandomForest &Obj);
	friend std::ifstream & operator >> (std::ifstream &In, FgLBFRandomForest &Obj);
public:
	FgLBFRandomForest(const vector<FgFaceData>& FaceDataVec, const int32_t Stage, const int32_t LandMarkIdx);
	~FgLBFRandomForest() = default;


	void TrainForest(const vector<Mat_d>& TargetVec);
	int32_t GetLeafNodesNum() { return m_AllLeafNum; }
public:
	vector<FgLBFNode*>	m_TreeVec;
private:
	const int32_t	m_Stage;
	const int32_t	m_LandmarkIdx;
	const vector<FgFaceData>&	m_FaceDataVec;

	vector<std::pair<cv::Point2d, cv::Point2d>>	m_LocationFeature;

	int32_t	m_AllLeafNum = 0;
private:
	FgLBFNode * BuildTree(const Mat_<int32_t>& PixDifferences,const vector<int32_t>& DataIdxVec, vector<int32_t>& SelectedIdxVec, int32_t Depth, const vector<Mat_d>& TargetVec);
	std::tuple<int32_t, int32_t, Mat_<int32_t>> CalcSplitFeature(const Mat_<int32_t>& Input, const vector<int32_t>& SelectedIdxVec, const vector<int32_t>& DataIdxVec, const vector<Mat_d>& TargetVec);
};

