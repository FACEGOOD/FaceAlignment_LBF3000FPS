#pragma once
#include "FgLBFUtil.h"
#include "./RandomForest/FgLBFRandomForest.h"
#include "linear.h"

class FgLBFRegressor
{
public:
	FgLBFRegressor(vector<FgFaceData>& FaceDataVec);
	~FgLBFRegressor();

	vector<Mat_d> Train(int32_t Stage);
	Mat_d Predict(Mat_uc& Image, Mat_d& CurrentShape, cv::Rect& Box, Mat_d& MeanShapeTo);

	void LoadFromPath(string Path, int32_t Idx);
	void SaveToPath(string Path, int32_t Idx);
private:
	vector<FgFaceData>&		m_FaceDataVec;

	vector<FgLBFRandomForest>		m_RandomForestVec; //ÿһ��stage����landmark�����ɭ��
	vector<model*>					m_LinearModelX;
	vector<model*>					m_LinearModelY;
};

