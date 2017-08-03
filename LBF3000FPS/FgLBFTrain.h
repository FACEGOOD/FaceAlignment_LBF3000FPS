#pragma once

#include "FgLBFUtil.h"
#include "FgLBFRegressor.h"

class FgLBFTrain
{
public:
	FgLBFTrain(string TrainPath);
	~FgLBFTrain();

	FgLBFTrain(const FgLBFTrain&) = delete;
	FgLBFTrain& operator = (const FgLBFTrain&) = delete;

public:
	void Train();

	void Predict(string ImageListPath);
	Mat_d Predict(Mat_uc & Image, cv::Rect Box, Mat_d& LastFrame = Mat_d());
private:
	string			m_TrainPath;
	string			m_TestImagePath;

	vector<FgFaceData>		m_FaceDataVec;
	vector<FgLBFRegressor>	m_RegressorVec;
private:
	void Load();
	void LoadFromPath(string File);
	void SaveToPath(string Path);

	void LoadImageList(string FilePath, vector<Mat_uc>& ImageVec, vector<Mat_d>& TruthShape, vector<cv::Rect>& BoxVec);
	void DataAugment();
};

