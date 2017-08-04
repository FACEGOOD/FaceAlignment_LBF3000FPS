/*
+	LBF3000
+	A Implementation of highly efficient and very accurate regression approach for face alignment.
	Quantum Dynamics Co.,Ltd. 量子动力（深圳）计算机科技有限公司

	Based on the paper 'Face Alignment at 3000 FPS via Regressing Local Binary Features'
	University of Science and Technology of China.
	Microsoft Research.

+	'LBF3000' is developing under the terms of the GNU General Public License as published by the Free Software Foundation.
+	The project lunched by 'Quantum Dynamics Lab.' since 4.Aug.2017.

+	You can redistribute it and/or modify it under the terms of the GNU General Public License version 2 (GPLv2) of
+	the license as published by the free software foundation.this program is distributed in the hope
+	that it will be useful,but without any warranty.without even the implied warranty of merchantability
+	or fitness for a particular purpose.

+	This project allows for academic research only.
+	本项目代码仅授权于学术研究，不可用于商业化。

+	(C)	Quantum Dynamics Lab. 量子动力实验室
+		Website : http://www.facegood.cc
+		Contact Us : jelo@facegood.cc

+		-Thanks to Our Committers and Friends
+		-Best Wish to all who Contributed and Inspired
*/

/*
+	Start Train & Start Predict

+	Date:		2017/7/20
+	Author:		ZhaoHang
*/
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

