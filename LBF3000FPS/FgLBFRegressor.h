/*
+	LBF3000
+	A Implementation of highly efficient and very accurate regression approach for face alignment.
	Quantum Dynamics Co.,Ltd. ���Ӷ��������ڣ�������Ƽ����޹�˾

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
+	����Ŀ�������Ȩ��ѧ���о�������������ҵ����

+	(C)	Quantum Dynamics Lab. ���Ӷ���ʵ����
+		Website : http://www.facegood.cc
+		Contact Us : jelo@facegood.cc

+		-Thanks to Our Committers and Friends
+		-Best Wish to all who Contributed and Inspired
*/

/*
+	Regressor

+	Date:		2017/7/20
+	Author:		ZhaoHang
*/
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

