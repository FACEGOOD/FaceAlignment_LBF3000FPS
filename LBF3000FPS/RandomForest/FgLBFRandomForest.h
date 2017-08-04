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
+	RandomForest

+	Date:		2017/7/17
+	Author:		ZhaoHang
*/
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

