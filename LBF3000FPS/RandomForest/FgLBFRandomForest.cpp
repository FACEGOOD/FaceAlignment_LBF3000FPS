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
+	RandomForest

+	Date:		2017/7/20
+	Author:		ZhaoHang
*/
#include "FgLBFRandomForest.h"
#include <ppl.h>

FgLBFRandomForest::FgLBFRandomForest(const vector<FgFaceData>& FaceDataVec, const int32_t Stage, const int32_t LandMarkIdx)
	:m_FaceDataVec(FaceDataVec), m_Stage(Stage), m_LandmarkIdx(LandMarkIdx)
{
}

void FgLBFRandomForest::TrainForest(const vector<Mat_d>& TargetVec)
{
	int32_t AllSamplesNum = static_cast<int32_t> (m_FaceDataVec.size());
	int32_t SamplesNumPerTree = static_cast<int32_t>(AllSamplesNum / (g_TrainParam.TreeNumPerForest - g_TrainParam.DataAugmentOverLap*(g_TrainParam.TreeNumPerForest - 1)));
	int32_t SamplesNumOverLap = static_cast<int32_t>(SamplesNumPerTree * g_TrainParam.DataAugmentOverLap);
	int32_t SamplesNumNotOverLap = SamplesNumPerTree - SamplesNumOverLap;

	double_t RandomRadiu = g_TrainParam.LocalRadiusPerStageVec[m_Stage];
	for (int32_t FeaturesIdx = 0; FeaturesIdx < g_TrainParam.LocalFeaturesNum; ++FeaturesIdx)
	{
		std::mt19937 rng;
		rng.seed(std::random_device()());
		std::uniform_real_distribution<double_t> RandomGen(-RandomRadiu, RandomRadiu);
		cv::Point2d A, B;
		do
		{
			A.x = RandomGen(rng);
			A.y = RandomGen(rng);
		} while (A.x*A.x + A.y*A.y > RandomRadiu*RandomRadiu);
		do
		{
			B.x = RandomGen(rng);
			B.y = RandomGen(rng);
		} while (B.x*B.x + B.y*B.y > RandomRadiu*RandomRadiu);

		m_LocationFeature.push_back(std::make_pair(A, B));
	}

	Mat_<int32_t> PixDifferences(g_TrainParam.LocalFeaturesNum, AllSamplesNum);

	for (int32_t Sample = 0; Sample < AllSamplesNum; ++Sample)
	{
		const FgFaceData& FaceData = m_FaceDataVec[Sample];
		for (int32_t Feature = 0; Feature < g_TrainParam.LocalFeaturesNum; ++Feature)
		{
			Mat_d RandomPoints = Mat_d::zeros(2, 2);

			cv::Point2d A = m_LocationFeature[Feature].first;
			cv::Point2d B = m_LocationFeature[Feature].second;

			vector<cv::Point2d> Vec{ A,B };
			cv::transform(Vec, Vec, FaceData.MeanShapeTo);

			RandomPoints(0, 0) = Vec[0].x + FaceData.CurrentShape(m_LandmarkIdx, 0);
			RandomPoints(0, 1) = Vec[0].y + FaceData.CurrentShape(m_LandmarkIdx, 1);
			RandomPoints(1, 0) = Vec[1].x + FaceData.CurrentShape(m_LandmarkIdx, 0);
			RandomPoints(1, 1) = Vec[1].y + FaceData.CurrentShape(m_LandmarkIdx, 1);

			Mat_d ImagePoints = Coordinate::Box2Image(RandomPoints, g_BoxVec[FaceData.BoxIdx]);
			const Mat_uc& Image = g_ImageVec[FaceData.ImageIdx];

			for (int32_t i = 0; i < 2; ++i)
			{
				if (ImagePoints(i, 0) >= Image.cols - 1)
					ImagePoints(i, 0) = Image.cols - 1;
				if (ImagePoints(i, 1) >= Image.rows - 1)
					ImagePoints(i, 1) = Image.rows - 1;
			}

			PixDifferences(Feature, Sample) = Image(static_cast<int32_t>(ImagePoints(0, 1)), static_cast<int32_t>(ImagePoints(0, 0))) - Image(static_cast<int32_t>(ImagePoints(1, 1)), static_cast<int32_t>(ImagePoints(1, 0)));
		}
	}

	m_TreeVec.resize(g_TrainParam.TreeNumPerForest);

	for (int32_t Tree = 0; Tree < g_TrainParam.TreeNumPerForest; ++Tree)
	{
		int32_t StartIdx = Tree * SamplesNumNotOverLap;
		int32_t EndIdx = StartIdx + SamplesNumPerTree;
		if (EndIdx > AllSamplesNum - 1)
			EndIdx = AllSamplesNum - 1;

		vector<int32_t> CurrentIndexVec;
		for (int32_t s = StartIdx; s < EndIdx; ++s)
			CurrentIndexVec.push_back(s);

		vector<int32_t> SelectedIdxVec;
		m_TreeVec[Tree] = BuildTree(PixDifferences, CurrentIndexVec, SelectedIdxVec, 0, TargetVec);
	}
}

FgLBFNode * FgLBFRandomForest::BuildTree(const Mat_<int32_t>& PixDifferences, const vector<int32_t>& DataIdxVec, vector<int32_t>& SelectedIdxVec, int32_t Depth, const vector<Mat_d>& TargetVec)
{
	FgLBFNode* Node = new FgLBFNode();
	Node->m_Depth = Depth;

	if (Depth == g_TrainParam.TreeDepth)
	{
		Node->m_IsLeaf = true;
		Node->m_LeafIdentity = m_AllLeafNum++;
		return Node;
	}


	Mat_<int32_t> TreeInput = Mat_d::zeros(g_TrainParam.LocalFeaturesNum, static_cast<int32_t>(DataIdxVec.size()));
	double_t RandomRadiu = g_TrainParam.LocalRadiusPerStageVec[m_Stage];

	for (int32_t SamplesIdx = 0; SamplesIdx < DataIdxVec.size(); ++SamplesIdx)
	{
		for (int32_t FeaturesIdx = 0; FeaturesIdx < g_TrainParam.LocalFeaturesNum; ++FeaturesIdx)
		{
			TreeInput(FeaturesIdx, SamplesIdx) = PixDifferences(FeaturesIdx, DataIdxVec[SamplesIdx]);
		}
	}

	vector<int32_t> LeftVec, RightVec;

	auto LeftAndRight = CalcSplitFeature(TreeInput, SelectedIdxVec, DataIdxVec, TargetVec);
	if (std::get<0>(LeftAndRight) == INT32_MIN)
		ThrowFaile;

	Mat_<int32_t> Samples = std::get<2>(LeftAndRight);

	Node->m_Threshold = std::get<0>(LeftAndRight);
	Node->m_FeatureLocations = m_LocationFeature[std::get<1>(LeftAndRight)];

	for (int32_t col = 0; col < Samples.cols; col++)
	{
		if (Samples(0, col) < Node->m_Threshold)
			LeftVec.push_back(DataIdxVec[col]);
		else
			RightVec.push_back(DataIdxVec[col]);
	}

	if (LeftVec.size() == 0 || RightVec.size() == 0)
	{
		Node->m_IsLeaf = true;
		Node->m_LeafIdentity = m_AllLeafNum++;
		return Node;
	}

	SelectedIdxVec.push_back(std::get<1>(LeftAndRight));

	Node->m_LeftChild = BuildTree(PixDifferences, LeftVec, SelectedIdxVec, Depth + 1, TargetVec);
	Node->m_RightChild = BuildTree(PixDifferences, RightVec, SelectedIdxVec, Depth + 1, TargetVec);

	return Node;
}

std::tuple<int32_t, int32_t, Mat_<int32_t>> FgLBFRandomForest::CalcSplitFeature(const Mat_<int32_t>& Input, const vector<int32_t>& SelectedIdxVec, const vector<int32_t>& DataIdxVec, const vector<Mat_d>& TargetVec)
{
	double_t MinVariance = FLT_MAX;
	int32_t Threshold = INT32_MIN;
	int32_t SamplesRows = INT32_MIN;
	Mat_<int32_t> Samples;

	vector<int32_t> LeftVec, RightVec;

	for (int32_t row = 0; row < Input.rows; ++row)
	{
		if (std::find(SelectedIdxVec.begin(), SelectedIdxVec.end(), row) != SelectedIdxVec.end())
			continue;

		Mat_<int32_t> InputToCalc = Input.row(row);
		{
			cv::Mat unuse;

			std::mt19937 rng;
			rng.seed(std::random_device()());
			std::uniform_real_distribution<double_t> RandomGen(0.05, 0.95);

			auto& e = InputToCalc(0, static_cast<int32_t> (floor(InputToCalc.cols * RandomGen(rng))));

			{
				int32_t TestThreshold = e;
				vector<double_t> LeftX, LeftY, RightX, RightY;
				for (int32_t i = 0; i < InputToCalc.cols; ++i)
				{
					Mat_d Var = TargetVec[DataIdxVec[i]].row(m_LandmarkIdx);

					if (InputToCalc(0, i) < TestThreshold)
					{
						LeftX.push_back(Var(0, 0));
						LeftY.push_back(Var(0, 1));
					}
					else
					{
						RightX.push_back(Var(0, 0));
						RightY.push_back(Var(0, 1));
					}
				}
				double_t Sum = (CalcVariance(LeftX) + CalcVariance(LeftY)) * LeftX.size() + (CalcVariance(RightX) + CalcVariance(RightY)) * RightX.size();
				if (Sum < MinVariance)
				{
					MinVariance = Sum;
					Threshold = TestThreshold;
					SamplesRows = row;
					Samples = InputToCalc;
				}
			}
		}
	}

	return std::make_tuple(Threshold, SamplesRows, Samples);
}

std::ofstream & operator<<(std::ofstream & Out, FgLBFRandomForest & Obj)
{
	Out << Obj.m_AllLeafNum << std::endl;
	for (auto& Tree : Obj.m_TreeVec)
		Out << *Tree;
	return Out;
}

std::ifstream & operator>>(std::ifstream & In, FgLBFRandomForest & Obj)
{
	Obj.m_TreeVec.resize(g_TrainParam.TreeNumPerForest);

	In >> Obj.m_AllLeafNum;
	for (auto& Tree : Obj.m_TreeVec)
	{
		Tree = new FgLBFNode();
		In >> *Tree;
	}
	return In;
}
