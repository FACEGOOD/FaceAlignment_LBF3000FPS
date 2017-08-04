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
+	Function & Global & Macro

+	Date:		2017/7/20
+	Author:		ZhaoHang
*/
#pragma once

#pragma warning(disable:4996)
#pragma warning(disable:4819)


#include <iostream>
#include <vector>
#include <random>
#include <opencv2/opencv.hpp>


#ifdef _DEBUG
#pragma comment(lib,"opencv_world320d.lib")
#else
#pragma comment(lib,"opencv_world320.lib")
#endif // _DEBUG


using cv::Mat_;
using std::vector;
using std::string;

typedef Mat_<double_t> Mat_d;
typedef Mat_<uchar> Mat_uc;

#define ThrowFaile throw (std::string(__FILE__) + "[line: " + std::to_string(__LINE__) + " ]")
#define IMG_MAX_SIZE 1000

namespace Coordinate
{
	Mat_d Image2Box(const Mat_d & shape, const cv::Rect & box);
	Mat_d Box2Image(const Mat_d & shape, const cv::Rect & box);
}

Mat_d GetMeanShape(const vector<Mat_d>& allShape, const vector<cv::Rect>& allBoxes);

double_t CalcVariance(const vector<double_t>& vec);

vector<cv::Point2d> ShapeToVecPoint(const Mat_d & Shape);

Mat_d VecPointToShape(const vector<cv::Point2d>& VecPoint);

Mat_d FgGetAffineTransform(const Mat_d & ShapeFrom, const Mat_d & ShapeTo);

double_t CalculateError(Mat_d& TruthShape, Mat_d& PredictedShape);

std::ofstream & operator << (std::ofstream &Out, Mat_d &Obj);
std::ifstream & operator >> (std::ifstream &In, Mat_d &Obj);

struct FgFaceData
{
	int32_t ImageIdx;
	int32_t TruthShapeIdx;
	int32_t BoxIdx;
	Mat_d CurrentShape; //[landmark x 2]

	Mat_d MeanShapeTo;
	Mat_d ToMeanShape;
};

struct FgLBFParam
{
	int32_t			LocalFeaturesNum;
	int32_t			LandmarkNumPerFace;
	int32_t			RegressroStage;
	int32_t			TreeDepth;
	int32_t			TreeNumPerForest;
	Mat_d			MeanShape;
	double_t		DataAugmentOverLap;
	int32_t			DataAugmentScale;


	vector<double_t>		LocalRadiusPerStageVec;
};

extern vector<Mat_uc>			g_ImageVec;
extern vector<Mat_d>			g_TruthShapeVec;
extern vector<cv::Rect>			g_BoxVec;
extern FgLBFParam				g_TrainParam;