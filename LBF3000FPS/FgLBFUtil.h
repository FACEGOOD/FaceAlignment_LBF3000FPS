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

double CalculateError(cv::Mat_<double>& ground_truth_shape, cv::Mat_<double>& predicted_shape);

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