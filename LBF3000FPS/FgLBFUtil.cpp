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
#include "FgLBFUtil.h"


Mat_d Coordinate::Image2Box(const Mat_d& shape, const cv::Rect& box)
{
	double_t boxcenterX = box.x + box.width / 2.0;
	double_t boxcenterY = box.y + box.height / 2.0;

	cv::Mat_<double_t> results(shape.rows, 2);
	for (int32_t i = 0; i < shape.rows; i++)
	{
		results(i, 0) = (shape(i, 0) - boxcenterX) / (box.width * 1.0);
		results(i, 1) = (shape(i, 1) - boxcenterY) / (box.height * 1.0);
	}
	return results;
}

Mat_d Coordinate::Box2Image(const Mat_d& shape, const cv::Rect& box)
{
	double_t boxcenterX = box.x + box.width / 2.0;
	double_t boxcenterY = box.y + box.height / 2.0;

	cv::Mat_<double_t> results(shape.rows, 2);
	for (int32_t i = 0; i < shape.rows; i++)
	{
		results(i, 0) = std::max(shape(i, 0) * box.width * 1.0 + boxcenterX, 0.0);
		results(i, 1) = std::max(shape(i, 1) * box.height * 1.0 + boxcenterY, 0.0);
	}
	return results;
}

Mat_d GetMeanShape(const vector<Mat_d>& allShape, const vector<cv::Rect>& allBoxes)
{
	Mat_d meanShape = Mat_d::zeros(allShape[0].rows, 2);

	for (int32_t i = 0; i < allShape.size(); ++i)
		meanShape += allShape[i];
	meanShape /= static_cast<double_t>(allShape.size());

	return meanShape;
}

double CalcVariance(const vector<double_t>& vec)
{
	double_t variance = 0.0;
	if (vec.size() == 0)
		return variance;

	Mat_d vec_(vec);
	double_t m1 = cv::mean(vec_)[0];
	double_t m2 = cv::mean(vec_.mul(vec_))[0];
	variance = m2 - m1*m1;
	return variance;
}

vector<cv::Point2d> ShapeToVecPoint(const Mat_d& Shape)
{
	vector<cv::Point2d> Ret(Shape.rows);
	for (int32_t row = 0; row < Shape.rows; ++row)
		Ret[row] = cv::Point2d(Shape(row, 0), Shape(row, 1));
	return Ret;
}

Mat_d VecPointToShape(const vector<cv::Point2d>& VecPoint)
{
	Mat_d Ret = Mat_d::zeros(static_cast<int32_t>(VecPoint.size()), 2);
	for (int32_t row = 0; row < VecPoint.size(); ++row)
	{
		Ret(row, 0) = VecPoint[row].x;
		Ret(row, 1) = VecPoint[row].y;
	}
	return Ret;
}


vector<Mat_uc>			g_ImageVec;
vector<Mat_d>			g_TruthShapeVec;
vector<cv::Rect>		g_BoxVec;

FgLBFParam				g_TrainParam;

std::ofstream & operator<<(std::ofstream & Out, Mat_d & Obj)
{
	Out << Obj.rows << " " << Obj.cols << std::endl;
	for (int32_t row = 0; row < Obj.rows; ++row)
	{
		for (int32_t col = 0; col < Obj.cols; ++col)
			Out << Obj(row, col) << " ";
		Out << std::endl;
	}

	return Out;
}

std::ifstream & operator>>(std::ifstream & In, Mat_d & Obj)
{
	int32_t row, col;

	In >> row >> col;
	if (In.bad())
		return In;

	Obj = Mat_d(row, col, 0.0);

	for (int32_t row = 0; row < Obj.rows; ++row)
	{
		for (int32_t col = 0; col < Obj.cols; ++col)
		{
			if (In.bad())
				return In;
			In >> Obj(row, col);
		}
	}

	return In;
}


Mat_d FgGetAffineTransform(const Mat_d& ShapeFrom, const Mat_d& ShapeTo)
{
	if (ShapeFrom.rows != ShapeTo.rows || ShapeFrom.cols != 2 || ShapeTo.cols != 2)
		ThrowFaile;

	Mat_d X(ShapeFrom.rows, 3, 0.0);

	for (int32_t i = 0; i < ShapeFrom.rows; ++i)
	{
		X(i, 0) = ShapeFrom(i, 0);
		X(i, 1) = ShapeFrom(i, 1);
		X(i, 2) = 1;
	}
	return ((X.t() * X).inv()*(X.t() * ShapeTo)).t();
}


double_t CalculateError(Mat_d& TruthShape, Mat_d& PredictedShape)
{
	Mat_d Temp = TruthShape.rowRange(36, 41) - TruthShape.rowRange(42, 47);
	double_t x = cv::mean(Temp.col(0))[0];
	double_t y = cv::mean(Temp.col(1))[0];
	double_t InterocularDistance = sqrt(x*x + y*y);
	double_t Sum = 0;
	for (int i = 0; i < TruthShape.rows; i++) {
		Sum += norm(TruthShape.row(i) - PredictedShape.row(i));
	}
	return Sum / (TruthShape.rows * InterocularDistance);
}