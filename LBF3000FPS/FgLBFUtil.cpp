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

std::ofstream & MatFile::operator<<(std::ofstream & Out, Mat_d & Obj)
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

std::ifstream & MatFile::operator>>(std::ifstream & In, Mat_d & Obj)
{
	int32_t row, col;

	In >> row >> col;
	Obj = Mat_d(row, col, 0.0);

	for (int32_t row = 0; row < Obj.rows; ++row)
	{
		for (int32_t col = 0; col < Obj.cols; ++col)
		{
			In >> Obj(row, col);
		}
	}

	return In;
}


/**
* @brief CombineMultiImages  Combine the input images to a single big image.
* @param Images         The input images.
* @param NumberOfRows   The number of the rows to put the input images.
* @param NumberOfCols   The number of the cols to put the input images.
* @param Distance       The distance between each image.
* @param ImageWidth     The width of each image in the big image.
* @param ImageHeight    The height of each image in the big image.
* @return      The big image if the operation is successed;
*              otherwise return empty image.
*
* @author sheng
* @date 2015-03-24
* @version 0.1
*
* @history
*     <author>       <date>         <version>        <description>
*      sheng       2015-03-24          0.1           build the function
*
*/
cv::Mat CombineMultiImages(const std::vector<cv::Mat>& Images,
	const int NumberOfRows,
	const int NumberOfCols,
	const int Distance,
	const int ImageWidth,
	const int ImageHeight)
{
	// return empty mat if the Number of rows or cols is smaller than 1.  
	assert((NumberOfRows > 0) && (NumberOfCols > 0));
	if ((NumberOfRows < 1) || (NumberOfCols < 1))
	{
		std::cout << "The number of the rows or the cols is smaller than 1."
			<< std::endl;
		return cv::Mat();
	}


	// return empty mat if the distance, the width or the height of image  
	// is smaller than 1.  
	assert((Distance > 0) && (ImageWidth > 0) && (ImageHeight > 0));
	if ((Distance < 1) || (ImageWidth < 1) || (ImageHeight < 1))
	{
		std::cout << "The distance, the width or the height of the image is smaller than 1."
			<< std::endl;
		return cv::Mat();
	}


	// Get the number of the input images  
	const int NUMBEROFINPUTIMAGES = static_cast<int>(Images.size());


	// return empty mat if the number of the input images is too big.  
	assert(NUMBEROFINPUTIMAGES <= NumberOfRows * NumberOfCols);
	if (NUMBEROFINPUTIMAGES > NumberOfRows * NumberOfCols)
	{
		std::cout << "The number of images is too big." << std::endl;
		return cv::Mat();
	}


	// return empty mat if the number of the input images is too low.  
	assert(NUMBEROFINPUTIMAGES > 0);
	if (NUMBEROFINPUTIMAGES < 1)
	{
		std::cout << "The number of images is too low." << std::endl;
		return cv::Mat();
	}


	// create the big image  
	const int WIDTH = Distance * (NumberOfCols + 1) + ImageWidth * NumberOfCols;
	const int HEIGHT = Distance * (NumberOfRows + 1) + ImageHeight * NumberOfRows;
	cv::Scalar Color(255, 255, 255);
	if (Images[0].channels() == 1)
	{
		Color = cv::Scalar(255);
	}
	cv::Mat ResultImage(HEIGHT, WIDTH, Images[0].type(), Color);



	// copy the input images to the big image  
	for (int Index = 0; Index < NUMBEROFINPUTIMAGES; Index++)
	{

		assert(Images[Index].type() == ResultImage.type());
		if (Images[Index].type() != ResultImage.type())
		{
			std::cout << "The No." << Index << "image has the different type."
				<< std::endl;
			return cv::Mat();
		}


		// Get the row and the col of No.Index image  
		int Rows = Index / NumberOfCols;
		int Cols = Index % NumberOfCols;

		// The start point of No.Index image.  
		int StartRows = Distance * (Rows + 1) + ImageHeight * Rows;
		int StartCols = Distance * (Cols + 1) + ImageWidth * Cols;

		// copy  No.Index image to the big image  
		cv::Mat ROI = ResultImage(cv::Rect(StartCols, StartRows,
			ImageWidth, ImageHeight));
		cv::resize(Images[Index], ROI, cv::Size(ImageWidth, ImageHeight));

	}

	return ResultImage;
}

double CalculateError(cv::Mat_<double>& ground_truth_shape, cv::Mat_<double>& predicted_shape) {
	cv::Mat_<double> temp;
	temp = ground_truth_shape.rowRange(36, 41) - ground_truth_shape.rowRange(42, 47);
	double x = mean(temp.col(0))[0];
	double y = mean(temp.col(1))[0];
	double interocular_distance = sqrt(x*x + y*y);
	double sum = 0;
	for (int i = 0; i < ground_truth_shape.rows; i++) {
		sum += norm(ground_truth_shape.row(i) - predicted_shape.row(i));
	}
	return sum / (ground_truth_shape.rows*interocular_distance);
}

bool ShapeInRect(Mat_d& shape, cv::Rect& ret)
{
	double_t sum_x = 0.0, sum_y = 0.0;
	double_t max_x = 0, min_x = 10000, max_y = 0, min_y = 10000;
	for (int i = 0; i < shape.rows; i++)
	{
		if (shape(i, 0) > max_x) max_x = shape(i, 0);
		if (shape(i, 0) < min_x) min_x = shape(i, 0);
		if (shape(i, 1) > max_y) max_y = shape(i, 1);
		if (shape(i, 1) < min_y) min_y = shape(i, 1);

		sum_x += shape(i, 0);
		sum_y += shape(i, 1);
	}
	sum_x /= shape.rows;
	sum_y /= shape.rows;

	if ((max_x - min_x) > ret.width * 1.5) return false;
	if ((max_y - min_y) > ret.height * 1.5) return false;
	if (std::abs(sum_x - (ret.x + ret.width / 2.0)) > ret.width / 2.0) return false;
	if (std::abs(sum_y - (ret.y + ret.height / 2.0)) > ret.height / 2.0) return false;
	return true;
}