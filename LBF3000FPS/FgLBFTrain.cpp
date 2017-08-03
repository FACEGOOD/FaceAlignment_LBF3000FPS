#include <fstream>
#include <ppl.h>
#include <Mutex>
#include <direct.h>

#include "FgLBFTrain.h"

FgLBFTrain::FgLBFTrain(string TrainPath)
	: m_TrainPath(TrainPath)
{
}

FgLBFTrain::~FgLBFTrain()
{
}

void FgLBFTrain::Load()
{
	std::ifstream TrainConfig(m_TrainPath + "TrainConfig.txt");
	if (!TrainConfig.is_open())
		ThrowFaile;

	TrainConfig >> g_TrainParam.LocalFeaturesNum
		>> g_TrainParam.LandmarkNumPerFace
		>> g_TrainParam.RegressroStage
		>> g_TrainParam.TreeDepth
		>> g_TrainParam.TreeNumPerForest
		>> g_TrainParam.DataAugmentScale
		>> g_TrainParam.DataAugmentOverLap;

	g_TrainParam.LocalRadiusPerStageVec.resize(g_TrainParam.RegressroStage);
	for (int32_t i = 0; i < g_TrainParam.RegressroStage; ++i)
	{
		TrainConfig >> g_TrainParam.LocalRadiusPerStageVec[i];
	}

	string TrainImageList;
	TrainConfig >> TrainImageList;

	LoadImageList(m_TrainPath + TrainImageList, g_ImageVec, g_TruthShapeVec, g_BoxVec);

	int32_t HasTestImage;
	TrainConfig >> HasTestImage;
	if (HasTestImage > 0)
	{
		TrainConfig >> m_TestImagePath;
	}

	TrainConfig.close();

	g_TrainParam.MeanShape = GetMeanShape(g_TruthShapeVec, g_BoxVec);

	//Data Augment
	DataAugment();
}

void FgLBFTrain::Train()
{
	Load();
	m_RegressorVec.resize(g_TrainParam.RegressroStage, FgLBFRegressor(m_FaceDataVec));
	for (int32_t i = 0; i < m_RegressorVec.size(); ++i)
	{
		std::cout << "Train Stage " << i << std::endl;
		vector<Mat_d> Delta = m_RegressorVec[i].Train(i);
		for (int32_t j = 0; j < m_FaceDataVec.size(); ++j)
			m_FaceDataVec[j].CurrentShape += Delta[j];
	}

	SaveToPath(m_TrainPath + "Model/");

	if (!m_TestImagePath.empty())
	{
		std::cout << "Test Image" << std::endl;

		vector<Mat_uc>			ImageVec;
		vector<Mat_d>			TruthShapeVec;
		vector<cv::Rect>		BoxVec;
		LoadImageList(m_TrainPath + "Test_Image.txt", ImageVec, TruthShapeVec, BoxVec);

		double_t E = 0;
		for (int32_t i = 0; i < ImageVec.size(); ++i)
		{
			Mat_d PredictShape = Predict(ImageVec[i], BoxVec[i]);
			E += CalculateError(TruthShapeVec[i], PredictShape);
		}
		std::cout << "Error :[" << E << "]\t Mean Error :[" << E / ImageVec.size() << "]" << std::endl;
	}
}

void FgLBFTrain::Predict(string ImageListPath)
{
	LoadFromPath(m_TrainPath + "Model/");

	cv::VideoCapture Cam(/*"K:/2.mp4"*/0);

	//vector<Mat_uc>			ImageVec;
	//vector<Mat_d>			TruthShapeVec;
	//vector<cv::Rect>		BoxVec;
	//LoadImageList(ImageListPath, ImageVec, TruthShapeVec, BoxVec);

	//double_t E = 0;
	//for (int32_t i = 0; i < ImageVec.size(); ++i)
	//{
	//	Mat_d PredictShape = Predict(ImageVec[i], BoxVec[i]);
	//	Mat_d ImageShape = Coordinate::Box2Image(PredictShape, BoxVec[i]);
	//	for (int32_t Landmark = 0; Landmark < ImageShape.rows; ++Landmark)
	//	{
	//		cv::circle(ImageVec[i], { static_cast<int32_t>(ImageShape(Landmark,0)) ,static_cast<int32_t>(ImageShape(Landmark,1)) }, 4, { 0 }, -1);
	//	}
	//	cv::imshow("T", ImageVec[i]);
	//	double_t Err = CalculateError(TruthShapeVec[i], PredictShape);
	//	std::cout << Err << std::endl;
	//	E += Err;
	//	cv::waitKey(-1);
	//}
	//std::cout << "Error :[" << E << "]\t Mean Error :[" << E / ImageVec.size() << "]" << std::endl;

	cv::Mat ImageColor;
	cv::CascadeClassifier Cs("../OpenCV/etc/haarcascades/haarcascade_frontalface_alt.xml");

	vector<cv::Rect> FaceBoxs;
	int32_t i = 0;
	Mat_d LastFrame;
	while (true)
	{
		Mat_uc Image;
		Cam >> ImageColor;
		cvtColor(ImageColor, Image, CV_BGR2GRAY);

		//if (FaceBoxs.empty() || LastFrame.empty())
		Cs.detectMultiScale(Image, FaceBoxs);

		for (auto& var : FaceBoxs)
		{
			//scale 1.5
			cv::Rect FaceBox = var;
			double_t CenterX = FaceBox.x + 0.5 * FaceBox.width;
			double_t CenterY = FaceBox.y + 0.5 * FaceBox.height;

			FaceBox.x = static_cast<int32_t>(CenterX - 0.75 * FaceBox.width);
			FaceBox.y = static_cast<int32_t>(CenterY - 0.75 * FaceBox.height);
			FaceBox.width = static_cast<int32_t>(1.5 * FaceBox.width);
			FaceBox.height = static_cast<int32_t>(1.5 * FaceBox.height);

			Mat_d PredictShape = Predict(Image, FaceBox, LastFrame);
			if (LastFrame.empty())
				LastFrame = PredictShape.clone();
			//else
			//{
			//	Mat_d This = Coordinate::Box2Image(PredictShape, FaceBox);
			//	Mat_d Last = Coordinate::Box2Image(LastFrame, FaceBox);

			//	double_t ShiftX = cv::mean(This.col(0))(0) - cv::mean(Last.col(0))(0);
			//	double_t ShiftY = cv::mean(This.col(1))(0) - cv::mean(Last.col(1))(0);

			//	var.x += static_cast<int32_t>(ShiftX);
			//	var.y += static_cast<int32_t>(ShiftY);
			//}
			PredictShape = Coordinate::Box2Image(PredictShape, FaceBox);

			for (int32_t Landmark = 17; Landmark < g_TrainParam.LandmarkNumPerFace; ++Landmark)
			{
				cv::circle(ImageColor, { static_cast<int32_t>(PredictShape(Landmark,0)) ,static_cast<int32_t>(PredictShape(Landmark,1)) }, 2, { 255 }, -1);
			}
			cv::rectangle(ImageColor, FaceBox, { 0 });

			cv::imshow("T", ImageColor);
			cv::waitKey(30);
		}
	}



	//for (int32_t i = 0; i < ImageVec.size(); ++i)
	//{
	//	//vector<cv::Rect> FaceBoxs;
	//	//Cs.detectMultiScale(ImageVec[i], FaceBoxs);

	//	Mat_uc Image = ImageVec[i].clone();

	//	Mat_d PredictShape = Predict(ImageVec[i], BoxVec[i]);

	//	for (int32_t Landmark = 0; Landmark < g_TrainParam.LandmarkNumPerFace; ++Landmark)
	//		cv::circle(Image, { static_cast<int32_t>(PredictShape(Landmark,0)) ,static_cast<int32_t>(PredictShape(Landmark,1)) }, 2, { 255 }, -1);
	//	cv::rectangle(Image, BoxVec[i], { 0 });

	//	cv::imshow("T", Image);
	//	cv::waitKey(-1);
	//}
}

Mat_d FgLBFTrain::Predict(Mat_uc & Image, cv::Rect Box, Mat_d& LastFrame)
{
	Mat_d CurrentShape = g_TrainParam.MeanShape.clone();
	for (int32_t Stage = 0; Stage < m_RegressorVec.size(); ++Stage)
	{
		Mat_d MeanShapeTo = cv::estimateAffine2D(LastFrame.empty() ? g_TrainParam.MeanShape : LastFrame, CurrentShape, cv::noArray(), cv::LMEDS);
		CurrentShape += m_RegressorVec[Stage].Predict(Image, CurrentShape, Box, MeanShapeTo);
	}
	return CurrentShape;
}

void FgLBFTrain::LoadFromPath(string Path)
{
	std::ifstream ifs(Path + "Model.bin");
	if (!ifs.is_open())
		ThrowFaile;

	ifs >> g_TrainParam.LocalFeaturesNum;
	ifs >> g_TrainParam.LandmarkNumPerFace;
	ifs >> g_TrainParam.RegressroStage;
	ifs >> g_TrainParam.TreeDepth;
	ifs >> g_TrainParam.TreeNumPerForest;
	MatFile::operator>>(ifs, g_TrainParam.MeanShape);
	ifs >> g_TrainParam.DataAugmentOverLap;
	ifs >> g_TrainParam.DataAugmentScale;

	g_TrainParam.LocalRadiusPerStageVec.resize(g_TrainParam.RegressroStage);
	for (auto& var : g_TrainParam.LocalRadiusPerStageVec)
		ifs >> var;

	m_RegressorVec.resize(g_TrainParam.RegressroStage, FgLBFRegressor(vector<FgFaceData>()));
	for (int32_t i = 0; i < g_TrainParam.RegressroStage; ++i)
		m_RegressorVec[i].LoadFromPath(Path, i);

	ifs.close();
	return;
}

void FgLBFTrain::SaveToPath(string Path)
{
	_mkdir(Path.c_str());

	std::ofstream ofs(Path + "Model.bin");
	if (!ofs.is_open())
		ThrowFaile;

	ofs << g_TrainParam.LocalFeaturesNum << std::endl;
	ofs << g_TrainParam.LandmarkNumPerFace << std::endl;
	ofs << g_TrainParam.RegressroStage << std::endl;
	ofs << g_TrainParam.TreeDepth << std::endl;
	ofs << g_TrainParam.TreeNumPerForest << std::endl;
	MatFile::operator<<(ofs, g_TrainParam.MeanShape) << std::endl;
	ofs << g_TrainParam.DataAugmentOverLap << std::endl;
	ofs << g_TrainParam.DataAugmentScale << std::endl;

	for (auto& var : g_TrainParam.LocalRadiusPerStageVec)
		ofs << var;

	ofs.flush();
	ofs.close();

	int32_t Idx = 0;
	for (auto& var : m_RegressorVec)
		var.SaveToPath(Path, Idx++);

	return;
}

void FgLBFTrain::LoadImageList(string FilePath, vector<Mat_uc>& ImageVec, vector<Mat_d>& TruthShape, vector<cv::Rect>& BoxVec)
{
	std::ifstream fin(FilePath);
	if (!fin.is_open())
		ThrowFaile;

	vector<string> ImagePathVec;
	vector<string> PtsPathVec;
	vector<string> FgrPathVec;

	string ImagePath;
	while (std::getline(fin, ImagePath))
	{
		ImagePathVec.push_back(ImagePath);

		string PtsPath = ImagePath;
		auto Pos = PtsPath.find_last_of('.');
		PtsPath.replace(Pos, PtsPath.length() - 1, ".pts");

		string FgrPath = ImagePath;
		FgrPath.replace(Pos, FgrPath.length() - 1, ".fgr");

		PtsPathVec.push_back(PtsPath);
		FgrPathVec.push_back(FgrPath);
	}
	fin.close();

	std::mutex mtx;
	concurrency::parallel_for(0, static_cast<int32_t>(ImagePathVec.size()), [&](int32_t i)
	{
		Mat_d Shape(g_TrainParam.LandmarkNumPerFace, 2, 0.0);

		std::ifstream fpts(PtsPathVec[i]);
		if (fpts.is_open())
		{
			int32_t landmark;
			string unuse;

			std::getline(fpts, unuse);
			fpts >> unuse >> landmark;
			if (landmark == g_TrainParam.LandmarkNumPerFace)
			{
				std::getline(fpts, unuse);
				std::getline(fpts, unuse);
				for (int32_t i = 0; i < landmark; ++i)
					fpts >> Shape(i, 0) >> Shape(i, 1);
			}
			fpts.close();
		}

		Mat_uc Image = cv::imread(ImagePathVec[i], CV_LOAD_IMAGE_GRAYSCALE);
		if (Image.empty())
			ThrowFaile;

		std::cout << ImagePathVec[i] << std::endl;

		double MinX, MinY, MaxX, MaxY;
		cv::minMaxIdx(Shape.col(0), &MinX, &MaxX);
		cv::minMaxIdx(Shape.col(1), &MinY, &MaxY);
		cv::Point2i MinPoint = cv::Point2i(static_cast<int32_t>(MinX), static_cast<int32_t>(MinY));
		cv::Point2i MaxPoint = cv::Point2i(cvCeil(MaxX), cvCeil(MaxY));

		vector<cv::Rect> FaceBoxs;
		std::ifstream fgrs(FgrPathVec[i]);
		if (fgrs.is_open())
		{
			int32_t RectNum = 0;
			fgrs >> RectNum;
			FaceBoxs.resize(RectNum);
			for (auto& var : FaceBoxs)
				fgrs >> var.x >> var.y >> var.width >> var.height;
			fgrs.close();
		}

		for (auto& var : FaceBoxs)
		{
			//scale 1.5
			double_t Scale = 1.5;

			double_t CenterX = var.x + 0.5 * var.width;
			double_t CenterY = var.y + 0.5 * var.height;

			var.x = std::max(static_cast<int32_t>(CenterX - Scale / 2 * var.width), 0);
			var.y = std::max(static_cast<int32_t>(CenterY - Scale / 2 * var.height), 0);
			var.width = std::min(static_cast<int32_t>(Scale * var.width), Image.cols - var.x);
			var.height = std::min(static_cast<int32_t>(Scale * var.height), Image.rows - var.y);

			if (var.contains(MinPoint) && var.contains(MaxPoint))
			{
				//scale Image
				Mat_uc ImageToSave = Image(var);
				cv::Rect Box = { 0,0,400,var.height * 400 / var.width };
				cv::resize(ImageToSave, ImageToSave, { Box.width,Box.height }, 0, 0, CV_INTER_AREA);

				std::lock_guard<std::mutex> Lock(mtx);
				ImageVec.push_back(ImageToSave);
				BoxVec.push_back(Box);
				TruthShape.push_back(Coordinate::Image2Box(Shape, var));
				break;
			}
		}
	}
	);
}

void FgLBFTrain::DataAugment()
{
	std::mt19937 rng;
	rng.seed(std::random_device()());
	std::uniform_int_distribution<int32_t> RandomGen(0, static_cast<int32_t>(g_ImageVec.size() - 1));
	std::uniform_int_distribution<int32_t> RotateAngleGen(0, 60);

	for (int32_t i = 0; i < g_ImageVec.size(); ++i)
	{
		for (int32_t j = 0; j < g_TrainParam.DataAugmentScale; ++j)
		{
			int32_t idx = 0;
			do
			{
				idx = RandomGen(rng);
			} while (idx == i);

			FgFaceData temp;
			temp.BoxIdx = i;
			temp.ImageIdx = i;
			temp.TruthShapeIdx = i;
			temp.CurrentShape = g_TruthShapeVec[idx].clone();

			m_FaceDataVec.push_back(temp);
		}

		for (int32_t j = 0; j < g_TrainParam.DataAugmentScale; ++j)
		{
			Mat_d RotationMat = cv::getRotationMatrix2D({ 0,0 }, RotateAngleGen(rng), 1);
			vector<cv::Point2d> TempVec;
			cv::transform(ShapeToVecPoint(g_TruthShapeVec[i]), TempVec, RotationMat);

			FgFaceData temp;
			temp.BoxIdx = i;
			temp.ImageIdx = i;
			temp.TruthShapeIdx = i;
			temp.CurrentShape = VecPointToShape(TempVec);

			m_FaceDataVec.push_back(temp);
		}

		FgFaceData temp;
		temp.BoxIdx = i;
		temp.ImageIdx = i;
		temp.TruthShapeIdx = i;
		temp.CurrentShape = g_TrainParam.MeanShape.clone();

		m_FaceDataVec.push_back(temp);
	}
}
