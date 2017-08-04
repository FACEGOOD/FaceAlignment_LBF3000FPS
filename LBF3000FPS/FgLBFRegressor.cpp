#include "FgLBFRegressor.h"
#include <mutex>
#include <ppl.h>

#pragma comment(lib,"liblinear.lib")

static std::mutex mtx;
static int32_t LogCount = 1;

FgLBFRegressor::FgLBFRegressor(vector<FgFaceData>& FaceDataVec)
	:m_FaceDataVec(FaceDataVec)
{
}

FgLBFRegressor::~FgLBFRegressor()
{
}

vector<Mat_d> FgLBFRegressor::Train(int32_t Stage)
{
	vector<Mat_d> RegressionTarget(m_FaceDataVec.size());

	//Concurrency::parallel_for(0, static_cast<int32_t>(m_FaceDataVec.size()), [&, this](int32_t i)
	//{
	//	m_FaceDataVec[i].ToMeanShape = cv::estimateAffine2D(m_FaceDataVec[i].CurrentShape, g_TrainParam.MeanShape, cv::noArray(), cv::LMEDS);
	//	m_FaceDataVec[i].MeanShapeTo = cv::estimateAffine2D(g_TrainParam.MeanShape, m_FaceDataVec[i].CurrentShape, cv::noArray(), cv::LMEDS);

	//	RegressionTarget[i] = g_TruthShapeVec[m_FaceDataVec[i].TruthShapeIdx] - m_FaceDataVec[i].CurrentShape;
	//	vector<cv::Point2d> TempVec;
	//	cv::transform(ShapeToVecPoint(RegressionTarget[i]), TempVec, m_FaceDataVec[i].ToMeanShape);
	//	RegressionTarget[i] = VecPointToShape(TempVec);
	//});

	Concurrency::parallel_for(0, static_cast<int32_t>(m_FaceDataVec.size()), [&, this](int32_t i)
	{
		m_FaceDataVec[i].ToMeanShape = FgGetAffineTransform(m_FaceDataVec[i].CurrentShape, g_TrainParam.MeanShape);
		m_FaceDataVec[i].MeanShapeTo = FgGetAffineTransform(g_TrainParam.MeanShape, m_FaceDataVec[i].CurrentShape);

		m_FaceDataVec[i].ToMeanShape(0, 2) = m_FaceDataVec[i].ToMeanShape(1, 2) = 0.0;
		m_FaceDataVec[i].MeanShapeTo(0, 2) = m_FaceDataVec[i].MeanShapeTo(1, 2) = 0.0;

		RegressionTarget[i] = g_TruthShapeVec[m_FaceDataVec[i].TruthShapeIdx] - m_FaceDataVec[i].CurrentShape;
		vector<cv::Point2d> TempVec;
		cv::transform(ShapeToVecPoint(RegressionTarget[i]), TempVec, m_FaceDataVec[i].ToMeanShape);
		RegressionTarget[i] = VecPointToShape(TempVec);
	});

	for (int32_t i = 0; i < g_TrainParam.LandmarkNumPerFace; ++i)
		m_RandomForestVec.push_back(FgLBFRandomForest(m_FaceDataVec, Stage, i));

	LogCount = 1;
	Concurrency::parallel_for(0, static_cast<int32_t>(m_RandomForestVec.size()), [&](int32_t i)
	{
		m_RandomForestVec[i].TrainForest(RegressionTarget);
		std::lock_guard<std::mutex> lock(mtx);
		std::cout << "Landmark Forest:[" << LogCount++ << "/" << g_TrainParam.LandmarkNumPerFace << "]" << std::endl;
	}
	);

	std::cout << "Get Global Binary Features" << std::endl;

	feature_node **GlobalBinaryFeatures = new feature_node*[m_FaceDataVec.size()];

	for (int32_t i = 0; i < m_FaceDataVec.size(); ++i)
		GlobalBinaryFeatures[i] = new feature_node[g_TrainParam.TreeNumPerForest * g_TrainParam.LandmarkNumPerFace + 1];


	int32_t NumFeature = 0;
	for (int32_t i = 0; i < g_TrainParam.LandmarkNumPerFace; ++i)
		NumFeature += m_RandomForestVec[i].GetLeafNodesNum();

	concurrency::parallel_for(0, static_cast<int32_t>(m_FaceDataVec.size()), [&](int32_t i)
	{
		int32_t Ind = 0;
		int32_t Index = 1;

		const FgFaceData& FaceData = m_FaceDataVec[i];

		for (int32_t Landmark = 0; Landmark < g_TrainParam.LandmarkNumPerFace; ++Landmark)
		{
			for (int32_t Tree = 0; Tree < g_TrainParam.TreeNumPerForest; ++Tree)
			{
				FgLBFNode* Node = m_RandomForestVec[Landmark].m_TreeVec[Tree];

				while (!Node->m_IsLeaf)
				{
					Mat_d RandomPoints = Mat_d::zeros(2, 2);

					cv::Point2d A = Node->m_FeatureLocations.first;
					cv::Point2d B = Node->m_FeatureLocations.second;

					vector<cv::Point2d> Vec{ A,B };
					cv::transform(Vec, Vec, FaceData.MeanShapeTo);

					RandomPoints(0, 0) = Vec[0].x + FaceData.CurrentShape(Landmark, 0);
					RandomPoints(0, 1) = Vec[0].y + FaceData.CurrentShape(Landmark, 1);
					RandomPoints(1, 0) = Vec[1].x + FaceData.CurrentShape(Landmark, 0);
					RandomPoints(1, 1) = Vec[1].y + FaceData.CurrentShape(Landmark, 1);

					Mat_d ImagePoints = Coordinate::Box2Image(RandomPoints, g_BoxVec[m_FaceDataVec[i].BoxIdx]);
					auto& Image = g_ImageVec[m_FaceDataVec[i].ImageIdx];

					for (int32_t i = 0; i < 2; ++i)
					{
						if (ImagePoints(i, 0) >= Image.cols - 1)
							ImagePoints(i, 0) = Image.cols - 1;
						if (ImagePoints(i, 1) >= Image.rows - 1)
							ImagePoints(i, 1) = Image.rows - 1;
					}

					int32_t F = Image(static_cast<int32_t>(ImagePoints(0, 1)), static_cast<int32_t>(ImagePoints(0, 0))) - Image(static_cast<int32_t>(ImagePoints(1, 1)), static_cast<int32_t>(ImagePoints(1, 0)));
					if (F < Node->m_Threshold)
						Node = Node->m_LeftChild;
					else
						Node = Node->m_RightChild;
				}

				GlobalBinaryFeatures[i][Ind].index = Index + Node->m_LeafIdentity;
				GlobalBinaryFeatures[i][Ind].value = 1.0;
				Ind++;
			}
			Index += m_RandomForestVec[Landmark].GetLeafNodesNum();
		}

		GlobalBinaryFeatures[i][g_TrainParam.TreeNumPerForest * g_TrainParam.LandmarkNumPerFace].index = -1;
		GlobalBinaryFeatures[i][g_TrainParam.TreeNumPerForest * g_TrainParam.LandmarkNumPerFace].value = -1.0;
	});

	parameter* RegressionParams = new parameter();
	RegressionParams->solver_type = L2R_L2LOSS_SVR_DUAL;
	RegressionParams->C = 1.0 / m_FaceDataVec.size();
	RegressionParams->p = 0;
	RegressionParams->eps = 0.1;

	m_LinearModelX.resize(g_TrainParam.LandmarkNumPerFace);
	m_LinearModelY.resize(g_TrainParam.LandmarkNumPerFace);

	double** Targets = new double*[g_TrainParam.LandmarkNumPerFace];
	for (int32_t i = 0; i < g_TrainParam.LandmarkNumPerFace; ++i)
	{
		Targets[i] = new double[m_FaceDataVec.size()];
	}

	LogCount = 1;
	concurrency::parallel_for(0, g_TrainParam.LandmarkNumPerFace, [&](int32_t i)
	{
		problem* prob = new problem();
		prob->l = static_cast<int32_t>(m_FaceDataVec.size());
		prob->n = NumFeature + 1;
		prob->x = GlobalBinaryFeatures;
		prob->bias = 1;

		for (int32_t j = 0; j < m_FaceDataVec.size(); j++)
			Targets[i][j] = RegressionTarget[j](i, 0);
		prob->y = Targets[i];
		check_parameter(prob, RegressionParams);
		model* RegressionModel = train(prob, RegressionParams);
		m_LinearModelX[i] = RegressionModel;

		for (int32_t j = 0; j < m_FaceDataVec.size(); j++)
			Targets[i][j] = RegressionTarget[j](i, 1);
		prob->y = Targets[i];
		check_parameter(prob, RegressionParams);
		RegressionModel = train(prob, RegressionParams);
		m_LinearModelY[i] = RegressionModel;

		delete prob;
	}
	);

	for (int32_t i = 0; i < g_TrainParam.LandmarkNumPerFace; ++i) {
		delete[] Targets[i];
	}
	delete[] Targets;

	vector<Mat_d> PredictRegressionTargets(m_FaceDataVec.size());
	concurrency::parallel_for(0, static_cast<int32_t>(m_FaceDataVec.size()), [&](int32_t i)
	{
		Mat_d Temp(g_TrainParam.LandmarkNumPerFace, 2, 0.0);

		for (int32_t j = 0; j < g_TrainParam.LandmarkNumPerFace; ++j)
		{
			Temp(j, 0) = predict(m_LinearModelX[j], GlobalBinaryFeatures[i]);
			Temp(j, 1) = predict(m_LinearModelY[j], GlobalBinaryFeatures[i]);
		}

		vector<cv::Point2d> TempVec;
		cv::transform(ShapeToVecPoint(Temp), TempVec, m_FaceDataVec[i].MeanShapeTo);
		PredictRegressionTargets[i] = VecPointToShape(TempVec);
	});


	return PredictRegressionTargets;
}

Mat_d FgLBFRegressor::Predict(Mat_uc& Image, Mat_d& CurrentShape, cv::Rect& Box, Mat_d& MeanShapeTo)
{
	static feature_node* Features = new feature_node[g_TrainParam.TreeNumPerForest * g_TrainParam.LandmarkNumPerFace + 1]();

	vector<int32_t> IndexVec(g_TrainParam.LandmarkNumPerFace);
	IndexVec[0] = 1;
	for (int32_t Landmark = 0; Landmark < g_TrainParam.LandmarkNumPerFace - 1; ++Landmark)
		IndexVec[Landmark + 1] = IndexVec[Landmark] + m_RandomForestVec[Landmark].GetLeafNodesNum();

	concurrency::parallel_for(0, g_TrainParam.LandmarkNumPerFace, [&](int32_t Landmark)
	{
		for (int32_t Tree = 0; Tree < g_TrainParam.TreeNumPerForest; ++Tree)
		{
			FgLBFNode* Node = m_RandomForestVec[Landmark].m_TreeVec[Tree];

			while (!Node->m_IsLeaf)
			{
				Mat_d RandomPoints = Mat_d::zeros(2, 2);

				vector<cv::Point2d> Vec{ Node->m_FeatureLocations.first,Node->m_FeatureLocations.second };
				cv::transform(Vec, Vec, MeanShapeTo);

				RandomPoints(0, 0) = Vec[0].x + CurrentShape(Landmark, 0);
				RandomPoints(0, 1) = Vec[0].y + CurrentShape(Landmark, 1);
				RandomPoints(1, 0) = Vec[1].x + CurrentShape(Landmark, 0);
				RandomPoints(1, 1) = Vec[1].y + CurrentShape(Landmark, 1);

				Mat_d ImagePoints = Coordinate::Box2Image(RandomPoints, Box);

				for (int32_t i = 0; i < 2; ++i)
				{
					if (ImagePoints(i, 0) >= Image.cols - 1)
						ImagePoints(i, 0) = Image.cols - 1;
					if (ImagePoints(i, 1) >= Image.rows - 1)
						ImagePoints(i, 1) = Image.rows - 1;
				}

				int32_t F = Image(static_cast<int32_t>(ImagePoints(0, 1)), static_cast<int32_t>(ImagePoints(0, 0))) - Image(static_cast<int32_t>(ImagePoints(1, 1)), static_cast<int32_t>(ImagePoints(1, 0)));
				if (F < Node->m_Threshold)
					Node = Node->m_LeftChild;
				else
					Node = Node->m_RightChild;
			}
			Features[Tree + Landmark * g_TrainParam.TreeNumPerForest].index = IndexVec[Landmark] + Node->m_LeafIdentity;
			Features[Tree + Landmark * g_TrainParam.TreeNumPerForest].value = 1.0;
		}
	}
	);


	Features[g_TrainParam.TreeNumPerForest * g_TrainParam.LandmarkNumPerFace].index = -1;
	Features[g_TrainParam.TreeNumPerForest * g_TrainParam.LandmarkNumPerFace].value = -1.0;

	static vector<cv::Point2d> PredictDeltaVec(g_TrainParam.LandmarkNumPerFace);
	concurrency::parallel_for(0, g_TrainParam.LandmarkNumPerFace, [&](int32_t Landmark)
	{
		PredictDeltaVec[Landmark].x = predict(m_LinearModelX[Landmark], Features);
		PredictDeltaVec[Landmark].y = predict(m_LinearModelY[Landmark], Features);
	});


	vector<cv::Point2d> TempVec;
	cv::transform(PredictDeltaVec, TempVec, MeanShapeTo);
	return VecPointToShape(TempVec);
}

void FgLBFRegressor::LoadFromPath(string Path, int32_t Idx)
{
	m_RandomForestVec.resize(g_TrainParam.LandmarkNumPerFace, FgLBFRandomForest(vector<FgFaceData>(), -1, 0));
	m_LinearModelX.resize(g_TrainParam.LandmarkNumPerFace);
	m_LinearModelY.resize(g_TrainParam.LandmarkNumPerFace);

	string TreeName = Path + "R_" + std::to_string(Idx) + "Tree";
	concurrency::parallel_for(0, g_TrainParam.LandmarkNumPerFace, [&](int32_t i)
	{
		std::ifstream OfTree(TreeName + std::to_string(i));
		OfTree >> m_RandomForestVec[i];
		OfTree.close();
	});

	string ModelName = Path + "R_" + std::to_string(Idx) + "Model";
	concurrency::parallel_for(0, g_TrainParam.LandmarkNumPerFace, [&](int32_t i)
	{
		m_LinearModelX[i] = load_model((ModelName + "X" + std::to_string(i)).c_str());
		m_LinearModelY[i] = load_model((ModelName + "Y" + std::to_string(i)).c_str());
	});
}

void FgLBFRegressor::SaveToPath(string Path, int32_t Idx)
{
	string TreeName = Path + "R_" + std::to_string(Idx) + "Tree";

	for (int32_t i = 0; i < m_RandomForestVec.size(); ++i)
	{
		std::ofstream OfTree(TreeName + std::to_string(i));
		OfTree << m_RandomForestVec[i];
		OfTree.close();
	}

	string ModelName = Path + "R_" + std::to_string(Idx) + "Model";

	for (int32_t i = 0; i < m_LinearModelX.size(); ++i)
	{
		save_model((ModelName + "X" + std::to_string(i)).c_str(), m_LinearModelX[i]);
	}

	for (int32_t i = 0; i < m_LinearModelY.size(); ++i)
	{
		save_model((ModelName + "Y" + std::to_string(i)).c_str(), m_LinearModelY[i]);
	}
}