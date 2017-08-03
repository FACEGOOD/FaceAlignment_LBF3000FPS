#pragma once

#pragma warning(disable:4996)

#include <iostream>
#include <vector>
#include <random>
#include <opencv2/opencv.hpp>

using namespace std;
typedef cv::Mat_<uchar> Mat_uc;

#ifdef _DEBUG
#pragma comment(lib,"opencv_world320d.lib")
#else
#pragma comment(lib,"opencv_world320.lib")
#endif // _DEBUG

#define ThrowFaile throw (std::string(__FILE__) + "[line: " + std::to_string(__LINE__) + " ]")

int main(int argc, char *argv[])
{
	if (argc != 2)
		ThrowFaile;

	std::ifstream fin(argv[1]);
	if (!fin.is_open())
		ThrowFaile;

	vector<string> ImagePathVec;
	vector<string> FgrPathVec;

	string ImagePath;
	while (std::getline(fin, ImagePath))
	{
		ImagePathVec.push_back(ImagePath);

		string FgrPath = ImagePath;
		auto Pos = FgrPath.find_last_of('.');
		FgrPath.replace(Pos, FgrPath.length() - 1, ".fgr");

		FgrPathVec.push_back(FgrPath);
	}
	fin.close();

	cv::CascadeClassifier Cc;
	if (!Cc.load("../OpenCV/etc/haarcascades/haarcascade_frontalface_alt.xml"))
		ThrowFaile;

	for (int32_t i = 0; i < ImagePathVec.size(); ++i)
	{
		Mat_uc Image = cv::imread(ImagePathVec[i], CV_LOAD_IMAGE_GRAYSCALE);
		if (Image.empty())
			ThrowFaile;

		vector<cv::Rect> FaceBoxs;
		std::ofstream oFgr(FgrPathVec[i]);
		Cc.detectMultiScale(Image, FaceBoxs);

		oFgr << FaceBoxs.size() << endl;
		for (auto& var : FaceBoxs)
			oFgr << var.x << " " << var.y << " " << var.width << " " << var.height << endl;
		oFgr.flush();
		oFgr.close();
	}
	return 0;
}