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
+	Main
+	Input some images ,Out *.fgr.

+	Date:		2017/7/28
+	Author:		ZhaoHang
*/
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