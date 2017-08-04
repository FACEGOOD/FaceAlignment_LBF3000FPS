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
+	Node in RandomForest

+	Date:		2017/7/15
+	Author:		ZhaoHang
*/
#pragma once
#include "../FgLBFUtil.h"

typedef std::pair<cv::Point2d, cv::Point2d> FeatureLocation;

class FgLBFNode
{
	friend std::ofstream & operator << (std::ofstream &Out, FgLBFNode &Obj);
	friend std::ifstream & operator >> (std::ifstream &In, FgLBFNode &Obj);
public:
	FgLBFNode() = default;
	~FgLBFNode() = default;

	FgLBFNode(FgLBFNode* left, FgLBFNode* right, double_t thres);
	FgLBFNode(FgLBFNode* left, FgLBFNode* right, double_t thres , bool isLeaf);

	FgLBFNode*		m_LeftChild = NULL;
	FgLBFNode*		m_RightChild = NULL;

	int32_t			m_Depth = 0;
	int32_t			m_LeafIdentity = -1;

	double_t		m_Threshold = 0.0;
	bool			m_IsLeaf = false;

	FeatureLocation	m_FeatureLocations;
};

