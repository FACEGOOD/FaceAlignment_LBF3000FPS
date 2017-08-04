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

+	Date:		2017/7/16
+	Author:		ZhaoHang
*/

#include "FgLBFNode.h"


FgLBFNode::FgLBFNode(FgLBFNode * left, FgLBFNode * right, double_t thres)
	:FgLBFNode(left, right, thres, false)
{
}

FgLBFNode::FgLBFNode(FgLBFNode * left, FgLBFNode * right, double_t thres, bool isLeaf)
{
	m_LeftChild = left;
	m_RightChild = right;
	m_IsLeaf = isLeaf;
	m_Threshold = thres;
}

std::ofstream & operator<<(std::ofstream & Out, FgLBFNode & Obj)
{
	Out << Obj.m_Depth << std::endl;
	Out << Obj.m_FeatureLocations.first.x << std::endl;
	Out << Obj.m_FeatureLocations.first.y << std::endl;
	Out << Obj.m_FeatureLocations.second.x << std::endl;
	Out << Obj.m_FeatureLocations.second.y << std::endl;
	Out << Obj.m_LeafIdentity << std::endl;
	Out << Obj.m_Threshold << std::endl;
	Out << Obj.m_IsLeaf << std::endl;

	if (!Obj.m_IsLeaf)
	{
		Out << *Obj.m_LeftChild;
		Out << *Obj.m_RightChild;
	}

	return Out;
}

std::ifstream & operator>>(std::ifstream & In, FgLBFNode & Obj)
{
	In >> Obj.m_Depth;
	In >> Obj.m_FeatureLocations.first.x;
	In >> Obj.m_FeatureLocations.first.y;
	In >> Obj.m_FeatureLocations.second.x;
	In >> Obj.m_FeatureLocations.second.y;
	In >> Obj.m_LeafIdentity;
	In >> Obj.m_Threshold;
	In >> Obj.m_IsLeaf;

	if (!Obj.m_IsLeaf)
	{
		Obj.m_LeftChild = new FgLBFNode();
		Obj.m_RightChild = new FgLBFNode();

		In >> *Obj.m_LeftChild;
		In >> *Obj.m_RightChild;
	}

	return In;
}
