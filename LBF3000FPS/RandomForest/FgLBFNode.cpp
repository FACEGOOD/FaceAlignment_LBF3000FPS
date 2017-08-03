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
