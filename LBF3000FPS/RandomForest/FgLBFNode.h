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

