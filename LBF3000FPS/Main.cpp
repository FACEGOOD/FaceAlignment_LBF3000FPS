/*
+	LBF3000
+	A Implementation of highly efficient and very accurate regression approach for face alignment.
	Quantum Dynamics Co.,Ltd. ���Ӷ��������ڣ�������Ƽ����޹�˾

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
+	����Ŀ�������Ȩ��ѧ���о�������������ҵ����

+	(C)	Quantum Dynamics Lab. ���Ӷ���ʵ����
+		Website : http://www.facegood.cc
+		Contact Us : jelo@facegood.cc

+		-Thanks to Our Committers and Friends
+		-Best Wish to all who Contributed and Inspired
*/

/*
+	Main

+	Date:		2017/7/20
+	Author:		ZhaoHang
*/
#include "FgLBFUtil.h"
#include "FgLBFTrain.h"

using namespace std;

int main(int argc, char *argv[])
{
	try
	{
		FgLBFTrain Lbf(argv[2]);
		if (string(argv[1]) == "Train")
			Lbf.Train();
		if (string(argv[1]) == "Predict")
			Lbf.Predict(argv[3]);
	}
	catch (string& e)
	{
		cout << e << endl;
	}
	system("pause");
	return 0;
}