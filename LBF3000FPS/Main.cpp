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