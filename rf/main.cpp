#include <opencv2\core.hpp>
#include <opencv2\highgui.hpp>
#include <opencv2\imgproc.hpp>
#include "RF.h"

#include <iostream>
#include <iomanip>
#include <fstream>
#include <math.h>
#include <windows.h>
#include <vector>
#include <string>


#define T (3.1415)


using namespace std;
using namespace cv;



int main()
{


	//Mat img1 = imread("frame10.png");
	//Mat img2 = imread("frame11.png");
	string pre_path = "E:\\Google Drive\\Database\\other-data\\GT\\Venus\\";
	Mat img1 = imread(pre_path+ "frame10.png");
	Mat img2 = imread(pre_path+ "frame11.png");
	//000003_11.png"
	//if (MEASURE) QueryPerformanceCounter(&t1);
	RF flow_rf(img1, img2, 2);
	//if (MEASURE) {
	//	QueryPerformanceCounter(&t2);
	//	elapsedTime = (t2.QuadPart - t1.QuadPart) * 1000.0 / frequency.QuadPart;
	//	printf("Initialize pyramidal images: %.2f ms\n", elapsedTime);
	//}
	Mat *uv;
	Mat *colflow= NULL;

	uv = flow_rf.find_flow();
	MotionToColor(uv, colflow,0);
	imshow("Color KLT", (*colflow));
	cv::waitKey(0);
}

