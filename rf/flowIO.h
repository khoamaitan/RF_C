// flowIO.h
#include "RF.h"
// the "official" threshold - if the absolute value of either 
// flow component is greater, it's considered unknown
#define UNKNOWN_FLOW_THRESH 1e9

// value to use to represent unknown flow
#define UNKNOWN_FLOW 1e10

using namespace cv;
using namespace std;
// return whether flow vector is unknown


// read a flow file into 2-band image
void ReadFlowFile(Mat &img, const char* filename);
void ReadFlowKITTIFile(Mat &img, string filename);
void ReadWeightFile(Mat& img, const char* filename);
// write a 2-band image into flow file 
void WriteFlowFile(Mat &img, const char* filename);
// Write a 1-band image into flow file
void WriteWeightFile(Mat &img, const char* filename);

