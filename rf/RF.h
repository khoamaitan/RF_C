#ifndef RF_HEADER
#define RF_HEADER
#include <opencv2\core.hpp>
#include <opencv2\highgui.hpp>
#include <opencv2\imgproc.hpp>
#include "colorcode.h"

#include <iostream>
#include <iomanip>
#include <fstream>
#include <math.h>
#include <windows.h>
#include <vector>
#include <string>

#define VERBOSE 1
#define UNKNOWN_FLOW_THRESH 1e9
#define MEASURE 0


using namespace std;
using namespace cv;

class RF
{
public:
	RF(Mat _img1, Mat _img2, float pyr_space);
	void show_pyr();
	Mat* find_flow();
	Mat *find_flow2();
	void partial_derivation(Mat*Ix, Mat*Iy, Mat*It, Mat* warpImc, Mat*pimg1, Mat*pimg2, Mat*uv, bool fusion);
	void KLT(Mat*uv_res, Mat* w_corner, Mat* iD, Mat* Ix, Mat* Iy, Mat* It, int w_size);
	void KLT2(Mat*uv_res, Mat* w_corner, Mat* iD, Mat* Ix, Mat* Iy, Mat* It, int w_size,int eig);
	void eval_resuv(Mat *uv_res, Mat *conv, const Mat *iD, const Mat *w_corner, const int w_size, const Mat *Ix, const Mat *Iy, const Mat *It);
	void propagation(const Mat*scene, Mat* conv, Mat*uv, const Mat*uv_res);
	void rescale_flow(Mat *uv);
	Mat eval_var(Mat* &uv_res, const int &w_size);
	Mat eval_corner(const Mat* w_corner, const int &w_size);
	Mat eval_mix(Mat &w_corner, Mat &w_var, Mat* iD);
	Mat* getuv();
private:
	int pyr_lvl;
	int H, W;
	int mH, mW;
	Mat** p_img1;
	Mat** p_img2;
	Mat** p_img1g;
	Mat** p_img2g;
	Mat* uv=nullptr;
	Mat kernel_der, XY;
	//Mat I1x, I1y, I2x, I2y;
	char median_filter_size = 5;
	char nb_warp = 12;
	int w_klt = 5;
	void limit_res(Mat* uv_res);

};

void meshgrid(const cv::Range &xgv, const cv::Range &ygv, cv::Mat &X, cv::Mat &Y);
void MotionToColor(Mat* motim, Mat* &colim, float maxmotion);
SparseMat create_data(const Mat* pimg, Mat* &sum_taux);// Using sparse matrix
float e_simi(const Mat* pimg, int row, int tg, int ng);
Mat create_data2(const Mat* pimg, float* &array_data, float*&array_sum);//Using index matrix
void update_neighbor(const int &idx_point, const int &i, const int &j, const int &w, const int &W, const int code, const Mat* &pimg, int &i_arr, float* &array_data, float* &array_sum, int* &p_sum, int* &p_n, int* &p_idxn);
void verify_data(const Mat &pixel_info, float* &sum_rate, const int &W);
Mat display_flow_quiver(Mat img_display, Mat &flow);
void cvQuiver(Mat& Image, int x, int y, int u, int v, Scalar Color, int Size, int Thickness);
inline bool unknown_flow(float u, float v)
{
	return (fabs(u) >  UNKNOWN_FLOW_THRESH)
		|| (fabs(v) >  UNKNOWN_FLOW_THRESH)
		|| isnan(u) || isnan(v);
}
inline bool unknown_flow(float *f) {
	return unknown_flow(f[0], f[1]);
}
#endif