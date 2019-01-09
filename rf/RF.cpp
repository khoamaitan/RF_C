#include "RF.h"

RF::RF(Mat _img1, Mat _img2, float pyr_space)
{
	_img1.convertTo(_img1, CV_32FC3);
	_img2.convertTo(_img2, CV_32FC3);
	H = _img1.rows;
	W = _img1.cols;
	/**********************Creating pyramidal images**************************/
	pyr_lvl = 1 + floor(log(min(H, W) / 16) / log(pyr_space)); // Determine number of gaussian level
															   //pyr_lvl = 2;
	if (VERBOSE) printf("Max pyramid level: %d \n", pyr_lvl);
	float smooth_sigma = sqrt(pyr_space) / sqrt(2); // Sigma of gaussian filter
	int w = 2 * round(1.5*smooth_sigma) + 1; // Window size of gaussian filter

	p_img1 = new Mat*[pyr_lvl];
	p_img2 = new Mat*[pyr_lvl];
	p_img1[0] = new Mat(_img1);
	p_img2[0] = new Mat(_img2);
	for (int i = 1;i < pyr_lvl;i++)
	{
		Mat temp, temp2;
		pyrDown(*p_img1[i - 1], temp);
		p_img1[i] = new Mat(temp);

		pyrDown(*p_img2[i - 1], temp2);
		p_img2[i] = new Mat(temp2);

	}
	/*************************************************************************/
	kernel_der = (Mat_<float>(1, 5) << 1.0f / 12, -8.0f / 12, 0, 8.0f / 12, -1.0f / 12); // Kernel for derivation

}

void RF::show_pyr()
{
	for (int i = 0;i < pyr_lvl;i++)
	{
		imshow("Pyr " + to_string(i), **(p_img1 + i)/255.0f);
	}
}
// Main function of finding the optical flow
Mat * RF::find_flow()
{
	int i, k;
	Mat *Ix, *Iy, *It, *warpImc, *uv_res;
	Mat *iD, *w_corner, *conv=nullptr;
	Mat *pimg1, *pimg2;
	Mat X, Y;
	/******************************Start Pyramidal level***********************/
	for (i = pyr_lvl - 1;i >= 0;i--)
	{
		printf("Level: %d \n", i);

		pimg1 = p_img1[i];
		pimg2 = p_img2[i];
		mH = pimg1->rows;
		mW = pimg1->cols;
		Ix = new Mat(mH, mW, CV_32F);
		Iy = new Mat(mH, mW, CV_32F);
		It = new Mat(mH, mW, CV_32F);
		iD = new Mat(mH, mW, CV_8U);
		w_corner = new Mat(mH, mW, CV_32F);
		uv_res = new Mat(mH, mW, CV_32FC2);
		warpImc = new Mat(mH, mW, CV_32FC3);
		conv = new Mat(mH, mW, CV_32F);


		meshgrid(Range(0, mW - 1), Range(0, mH - 1), X, Y);
		vector<Mat> ch;
		ch.push_back(X);
		ch.push_back(Y);
		merge(ch, XY);

		if (uv != nullptr)
			rescale_flow(uv);
		else
			uv = new Mat(mH, mW, CV_32FC2, Scalar(0, 0));
		/**************************Start Warping********************************/

		for (k = 0;k < nb_warp;k++)
		{
			printf("Warp: %d\n", k);
			*uv_res = 0;
			*iD = 0;
			//if (MEASURE) QueryPerformanceCounter(&t1);
			partial_derivation(Ix, Iy, It, warpImc, pimg1, pimg2, uv, 1);
			//if (MEASURE) {
			//	QueryPerformanceCounter(&t2);
			//	elapsedTime = (t2.QuadPart - t1.QuadPart) * 1000.0 / frequency.QuadPart;
			//	printf("Derivation at lvl %d warp %d: %.2f ms\n", i, k, elapsedTime);
			//	QueryPerformanceCounter(&t1);
			//}
			KLT(uv_res, w_corner, iD, Ix, Iy, It, w_klt);
			/*if (MEASURE) {
				QueryPerformanceCounter(&t2);
				elapsedTime = (t2.QuadPart - t1.QuadPart) * 1000.0 / frequency.QuadPart;
				printf("KLT at lvl %d warp %d: %.2f ms\n", i, k, elapsedTime);
			}*/
			limit_res(uv_res);
			*uv = *uv + *uv_res;
			//Evaluate flow:
			eval_resuv(uv_res, conv, iD, w_corner, 5, Ix, Iy, It);
			//Pre-refine flow:
			//Propagate flow:
			//if(i==0)
			propagation(pimg1, conv, uv, uv_res);
			//else
			//		*uv = *uv + *uv_res;
			medianBlur(*uv, *uv, median_filter_size);
			medianBlur(*uv, *uv, median_filter_size);
		}
		//Ix->release();
		//Iy->release();
		//It->release();
		//uv_res->release();
		//warpImc->release();
		//conv->release();
		//w_corner->release();
		delete Ix;
		delete Iy;
		delete It;
		delete uv_res;
		delete warpImc;
		delete iD;
		delete w_corner;
		if (i > 0)
		delete conv;
		//I1x.release();
		//I1y.release();
		//I2x.release();
		//I2y.release();
		//Display after each level
		//Mat *colflow = NULL;
		//MotionToColor(uv, colflow, 0);
		//imshow("Color KLT", (*colflow));
		//cvWaitKey(0);
		//delete colflow;
	}

	//return conv;
	return uv;
}

Mat * RF::find_flow2()
{
	int i, k;
	Mat *Ix, *Iy, *It, *warpImc, *uv_res;
	Mat *iD, *w_corner, *conv = nullptr,*w_var;
	Mat *pimg1, *pimg2;
	Mat X, Y;
	/******************************Start Pyramidal level***********************/
	for (i = pyr_lvl - 1; i >= 0; i--)
	{
		printf("Level: %d \n", i);

		pimg1 = p_img1[i];
		pimg2 = p_img2[i];
		mH = pimg1->rows;
		mW = pimg1->cols;
		Ix = new Mat(mH, mW, CV_32F);
		Iy = new Mat(mH, mW, CV_32F);
		It = new Mat(mH, mW, CV_32F);
		iD = new Mat(mH, mW, CV_8U);
		w_corner = new Mat(mH, mW, CV_32F);
		uv_res = new Mat(mH, mW, CV_32FC2);
		warpImc = new Mat(mH, mW, CV_32FC3);
		conv = new Mat(mH, mW, CV_32F);
		w_var = new Mat(mH, mW, CV_32F);

		meshgrid(Range(0, mW - 1), Range(0, mH - 1), X, Y);
		vector<Mat> ch;
		ch.push_back(X);
		ch.push_back(Y);
		merge(ch, XY);

		if (uv != nullptr)
			rescale_flow(uv);
		else
			uv = new Mat(mH, mW, CV_32FC2, Scalar(0, 0));
		/**************************Start Warping********************************/

		for (k = 0; k < nb_warp; k++)
		{
			printf("Warp: %d\n", k);
			*uv_res = 0;
			*iD = 0;
			//if (MEASURE) QueryPerformanceCounter(&t1);
			partial_derivation(Ix, Iy, It, warpImc, pimg1, pimg2, uv, 1);
			//if (MEASURE) {
			//	QueryPerformanceCounter(&t2);
			//	elapsedTime = (t2.QuadPart - t1.QuadPart) * 1000.0 / frequency.QuadPart;
			//	printf("Derivation at lvl %d warp %d: %.2f ms\n", i, k, elapsedTime);
			//	QueryPerformanceCounter(&t1);
			//}
			KLT2(uv_res, w_corner, iD, Ix, Iy, It, w_klt, !((k+1)%4));
			/*if (MEASURE) {
			QueryPerformanceCounter(&t2);
			elapsedTime = (t2.QuadPart - t1.QuadPart) * 1000.0 / frequency.QuadPart;
			printf("KLT at lvl %d warp %d: %.2f ms\n", i, k, elapsedTime);
			}*/
			limit_res(uv_res);
			*uv = *uv + *uv_res;
			//Evaluate flow:
			if (!((k+1)%4))
			{ 
				Mat w1 = eval_corner(w_corner, w_klt);
				Mat w2 = eval_var(uv, w_klt);
				Mat w_mix = eval_mix(w1, w2, iD);
			// Create patches:

			// Propagation:

			//		*uv = *uv + *uv_res;
			medianBlur(*uv, *uv, median_filter_size);
			medianBlur(*uv, *uv, median_filter_size);
		
			}
		}
		//Ix->release();
		//Iy->release();
		//It->release();
		//uv_res->release();
		//warpImc->release();
		//conv->release();
		//w_corner->release();
		delete Ix;
		delete Iy;
		delete It;
		delete uv_res;
		delete warpImc;
		delete iD;
		delete w_corner;
		delete w_var;
		if (i > 0)
			delete conv;
		//I1x.release();
		//I1y.release();
		//I2x.release();
		//I2y.release();
		//Display after each level
		//Mat *colflow = NULL;
		//MotionToColor(uv, colflow, 0);
		//imshow("Color KLT", (*colflow));
		//cvWaitKey(0);
		//delete colflow;
	}

	//return conv;
	return uv;
}

void RF::partial_derivation(Mat * Ix, Mat * Iy, Mat * It, Mat * warpImc, Mat * pimg1, Mat * pimg2, Mat * uv, bool fusion)
{
	int i, j;
	Mat uv2 = Mat::zeros(mH, mW, CV_32FC2);
	Mat I1x = Mat(mH, mW, CV_32FC3);
	Mat I1y = Mat(mH, mW, CV_32FC3);
	Mat I2x = Mat(mH, mW, CV_32FC3);
	Mat I2y = Mat(mH, mW, CV_32FC3);
	uv2 = *uv + XY; // Find new position for x,y
					// Limit the border
	float* puv2 = (float*)uv2.data;
	char stepW = uv2.dims;
	for (i = 0;i < mH;i++)
		for (j = 0;j < mW;j++)
		{
			int idx = j*stepW + i*mW*stepW;
			if ((*(puv2 + idx)) < 0) *(puv2 + idx) = 0;
			if ((*(puv2 + idx)) > (mW - 1)) *(puv2 + idx) = mW - 1;
			if ((*(puv2 + idx + 1)) < 0) *(puv2 + idx + 1) = 0;
			if ((*(puv2 + idx + 1)) > (mH - 1)) *(puv2 + idx + 1) = mH - 1;
		}
	//if (VERBOSE) cout << "Kernel: \n" << uv2 << endl;
	// Find derivation of I2
	cv::filter2D(*pimg2, I2x, CV_32F, kernel_der);
	cv::filter2D(*pimg2, I2y, CV_32F, kernel_der.t());
	//Find Ix,Iy by interpolation and It by subtraction
	cv::remap(*pimg2, *warpImc, uv2, Mat(), INTER_CUBIC);
	//imshow("WarpImage", *warpImc/255);
	//imshow("Originale", *pimg1 / 255);
	//cvWaitKey(0);
	cv::remap(I2x, *Ix, uv2, Mat(), INTER_CUBIC);
	cv::remap(I2y, *Iy, uv2, Mat(), INTER_CUBIC);

	//It->convertTo(*It, CV_32F);
	// Fusion the gradient of warpped image and the one of I1
	if (fusion)
	{
		cv::filter2D(*pimg1, I1x, CV_32F, kernel_der);
		cv::filter2D(*pimg1, I1y, CV_32F, kernel_der.t());
		*Ix = (0.5f)*(*Ix) + (0.5f)*I1x;
		*Iy = (0.5f)*(*Iy) + (0.5f)*I1y;
		(*warpImc) = 0.5f*(*warpImc) + 0.5f*(*pimg1);
		*It = (*warpImc) - (*pimg1);
		I1x.release();
		I2x.release();
		I1y.release();
		I2y.release();
		return;
	}
	*It = (*warpImc) - (*pimg1);
	I1x.release();
	I2x.release();
	I1y.release();
	I2y.release();
	return;
}

/**********************************Compute KLT residual value******************
*uv_res,w_corner,iD has to be allocated before used                           */
void RF::KLT(Mat * uv_res, Mat* w_corner, Mat* iD, Mat * Ix, Mat * Iy, Mat * It, int w_size)
{
	int i, j, idx;
	float fD;
	Mat Ixx, Ixy, Iyy, Ixt, Iyt, kernel;
	Mat IIxx, IIxy, IIyy, IIxt, IIyt;
	kernel = Mat::ones(w_size, w_size, CV_32F);
	//Multiply to create value for hessian matrix
	cv::multiply(*Ix, *Ix, Ixx);
	cv::multiply(*Ix, *Iy, Ixy);
	cv::multiply(*Iy, *Iy, Iyy);
	cv::multiply(*Ix, *It, Ixt);
	cv::multiply(*Iy, *It, Iyt);
	//Sum around the neighbor of each point to create hessian matrix for each point
	cv::filter2D(Ixx, Ixx, CV_32F, kernel);
	cv::filter2D(Ixy, Ixy, CV_32F, kernel);
	cv::filter2D(Ixt, Ixt, CV_32F, kernel);
	cv::filter2D(Iyy, Iyy, CV_32F, kernel);
	cv::filter2D(Iyt, Iyt, CV_32F, kernel);
	//Sum 3 channels R+G+B
	cv::transform(Ixx, IIxx, Matx13f(1, 1, 1));
	cv::transform(Ixy, IIxy, Matx13f(1, 1, 1));
	cv::transform(Ixt, IIxt, Matx13f(1, 1, 1));
	cv::transform(Iyy, IIyy, Matx13f(1, 1, 1));
	cv::transform(Iyt, IIyt, Matx13f(1, 1, 1));

	Mat D(mH, mW, CV_32F);//Det(A) matrix
	float* pD = (float*)D.data;
	float* pIxx = (float*)IIxx.data;
	float* pIxy = (float*)IIxy.data;
	float* pIxt = (float*)IIxt.data;
	float* pIyy = (float*)IIyy.data;
	float* pIyt = (float*)IIyt.data;
	float* puv_res = (float*)uv_res->data; // H*W*2
	float* pw_corner = (float*)w_corner->data;//H*W
	uchar* pID = (uchar*)iD->data;
	float alpha = 0.1;
	for (i = 0;i < mH;i++)
		for (j = 0;j < mW;j++)
		{
			idx = i*mW + j;
			pD[idx] = pIxx[idx] * pIyy[idx] - pIxy[idx] * pIxy[idx]; //Find det of A matrix
																	 //minEig = (IIyy + IIxx - sqrt((IIxx - IIyy).*(IIxx - IIyy) + 4.0*(IIxy.*IIxy))) / (2 * maxh*maxw);
			//pw_corner[idx] = (pIyy[idx] + pIxx[idx] - sqrt((pIxx[idx] - pIyy[idx])*(pIxx[idx] - pIyy[idx]) + 4.f*pIxy[idx] * pIxy[idx])) / (2 * w_size*w_size);
			if (pD[idx] > 1e-4f)
			{
				puv_res[j * 2 + i * 2 * mW] = (-pIyy[idx] * pIxt[idx] + pIxy[idx] * pIyt[idx]) / (pD[idx]);//(-IIyy(iD).*IIxt(iD) + IIxy(iD).*IIyt(iD)). / D(iD);
				puv_res[j * 2 + i * 2 * mW + 1] = (pIxy[idx] * pIxt[idx] - pIxx[idx] * pIyt[idx]) / (pD[idx]);//(IIxy(iD).*IIxt(iD)-IIxx(iD).*IIyt(iD))./D(iD);
				pw_corner[idx] = pD[idx] - alpha*(pIxx[idx] + pIyy[idx]);
				//pw_corner[idx] = pw_corner[idx] > 0 ? pw_corner[idx] : 0;
			}
			else
			{
				pw_corner[idx] = 0;
				pID[idx] = 1;
			}
			//if (pw_corner[idx] < 0)
			//	cout << "error" << endl;
		}
}


// Like KLT but compute the corner criteria at every s iteration
void RF::KLT2(Mat * uv_res, Mat * w_corner, Mat * iD, Mat * Ix, Mat * Iy, Mat * It, int w_size, int eig)
{
	int i, j, idx;
	float fD;
	Mat Ixx, Ixy, Iyy, Ixt, Iyt, kernel;
	Mat IIxx, IIxy, IIyy, IIxt, IIyt;
	kernel = Mat::ones(w_size, w_size, CV_32F);
	//Multiply to create value for hessian matrix
	cv::multiply(*Ix, *Ix, Ixx);
	cv::multiply(*Ix, *Iy, Ixy);
	cv::multiply(*Iy, *Iy, Iyy);
	cv::multiply(*Ix, *It, Ixt);
	cv::multiply(*Iy, *It, Iyt);
	//Sum around the neighbor of each point to create hessian matrix for each point
	cv::filter2D(Ixx, Ixx, CV_32F, kernel);
	cv::filter2D(Ixy, Ixy, CV_32F, kernel);
	cv::filter2D(Ixt, Ixt, CV_32F, kernel);
	cv::filter2D(Iyy, Iyy, CV_32F, kernel);
	cv::filter2D(Iyt, Iyt, CV_32F, kernel);
	//Sum 3 channels R+G+B
	cv::transform(Ixx, IIxx, Matx13f(1, 1, 1));
	cv::transform(Ixy, IIxy, Matx13f(1, 1, 1));
	cv::transform(Ixt, IIxt, Matx13f(1, 1, 1));
	cv::transform(Iyy, IIyy, Matx13f(1, 1, 1));
	cv::transform(Iyt, IIyt, Matx13f(1, 1, 1));

	Mat D(mH, mW, CV_32F);//Det(A) matrix
	float* pD = (float*)D.data;
	float* pIxx = (float*)IIxx.data;
	float* pIxy = (float*)IIxy.data;
	float* pIxt = (float*)IIxt.data;
	float* pIyy = (float*)IIyy.data;
	float* pIyt = (float*)IIyt.data;
	float* puv_res = (float*)uv_res->data; // H*W*2
	float* pw_corner = (float*)w_corner->data;//H*W
	uchar* pID = (uchar*)iD->data;
	float alpha = 0.1;
	for (i = 0; i < mH; i++)
		for (j = 0; j < mW; j++)
		{
			idx = i * mW + j;
			pD[idx] = pIxx[idx] * pIyy[idx] - pIxy[idx] * pIxy[idx]; //Find det of A matrix
			//minEig = (IIyy + IIxx - sqrt((IIxx - IIyy).*(IIxx - IIyy) + 4.0*(IIxy.*IIxy))) / (2 * maxh*maxw);
			if (eig)
			pw_corner[idx] = (pIyy[idx] + pIxx[idx] - sqrt((pIxx[idx] - pIyy[idx])*(pIxx[idx] - pIyy[idx]) + 4.f*pIxy[idx] * pIxy[idx])) / (2 * w_size*w_size);
			if (pD[idx] > 1e-4f)
			{
				puv_res[j * 2 + i * 2 * mW] = (-pIyy[idx] * pIxt[idx] + pIxy[idx] * pIyt[idx]) / (pD[idx]);//(-IIyy(iD).*IIxt(iD) + IIxy(iD).*IIyt(iD)). / D(iD);
				puv_res[j * 2 + i * 2 * mW + 1] = (pIxy[idx] * pIxt[idx] - pIxx[idx] * pIyt[idx]) / (pD[idx]);//(IIxy(iD).*IIxt(iD)-IIxx(iD).*IIyt(iD))./D(iD);
				//pw_corner[idx] = pD[idx] - alpha * (pIxx[idx] + pIyy[idx]);
				//pw_corner[idx] = pw_corner[idx] > 0 ? pw_corner[idx] : 0;
			}
			else
			{
				//pw_corner[idx] = 0;
				pID[idx] = 1;
			}
			//if (pw_corner[idx] < 0)
			//	cout << "error" << endl;
		}
}

void RF::eval_resuv(Mat * uv_res, Mat * conv, const Mat * iD, const Mat * w_corner, const int w_size, const Mat * Ix, const Mat * Iy, const Mat * It)
{
	int i;
	Mat conv_var = eval_var(uv_res, w_size);
	Mat conv_corner = eval_corner(w_corner, w_size);
	cv::multiply(conv_var, conv_corner, *conv);
	//float* pconv_var = (float*)conv_var.data;
	//float* pconv_corner = (float*)conv_corner.data;
	float max_val = 0;
	float min_val = 9999;
	uchar* piD = (uchar*)iD->data;
	float* pconv = (float*)conv->data;
	for (i = 0;i < mH*mW;i++) {
		if (piD[i] != 0) pconv[i] = 0;
		max_val = pconv[i] > max_val ? pconv[i] : max_val;
		min_val = min_val > pconv[i] ? pconv[i] : min_val;

	}

	//Normalze
	for (i = 0;i < mH*mW;i++) {
		pconv[i] /= max_val;
	}

	//printf("Max val of conv: %f \n", max_val);
	//printf("Min val of conv: %f \n", min_val);

}

void RF::propagation(const Mat * scene, Mat * conv, Mat * uv, const Mat * uv_res)
{
	int i, j, k, count = 0;
	(*uv) = (*uv) + (*uv_res);
	Mat XY2 = XY + (*uv);
	// Limit the border
	float* puv2 = (float*)XY2.data;
	float* pconv = (float*)conv->data;
	float* puv = (float*)uv->data;
	float* puv_res = (float*)uv_res->data;
	char stepW = XY2.dims;
	bool change = 0;
	for (i = 0;i < mH;i++)
		for (j = 0;j < mW;j++)
		{
			change = 0;
			int idx = j*stepW + i*mW*stepW;
			if (puv2[idx] < 0) {
				puv2[idx] = 0;
				pconv[i*mW + j] = 0;
				change = 1;
			}
			else if (puv2[idx] > (mW - 1)) {
				puv2[idx] = mW - 1;
				pconv[i*mW + j] = 0;
				change = 1;
			}
			if (puv2[idx + 1] < 0) {
				puv2[idx + 1] = 0;
				pconv[i*mW + j] = 0;
				change = 1;
			}
			else if (puv2[idx + 1] > (mH - 1)) {
				puv2[idx + 1] = mH - 1;
				pconv[i*mW + j] = 0;
				change = 1;
			}
			if (change) {
				puv[idx] -= puv_res[idx];
				puv[idx + 1] -= puv_res[idx + 1];
			}
		}
	//Mat *colflow = NULL;
	//MotionToColor(uv, colflow, 0);
	//imshow("Before", (*colflow));
	//cvWaitKey(0);
	//delete colflow;
	float* p_data = NULL;
	float* p_sum = NULL;
	Mat pixel_info = create_data2(scene, p_data, p_sum);
	int* p_n = (int*)pixel_info.data + 2; //3rd column // number of neighbor
	int* p_idxp = (int*)pixel_info.data;//1st column pixel index
	int* p_idxn = (int*)pixel_info.data + 3;//begin of neighbor 4th column

	Mat new_w = Mat::zeros(mH, mW, CV_32F);
	Mat new_uv = Mat::zeros(mH, mW, CV_32FC2);
	float* pnew_w = (float*)new_w.data;
	float* pnew_uv = (float*)new_uv.data;
	float val_row, urow, vrow,sum_ew;
	//	verify_data(pixel_info, p_sum, mW);
	// Start iteration for propagation
	for (k = 0;k < 50;k++)
	{
		for (i = 0;i < mH*mW;i++)
		{
			val_row = 0;
			count = 0;
			urow = 0;
			vrow = 0;
			sum_ew = 0;
			int n = p_n[i * 51];
			//printf("Neighbor: %d\n", n);
			for (j = 0;j < n;j++)
			{
				float* p_simi = (float*)p_idxn[i * 51 + j * 2 + 1];
				int idx_n = p_idxn[i * 51 + j * 2];
				sum_ew = (*p_simi)*pconv[idx_n];
				val_row += sum_ew;
				urow += puv[idx_n * 2] * (*p_simi);
				vrow += puv[idx_n * 2 + 1] * (*p_simi);
				//urow += puv[idx_n * 2] * sum_ew;
				//vrow += puv[idx_n * 2 + 1] * sum_ew;
			}
			if (p_sum[i] != 0) {
				pnew_w[i] = val_row / p_sum[i];
				if (pnew_w[i] > pconv[i])
				{
					pnew_uv[i * 2] = urow / p_sum[i];
					pnew_uv[i * 2 + 1] = vrow / p_sum[i];
					//cout << val_row << endl;
					//pnew_uv[i * 2] = urow / val_row;
					//pnew_uv[i * 2 + 1] = vrow / val_row;

				}
				else
				{
					pnew_uv[i * 2] = puv[i * 2];
					pnew_uv[i * 2 + 1] = puv[i * 2 + 1];
					pnew_w[i] = pconv[i];
				}

			}
			else printf("SUM psum ==0 \n");


		}

		//Reproject new value to old value
		for (i = 0;i < mH*mW;i++)
		{
			puv[i * 2] = pnew_uv[i * 2];
			puv[i * 2 + 1] = pnew_uv[i * 2 + 1];
			pconv[i] = pnew_w[i];
		}
		//Mat *colflow = NULL;
		//MotionToColor(uv, colflow, 0);
		//imshow("Color KLT", (*colflow));
		//imwrite("venus" + to_string(i) + ".jpg", *colflow);
		//cvWaitKey(0);
		//delete colflow;

	}
	//Mat *colflow = NULL;
	//MotionToColor(uv, colflow, 0);
	//imshow("After", (*colflow));
	//cvWaitKey(0);
	//delete colflow;

	delete[] p_data;
	delete[] p_sum;
	pixel_info.release();
	new_uv.release();
	new_w.release();
	XY2.release();

}

void RF::rescale_flow(Mat * uv)
{
	int oldH = uv->rows;
	float rate = mH / oldH;
	
	resize(*uv, *uv, Size(mW, mH), 0, 0, cv::INTER_CUBIC);
	*uv = (*uv)*rate;
	//Mat *colflow = NULL;
	//MotionToColor(uv, colflow, 0);
	//imshow("Color KLT", (*colflow));
	//cvWaitKey(0);
	//delete colflow;

}

Mat RF::eval_var(Mat *& uv_res, const int & w_size)
{
	int i, j;
	Mat cuv, cuv2;
	Mat conv_var = Mat::zeros(uv_res->rows, uv_res->cols, CV_32F);
	Mat uv2;
	cv::multiply(*uv_res, *uv_res, uv2);
	Mat kernel = Mat::ones(w_size, w_size, CV_32F);
	kernel.at<float>(((w_size - 1) / 2), ((w_size - 1) / 2)) = 0.f;
	int N = w_size*w_size - 1;
	int h_size = (w_size - 1) / 2;
	cv::filter2D(*uv_res, cuv, CV_32F, kernel);
	cv::filter2D(uv2, cuv2, CV_32F, kernel);
	float *pconv_var = (float*)conv_var.data;
	float *pcuv = (float*)cuv.data;
	float *pcuv2 = (float*)cuv2.data;
	float *puv_res = (float*)uv_res->data;
	float *puv2 = (float*)uv2.data;
	float max_val = 0;
	float min_val = 9999;
	float temp_val;
	for (i = h_size;i < (mH - h_size);i++)
		for (j = h_size;j < (mW - h_size);j++) {
			int ids = i*mW + j;
			int idu = i*mW * 2 + j * 2;
			int idv = idu + 1;
			/*pconv_var[ids] = temp_val = (pcuv2[idu] / N - 2 * (pcuv[idu] / N)*puv_res[idu] + puv2[idu]) + (pcuv2[idv] / N - 2 * (pcuv[idv] / N)*puv_res[idv] + puv2[idv]);*/
			temp_val = (pcuv2[idu] + pcuv2[idv] - ( pcuv[idu] * pcuv[idu] +   pcuv[idv] * pcuv[idv]) / (w_size*w_size) )/(w_size*w_size - 1);
			pconv_var[ids] = 1.0f / (temp_val + 0.000001);
			max_val = pconv_var[ids] > max_val ? pconv_var[ids] : max_val;
			min_val = min_val > pconv_var[ids] ? pconv_var[ids] : min_val;
		}
	// Normalize variance:
	//WARNING: max_val =0
	//printf("Max val of variance: %f \n", max_val);
	//printf("Min val of variance: %f \n", min_val);
	for (i = 0;i < mH*mW;i++)
	{
			pconv_var[i] = pconv_var[i] / max_val;
			//if( pconv_var[ids] < 0)
			//	cout << "Error" << endl;
		}
	//printf("Max val of variance after norm: %f \n", max_val/max_val);
	//printf("Min val of variance after norm: %f \n", min_val/max_val);
	return conv_var;
}

Mat RF::eval_corner(const Mat * w_corner, const int & w_size)
{
	Mat conv_corner = Mat::zeros(mH, mW, CV_32F);
	int half_size = (w_size - 1) / 2;
	float max_val = 0;
	float min_val = 9999;
	float temp_val;
	int i, j;
	float *pconv_corner = (float*)conv_corner.data;
	float *pw_corner = (float*)w_corner->data;
	for (i = half_size;i < (mH - half_size);i++)
		for (j = half_size;j < (mW - half_size);j++) {
			int ids = i*mW + j;
			temp_val = pw_corner[ids];
			max_val = temp_val > max_val ? temp_val : max_val;
			min_val = min_val > temp_val ? temp_val : min_val;
		}
	// Normalize variance:
	//WARNING: max_val =0
	//printf("Max val of corner: %f \n", max_val);
	//printf("Min val of corner: %f \n", min_val);
	for (i = 0;i < mH;i++)
		for (j = 0;j < mW;j++) {
			int ids = i*mW + j;
			pconv_corner[ids] = pw_corner[ids] / max_val;
		}
	//printf("Max val of corner after norm: %f \n", max_val / max_val);
	//printf("Min val of corner after norm: %f \n", min_val / max_val);
	return conv_corner;
}

Mat RF::eval_mix(Mat & w_corner, Mat & w_var, Mat * iD)
{
	Mat w_mix = Mat::zeros(mH, mW, CV_32F);
	float max_val = 0;
	float min_val = 9999;
	uchar* piD = (uchar*)iD->data;
	float* pconv = (float*)w_mix.data;
	float* pw_corner = (float*)w_corner.data;
	float* pw_var = (float*)w_var.data;

	//Find the min
	for (int i = 0; i < mH*mW; i++) {
		if (piD[i] ==0)
		{
			pconv[i] = min(pw_corner[i], pw_var[i]);
			max_val = pconv[i] > max_val ? pconv[i] : max_val;
			min_val = min_val > pconv[i] ? pconv[i] : min_val;
		}
	}

	//Normalize
	for (int i = 0; i < mH*mW; i++) {
		pconv[i] /= max_val;
	}
	return w_mix;
}

Mat * RF::getuv()
{
	return uv;
}

void RF::limit_res(Mat * uv_res)
{
	int i, j;
	float* puv_res = (float*)uv_res->data;
	for (i = 0;i < mH;i++)
		for (j = 0;j < mW;j++)
		{
			//u
			puv_res[j * 2 + i * 2 * mW] = puv_res[j * 2 + i * 2 * mW] > 1 ? 1 : puv_res[j * 2 + i * 2 * mW];
			puv_res[j * 2 + i * 2 * mW] = puv_res[j * 2 + i * 2 * mW] < -1 ? -1 : puv_res[j * 2 + i * 2 * mW];
			//v
			puv_res[j * 2 + i * 2 * mW + 1] = puv_res[j * 2 + i * 2 * mW + 1] > 1 ? 1 : puv_res[j * 2 + i * 2 * mW + 1];
			puv_res[j * 2 + i * 2 * mW + 1] = puv_res[j * 2 + i * 2 * mW + 1] < -1 ? -1 : puv_res[j * 2 + i * 2 * mW + 1];
		}
}

void meshgrid(const cv::Range &xgv, const cv::Range &ygv, cv::Mat &X, cv::Mat &Y)
{
	std::vector<float> t_x, t_y;
	for (int i = xgv.start; i <= xgv.end; i++) t_x.push_back(i);
	for (int i = ygv.start; i <= ygv.end; i++) t_y.push_back(i);
	cv::repeat(Mat(t_x).reshape(1, 1), Mat(t_y).total(), 1, X);
	cv::repeat(Mat(t_y).reshape(1, 1).t(), 1, Mat(t_x).total(), Y);
}

void MotionToColor(Mat* motim, Mat* &colim, float maxmotion)
{
	int width = motim->cols, height = motim->rows;
	int x, y;
	// determine motion range:
	colim = new Mat(height, width, CV_8UC3);

	float* pF = (float*)motim->data;

	float maxx = -999, maxy = -999;
	float minx = 999, miny = 999;
	float maxrad = -1;
	for (y = 0; y < height; y++) {
		for (x = 0; x < width; x++) {
			float fx = pF[x * 2 + y * 2 * width];//motim.Pixel(x, y, 0);
			float fy = pF[x * 2 + y * 2 * width + 1];//motim.Pixel(x, y, 1);
			if (unknown_flow(fx, fy))
				continue;
			maxx = __max(maxx, fx);
			maxy = __max(maxy, fy);
			minx = __min(minx, fx);
			miny = __min(miny, fy);
			float rad = sqrt(fx * fx + fy * fy);
			maxrad = __max(maxrad, rad);
		}
	}
	printf("max motion: %.4f  motion range: u = %.3f .. %.3f;  v = %.3f .. %.3f\n",
		maxrad, minx, maxx, miny, maxy);


	if (maxmotion > 0) // i.e., specified on commandline
		maxrad = maxmotion;

	if (maxrad == 0) // if flow == 0 everywhere
		maxrad = 1;

	if (VERBOSE)
		fprintf(stderr, "normalizing by %g\n", maxrad);
	for (y = 0; y < height; y++) {
		for (x = 0; x < width; x++) {
			//cout << "x:" << x << ", y:" << y << endl;
			float fx = pF[x * 2 + y * 2 * width];
			float fy = pF[x * 2 + y * 2 * width + 1];
			//cout << x * 3 + y * 3 * width << endl;
			unsigned char *pcolim = (unsigned char*)colim->data + x * 3 + y * 3 * width;
			if (unknown_flow(fx, fy)) {
				pcolim[0] = pcolim[1] = pcolim[2] = 0;
			}
			else {
				computeColor(fx / maxrad, fy / maxrad, pcolim);
			}
		}
	}
}

SparseMat create_data(const Mat* pimg, Mat* &sum_taux)
{
	int i, j;
	int W = pimg->cols, H = pimg->rows;
	const int np = H*W;
	float val;
	float* prow = NULL;
	//Mat data = Mat::zeros(5, 5, CV_32F);
	int size[] = { np,np };
	SparseMat data(2, size, CV_32F);
	sum_taux = new Mat(np, 1, CV_32F);
	(*sum_taux) = 0;
	int irow;
	prow = (float*)sum_taux->data;
	for (i = 0;i < H - 2;i++) {
		irow = i*W;


		// First point
		j = 0;

		data.ref<float>(irow + j, irow + j + 1) = val = e_simi(pimg, i, j, j + 1);// right 1  down 0
		data.ref<float>(irow + j + 1, irow + j) = val;
		prow[irow + j] += val;
		prow[irow + j + 1] += val;
		data.ref<float>(irow + j, irow + j + 2) = val = e_simi(pimg, i, j, j + 2);// right 2  down 0
		data.ref<float>(irow + j + 2, irow + j) = val;
		prow[irow + j] += val;
		prow[irow + j + 2] += val;

		data.ref<float>(irow + j, irow + j + W) = val = e_simi(pimg, i, j, j + W);// down 1
		data.ref<float>(irow + j + W, irow + j) = val;
		prow[irow + j] += val;
		prow[irow + j + W] += val;
		data.ref<float>(irow + j, irow + j + W + 1) = val = e_simi(pimg, i, j, j + W + 1);//right 1 down 1 
		data.ref<float>(irow + j + W + 1, irow + j) = val;
		prow[irow + j] += val;
		prow[irow + j + W + 1] += val;
		data.ref<float>(irow + j, irow + j + W + 2) = val = e_simi(pimg, i, j, j + W + 2);// right 2 down 1
		data.ref<float>(irow + j + W + 2, irow + j) = val;
		prow[irow + j] += val;
		prow[irow + j + W + 2] += val;

		data.ref<float>(irow + j, irow + j + 2 * W) = val = e_simi(pimg, i, j, j + 2 * W); // down 2 
		data.ref<float>(irow + j + 2 * W, irow + j) = val;
		prow[irow + j] += val;
		prow[irow + j + 2 * W] += val;
		data.ref<float>(irow + j, irow + j + 2 * W + 1) = val = e_simi(pimg, i, j, j + 2 * W + 1); // right 1 down 2 
		data.ref<float>(irow + j + 2 * W + 1, irow + j) = val;
		prow[irow + j] += val;
		prow[irow + j + 2 * W + 1] += val;
		data.ref<float>(irow + j, irow + j + 2 * W + 2) = val = e_simi(pimg, i, j, j + 2 * W); // right 2 down 2 
		data.ref<float>(irow + j + 2 * W + 2, irow + j) = val;
		prow[irow + j] += val;
		prow[irow + j + 2 * W + 2] += val;
		// Second point
		j = 1;
		data.ref<float>(irow + j, irow + j + 1) = val = e_simi(pimg, i, j, j + 1);// right 1  down 0
		data.ref<float>(irow + j + 1, irow + j) = val;
		prow[irow + j] += val;
		prow[irow + j + 1] += val;
		data.ref<float>(irow + j, irow + j + 2) = val = e_simi(pimg, i, j, j + 2);// right 2  down 0
		data.ref<float>(irow + j + 2, irow + j) = val;
		prow[irow + j] += val;
		prow[irow + j + 2] += val;

		data.ref<float>(irow + j, irow + j + W - 1) = val = e_simi(pimg, i, j, j + W - 1);// down 1 left 1
		data.ref<float>(irow + j + W - 1, irow + j) = val;
		prow[irow + j] += val;
		prow[irow + j + W - 1] += val;
		data.ref<float>(irow + j, irow + j + W) = val = e_simi(pimg, i, j, j + W);// down 1
		data.ref<float>(irow + j + W, irow + j) = val;
		prow[irow + j] += val;
		prow[irow + j + W] += val;
		data.ref<float>(irow + j, irow + j + W + 1) = val = e_simi(pimg, i, j, j + W + 1);//right 1 down 1 
		data.ref<float>(irow + j + W + 1, irow + j) = val;
		prow[irow + j] += val;
		prow[irow + j + W + 1] += val;
		data.ref<float>(irow + j, irow + j + W + 2) = val = e_simi(pimg, i, j, j + W + 2);// right 2 down 1
		data.ref<float>(irow + j + W + 2, irow + j) = val;
		prow[irow + j] += val;
		prow[irow + j + W + 2] += val;

		data.ref<float>(irow + j, irow + j + 2 * W - 1) = val = e_simi(pimg, i, j, j + 2 * W); // down 2 left1
		data.ref<float>(irow + j + 2 * W - 1, irow + j) = val;
		prow[irow + j] += val;
		prow[irow + j + 2 * W - 1] += val;
		data.ref<float>(irow + j, irow + j + 2 * W) = val = e_simi(pimg, i, j, j + 2 * W); // down 2 
		data.ref<float>(irow + j + 2 * W, irow + j) = val;
		prow[irow + j] += val;
		prow[irow + j + 2 * W] += val;
		data.ref<float>(irow + j, irow + j + 2 * W + 1) = val = e_simi(pimg, i, j, j + 2 * W + 1); // right 1 down 2 
		data.ref<float>(irow + j + 2 * W + 1, irow + j) = val;
		prow[irow + j] += val;
		prow[irow + j + 2 * W + 1] += val;
		data.ref<float>(irow + j, irow + j + 2 * W + 2) = val = e_simi(pimg, i, j, j + 2 * W); // right 2 down 2 
		data.ref<float>(irow + j + 2 * W + 2, irow + j) = val;
		prow[irow + j] += val;
		prow[irow + j + 2 * W + 2] += val;
		//Middle point
		for (j = 2;j < W - 2;j++) {
			data.ref<float>(irow + j, irow + j + 1) = val = e_simi(pimg, i, j, j + 1);// right 1  down 0
			data.ref<float>(irow + j + 1, irow + j) = val;
			prow[irow + j] += val;
			prow[irow + j + 1] += val;
			data.ref<float>(irow + j, irow + j + 2) = val = e_simi(pimg, i, j, j + 2);// right 2  down 0
			data.ref<float>(irow + j + 2, irow + j) = val;
			prow[irow + j] += val;
			prow[irow + j + 2] += val;

			data.ref<float>(irow + j, irow + j + W - 2) = val = e_simi(pimg, i, j, j + W - 2);// down 1 left 2
			data.ref<float>(irow + j + W - 2, irow + j) = val;
			prow[irow + j] += val;
			prow[irow + j + W - 2] += val;
			data.ref<float>(irow + j, irow + j + W - 1) = val = e_simi(pimg, i, j, j + W - 1);// down 1 left 1
			data.ref<float>(irow + j + W - 1, irow + j) = val;
			prow[irow + j] += val;
			prow[irow + j + W - 1] += val;
			data.ref<float>(irow + j, irow + j + W) = val = e_simi(pimg, i, j, j + W);// down 1
			data.ref<float>(irow + j + W, irow + j) = val;
			prow[irow + j] += val;
			prow[irow + j + W] += val;
			data.ref<float>(irow + j, irow + j + W + 1) = val = e_simi(pimg, i, j, j + W + 1);//right 1 down 1 
			data.ref<float>(irow + j + W + 1, irow + j) = val;
			prow[irow + j] += val;
			prow[irow + j + W + 1] += val;
			data.ref<float>(irow + j, irow + j + W + 2) = val = e_simi(pimg, i, j, j + W + 2);// right 2 down 1
			data.ref<float>(irow + j + W + 2, irow + j) = val;
			prow[irow + j] += val;
			prow[irow + j + W + 2] += val;

			data.ref<float>(irow + j, irow + j + 2 * W - 2) = val = e_simi(pimg, i, j, j + 2 * W - 2); // down 2 left 2
			data.ref<float>(irow + j + 2 * W - 2, irow + j) = val;
			prow[irow + j] += val;
			prow[irow + j + 2 * W - 2] += val;
			data.ref<float>(irow + j, irow + j + 2 * W - 1) = val = e_simi(pimg, i, j, j + 2 * W - 1); // down 2 left 1
			data.ref<float>(irow + j + 2 * W - 1, irow + j) = val;
			prow[irow + j] += val;
			prow[irow + j + 2 * W - 1] += val;
			data.ref<float>(irow + j, irow + j + 2 * W) = val = e_simi(pimg, i, j, j + 2 * W); // down 2 
			data.ref<float>(irow + j + 2 * W, irow + j) = val;
			prow[irow + j] += val;
			prow[irow + j + 2 * W] += val;
			data.ref<float>(irow + j, irow + j + 2 * W + 1) = val = e_simi(pimg, i, j, j + 2 * W + 1); //  down 2 right 1
			data.ref<float>(irow + j + 2 * W + 1, irow + j) = val;
			prow[irow + j] += val;
			prow[irow + j + 2 * W + 1] += val;
			data.ref<float>(irow + j, irow + j + 2 * W + 2) = val = e_simi(pimg, i, j, j + 2 * W + 2); //  down 2 right 2
			data.ref<float>(irow + j + 2 * W + 2, irow + j) = val;
			prow[irow + j] += val;
			prow[irow + j + 2 * W + 2] += val;
		}
		//(W-2)th point
		j = (W - 2);
		data.ref<float>(irow + j, irow + j + 1) = val = e_simi(pimg, i, j, j + 1);// right 1  down 0
		data.ref<float>(irow + j + 1, irow + j) = val;
		prow[irow + j] += val;
		prow[irow + j + 1] += val;

		data.ref<float>(irow + j, irow + j + W - 2) = val = e_simi(pimg, i, j, j + W - 2);// down 1 left 2
		data.ref<float>(irow + j + W - 2, irow + j) = val;
		prow[irow + j] += val;
		prow[irow + j + W - 2] += val;
		data.ref<float>(irow + j, irow + j + W - 1) = val = e_simi(pimg, i, j, j + W - 1);// down 1 left 1
		data.ref<float>(irow + j + W - 1, irow + j) = val;
		prow[irow + j] += val;
		prow[irow + j + W - 1] += val;
		data.ref<float>(irow + j, irow + j + W) = val = e_simi(pimg, i, j, j + W);// down 1
		data.ref<float>(irow + j + W, irow + j) = val;
		prow[irow + j] += val;
		prow[irow + j + W] += val;
		data.ref<float>(irow + j, irow + j + W + 1) = val = e_simi(pimg, i, j, j + W + 1);//down 1 right 1 
		data.ref<float>(irow + j + W + 1, irow + j) = val;
		prow[irow + j] += val;
		prow[irow + j + W + 1] += val;

		data.ref<float>(irow + j, irow + j + 2 * W - 2) = val = e_simi(pimg, i, j, j + 2 * W - 2); // down 2 left 2
		data.ref<float>(irow + j + 2 * W - 2, irow + j) = val;
		prow[irow + j] += val;
		prow[irow + j + 2 * W - 2] += val;
		data.ref<float>(irow + j, irow + j + 2 * W - 1) = val = e_simi(pimg, i, j, j + 2 * W - 1); // down 2 left 1
		data.ref<float>(irow + j + 2 * W - 1, irow + j) = val;
		prow[irow + j] += val;
		prow[irow + j + 2 * W - 1] += val;
		data.ref<float>(irow + j, irow + j + 2 * W) = val = e_simi(pimg, i, j, j + 2 * W); // down 2 
		data.ref<float>(irow + j + 2 * W, irow + j) = val;
		prow[irow + j] += val;
		prow[irow + j + 2 * W] += val;
		data.ref<float>(irow + j, irow + j + 2 * W + 1) = val = e_simi(pimg, i, j, j + 2 * W + 1); //  down 2 right 1
		data.ref<float>(irow + j + 2 * W + 1, irow + j) = val;
		prow[irow + j] += val;
		prow[irow + j + 2 * W + 1] += val;

		//(W-1)th point
		j = (W - 1);
		data.ref<float>(irow + j, irow + j + W - 2) = val = e_simi(pimg, i, j, j + W - 2);// down 1 left 2
		data.ref<float>(irow + j + W - 2, irow + j) = val;
		prow[irow + j] += val;
		prow[irow + j + W - 2] += val;
		data.ref<float>(irow + j, irow + j + W - 1) = val = e_simi(pimg, i, j, j + W - 1);// down 1 left 1
		data.ref<float>(irow + j + W - 1, irow + j) = val;
		prow[irow + j] += val;
		prow[irow + j + 2 * W - 1] += val;
		data.ref<float>(irow + j, irow + j + W) = val = e_simi(pimg, i, j, j + W);// down 1
		data.ref<float>(irow + j + W, irow + j) = val;
		prow[irow + j] += val;
		prow[irow + j + 2 * W] += val;

		data.ref<float>(irow + j, irow + j + 2 * W - 2) = val = e_simi(pimg, i, j, j + 2 * W - 2); // down 2 left 2
		data.ref<float>(irow + j + 2 * W - 2, irow + j) = val;
		prow[irow + j] += val;
		prow[irow + j + 2 * W - 2] += val;
		data.ref<float>(irow + j, irow + j + 2 * W - 1) = val = e_simi(pimg, i, j, j + 2 * W - 1); // down 2 left 1
		data.ref<float>(irow + j + 2 * W - 1, irow + j) = val;
		prow[irow + j] += val;
		prow[irow + j + 2 * W - 1] += val;
		data.ref<float>(irow + j, irow + j + 2 * W) = val = e_simi(pimg, i, j, j + 2 * W); // down 2 
		data.ref<float>(irow + j + 2 * W, irow + j) = val;
		prow[irow + j] += val;
		prow[irow + j + 2 * W] += val;
	}
	/***********************************(H-2)th line**************************/
	i = H - 2;
	irow = i*W;
	//prow = (float*)sum_taux->ptr(i*W);
	//First point
	j = 0;
	data.ref<float>(irow + j, irow + j + 1) = val = e_simi(pimg, i, j, j + 1);// right 1  down 0
	data.ref<float>(irow + j + 1, irow + j) = val;
	prow[irow + j] += val;
	prow[irow + j + 1] += val;
	data.ref<float>(irow + j, irow + j + 2) = val = e_simi(pimg, i, j, j + 2);// right 2  down 0
	data.ref<float>(irow + j + 2, irow + j) = val;
	prow[irow + j] += val;
	prow[irow + j + 2] += val;

	data.ref<float>(irow + j, irow + j + W) = val = e_simi(pimg, i, j, j + W);// down 1
	data.ref<float>(irow + j + W, irow + j) = val;
	prow[irow + j] += val;
	prow[irow + j + W] += val;
	data.ref<float>(irow + j, irow + j + W + 1) = val = e_simi(pimg, i, j, j + W + 1);//right 1 down 1 
	data.ref<float>(irow + j + W + 1, irow + j) = val;
	prow[irow + j] += val;
	prow[irow + j + W + 1] += val;
	data.ref<float>(irow + j, irow + j + W + 2) = val = e_simi(pimg, i, j, j + W + 2);// right 2 down 1
	data.ref<float>(irow + j + W + 2, irow + j) = val;
	prow[irow + j] += val;
	prow[irow + j + W + 2] += val;
	// Second point
	j = 1;
	data.ref<float>(irow + j, irow + j + 1) = val = e_simi(pimg, i, j, j + 1);// right 1  down 0
	data.ref<float>(irow + j + 1, irow + j) = val;
	prow[irow + j] += val;
	prow[irow + j + 1] += val;
	data.ref<float>(irow + j, irow + j + 2) = val = e_simi(pimg, i, j, j + 2);// right 2  down 0
	data.ref<float>(irow + j + 2, irow + j) = val;
	prow[irow + j] += val;
	prow[irow + j + 2] += val;

	data.ref<float>(irow + j, irow + j + W - 1) = val = e_simi(pimg, i, j, j + W - 1);// down 1 left 1
	data.ref<float>(irow + j + W - 1, irow + j) = val;
	prow[irow + j] += val;
	prow[irow + j + W - 1] += val;
	data.ref<float>(irow + j, irow + j + W) = val = e_simi(pimg, i, j, j + W);// down 1
	data.ref<float>(irow + j + W, irow + j) = val;
	prow[irow + j] += val;
	prow[irow + j + W] += val;
	data.ref<float>(irow + j, irow + j + W + 1) = val = e_simi(pimg, i, j, j + W + 1);//right 1 down 1 
	data.ref<float>(irow + j + W + 1, irow + j) = val;
	prow[irow + j] += val;
	prow[irow + j + W + 1] += val;
	data.ref<float>(irow + j, irow + j + W + 2) = val = e_simi(pimg, i, j, j + W + 2);// right 2 down 1
	data.ref<float>(irow + j + W + 2, irow + j) = val;
	prow[irow + j] += val;
	prow[irow + j + W + 2] += val;
	//Middle point
	for (j = 2;j < W - 2;j++) {
		data.ref<float>(irow + j, irow + j + 1) = val = e_simi(pimg, i, j, j + 1);// right 1  down 0
		data.ref<float>(irow + j + 1, irow + j) = val;
		prow[irow + j] += val;
		prow[irow + j + 1] += val;
		data.ref<float>(irow + j, irow + j + 2) = val = e_simi(pimg, i, j, j + 2);// right 2  down 0
		data.ref<float>(irow + j + 2, irow + j) = val;
		prow[irow + j] += val;
		prow[irow + j + 2] += val;

		data.ref<float>(irow + j, irow + j + W - 2) = val = e_simi(pimg, i, j, j + W - 2);// down 1 left 2
		data.ref<float>(irow + j + W - 2, irow + j) = val;
		prow[irow + j] += val;
		prow[irow + j + W - 2] += val;
		data.ref<float>(irow + j, irow + j + W - 1) = val = e_simi(pimg, i, j, j + W - 1);// down 1 left 1
		data.ref<float>(irow + j + W - 1, irow + j) = val;
		prow[irow + j] += val;
		prow[irow + j + W - 1] += val;
		data.ref<float>(irow + j, irow + j + W) = val = e_simi(pimg, i, j, j + W);// down 1
		data.ref<float>(irow + j + W, irow + j) = val;
		prow[irow + j] += val;
		prow[irow + j + W] += val;
		data.ref<float>(irow + j, irow + j + W + 1) = val = e_simi(pimg, i, j, j + W + 1);//right 1 down 1 
		data.ref<float>(irow + j + W + 1, irow + j) = val;
		prow[irow + j] += val;
		prow[irow + j + W + 1] += val;
		data.ref<float>(irow + j, irow + j + W + 2) = val = e_simi(pimg, i, j, j + W + 2);// right 2 down 1
		data.ref<float>(irow + j + W + 2, irow + j) = val;
		prow[irow + j] += val;
		prow[irow + j + W + 2] += val;

	}
	//(W-2)th point
	j = (W - 2);
	data.ref<float>(irow + j, irow + j + 1) = val = e_simi(pimg, i, j, j + 1);// right 1  down 0
	data.ref<float>(irow + j + 1, irow + j) = val;
	prow[irow + j] += val;
	prow[irow + j + 1] += val;

	data.ref<float>(irow + j, irow + j + W - 2) = val = e_simi(pimg, i, j, j + W - 2);// down 1 left 2
	data.ref<float>(irow + j + W - 2, irow + j) = val;
	prow[irow + j] += val;
	prow[irow + j + W - 2] += val;
	data.ref<float>(irow + j, irow + j + W - 1) = val = e_simi(pimg, i, j, j + W - 1);// down 1 left 1
	data.ref<float>(irow + j + W - 1, irow + j) = val;
	prow[irow + j] += val;
	prow[irow + j + W - 1] += val;
	data.ref<float>(irow + j, irow + j + W) = val = e_simi(pimg, i, j, j + W);// down 1
	data.ref<float>(irow + j + W, irow + j) = val;
	prow[irow + j] += val;
	prow[irow + j + W] += val;
	data.ref<float>(irow + j, irow + j + W + 1) = val = e_simi(pimg, i, j, j + W + 1);//right 1 down 1 
	data.ref<float>(irow + j + W + 1, irow + j) = val;
	prow[irow + j] += val;
	prow[irow + j + W + 1] += val;

	//(W-1)th point
	j = (W - 1);
	data.ref<float>(irow + j, irow + j + W - 2) = val = e_simi(pimg, i, j, j + W - 2);// down 1 left 2
	data.ref<float>(irow + j + W - 2, irow + j) = val;
	prow[irow + j] += val;
	prow[irow + j + W - 2] += val;
	data.ref<float>(irow + j, irow + j + W - 1) = val = e_simi(pimg, i, j, j + W - 1);// down 1 left 1
	data.ref<float>(irow + j + W - 1, irow + j) = val;
	prow[irow + j] += val;
	prow[irow + j + W - 1] += val;
	data.ref<float>(irow + j, irow + j + W) = val = e_simi(pimg, i, j, j + W);// down 1
	data.ref<float>(irow + j + W, irow + j) = val;
	prow[irow + j] += val;
	prow[irow + j + W] += val;
	/****************************(H-1)th line/last line**************************/
	i = H - 1;
	irow = i*W;
	//	prow = (float*)sum_taux->ptr(i*W);
	//	prow = (float*)data.ptr(i*np*W);
	//Middle point
	for (j = 0;j < W - 2;j++) {
		data.ref<float>(irow + j, irow + j + 1) = val = e_simi(pimg, i, j, j + 1);// right 1  down 0
		data.ref<float>(irow + j + 1, irow + j) = val;
		prow[irow + j] += val;
		prow[irow + j + 1] += val;
		data.ref<float>(irow + j, irow + j + 2) = val = e_simi(pimg, i, j, j + 2);// right 2  down 0
		data.ref<float>(irow + j + 2, irow + j) = val;
		prow[irow + j] += val;
		prow[irow + j + 2] += val;
	}
	//(W-2)th point
	j = (W - 2);
	data.ref<float>(irow + j, irow + j + 1) = val = e_simi(pimg, i, j, j + 1);// right 1  down 0
	data.ref<float>(irow + j + 1, irow + j) = val;
	prow[irow + j] += val;
	prow[irow + j + 1] += val;
	//(W-1)th point
	return data;
}

Mat create_data2(const Mat* pimg, float* &array_data, float* &array_sum) {
	int i, j, i_arr = 0, irow = 0, idx_point = 0;
	int W = pimg->cols, H = pimg->rows;
	const int np = H*W;
	const int w = 51;
	//[8+10+12(W-4)+9+6](H-2)+[5+6+7(W-4)+5+3]+[2(W-2)+1]
	if (array_data != NULL) delete[] array_data;
	if (array_sum != NULL) delete[] array_sum;
	int dim_array = (H - 2)*(33 + 12 * (W - 4)) + 19 + 7 * (W - 4) + 2 * (W - 2) + 1;
	array_data = new float[dim_array];
	array_sum = new float[np];
	for (i = 0;i < np;i++) array_sum[i] = 0;
	Mat pixel_info = Mat::zeros(np, w, CV_32S);
	
	int* p_idxp = (int*)pixel_info.data;//1st column pixel index
	int* p_sum = (int*)pixel_info.data + 1; // 2nd point to sum weight
	int* p_n = (int*)pixel_info.data + 2; //3rd column // number of neighbor
	int* p_idxn = (int*)pixel_info.data + 3;//begin of neighbor 4th column
	
	/**************************************************************************
	*********************************|0 |1 |2 |********************************
	****************************3 |4 |5 |6 |7 |********************************
	****************************8 |9 |10|11|12|*******************************/
	for (i = 0;i < H - 2;i++) {
		irow = i*W;
		// First point
		j = 0;
		idx_point = irow + j;
		p_idxp[(idx_point)*w] = idx_point; // index of pixel
		update_neighbor(idx_point, i, j, w, W, 1, pimg, i_arr, array_data, array_sum, p_sum, p_n, p_idxn);
		update_neighbor(idx_point, i, j, w, W, 2, pimg, i_arr, array_data, array_sum, p_sum, p_n, p_idxn);
		update_neighbor(idx_point, i, j, w, W, 5, pimg, i_arr, array_data, array_sum, p_sum, p_n, p_idxn);
		update_neighbor(idx_point, i, j, w, W, 6, pimg, i_arr, array_data, array_sum, p_sum, p_n, p_idxn);
		update_neighbor(idx_point, i, j, w, W, 7, pimg, i_arr, array_data, array_sum, p_sum, p_n, p_idxn);
		update_neighbor(idx_point, i, j, w, W, 10, pimg, i_arr, array_data, array_sum, p_sum, p_n, p_idxn);
		update_neighbor(idx_point, i, j, w, W, 11, pimg, i_arr, array_data, array_sum, p_sum, p_n, p_idxn);
		update_neighbor(idx_point, i, j, w, W, 12, pimg, i_arr, array_data, array_sum, p_sum, p_n, p_idxn);
		// Second point
		j = 1;
		idx_point = irow + j;
		p_idxp[(idx_point)*w] = idx_point; // index of pixel
		update_neighbor(idx_point, i, j, w, W, 1, pimg, i_arr, array_data, array_sum, p_sum, p_n, p_idxn);
		update_neighbor(idx_point, i, j, w, W, 2, pimg, i_arr, array_data, array_sum, p_sum, p_n, p_idxn);
		update_neighbor(idx_point, i, j, w, W, 4, pimg, i_arr, array_data, array_sum, p_sum, p_n, p_idxn);
		update_neighbor(idx_point, i, j, w, W, 5, pimg, i_arr, array_data, array_sum, p_sum, p_n, p_idxn);
		update_neighbor(idx_point, i, j, w, W, 6, pimg, i_arr, array_data, array_sum, p_sum, p_n, p_idxn);
		update_neighbor(idx_point, i, j, w, W, 7, pimg, i_arr, array_data, array_sum, p_sum, p_n, p_idxn);
		update_neighbor(idx_point, i, j, w, W, 9, pimg, i_arr, array_data, array_sum, p_sum, p_n, p_idxn);
		update_neighbor(idx_point, i, j, w, W, 10, pimg, i_arr, array_data, array_sum, p_sum, p_n, p_idxn);
		update_neighbor(idx_point, i, j, w, W, 11, pimg, i_arr, array_data, array_sum, p_sum, p_n, p_idxn);
		update_neighbor(idx_point, i, j, w, W, 12, pimg, i_arr, array_data, array_sum, p_sum, p_n, p_idxn);
		//Middle point
		for (j = 2;j < W - 2;j++) {
			idx_point = irow + j;
			p_idxp[(idx_point)*w] = idx_point; // index of pixel
			update_neighbor(idx_point, i, j, w, W, 1, pimg, i_arr, array_data, array_sum, p_sum, p_n, p_idxn);
			update_neighbor(idx_point, i, j, w, W, 2, pimg, i_arr, array_data, array_sum, p_sum, p_n, p_idxn);
			update_neighbor(idx_point, i, j, w, W, 3, pimg, i_arr, array_data, array_sum, p_sum, p_n, p_idxn);
			update_neighbor(idx_point, i, j, w, W, 4, pimg, i_arr, array_data, array_sum, p_sum, p_n, p_idxn);
			update_neighbor(idx_point, i, j, w, W, 5, pimg, i_arr, array_data, array_sum, p_sum, p_n, p_idxn);
			update_neighbor(idx_point, i, j, w, W, 6, pimg, i_arr, array_data, array_sum, p_sum, p_n, p_idxn);
			update_neighbor(idx_point, i, j, w, W, 7, pimg, i_arr, array_data, array_sum, p_sum, p_n, p_idxn);
			update_neighbor(idx_point, i, j, w, W, 8, pimg, i_arr, array_data, array_sum, p_sum, p_n, p_idxn);
			update_neighbor(idx_point, i, j, w, W, 9, pimg, i_arr, array_data, array_sum, p_sum, p_n, p_idxn);
			update_neighbor(idx_point, i, j, w, W, 10, pimg, i_arr, array_data, array_sum, p_sum, p_n, p_idxn);
			update_neighbor(idx_point, i, j, w, W, 11, pimg, i_arr, array_data, array_sum, p_sum, p_n, p_idxn);
			update_neighbor(idx_point, i, j, w, W, 12, pimg, i_arr, array_data, array_sum, p_sum, p_n, p_idxn);
		}
		//(W-2)th point
		j = (W - 2);
		idx_point = irow + j;
		p_idxp[(idx_point)*w] = idx_point; // index of pixel
		update_neighbor(idx_point, i, j, w, W, 1, pimg, i_arr, array_data, array_sum, p_sum, p_n, p_idxn);
		update_neighbor(idx_point, i, j, w, W, 3, pimg, i_arr, array_data, array_sum, p_sum, p_n, p_idxn);
		update_neighbor(idx_point, i, j, w, W, 4, pimg, i_arr, array_data, array_sum, p_sum, p_n, p_idxn);
		update_neighbor(idx_point, i, j, w, W, 5, pimg, i_arr, array_data, array_sum, p_sum, p_n, p_idxn);
		update_neighbor(idx_point, i, j, w, W, 6, pimg, i_arr, array_data, array_sum, p_sum, p_n, p_idxn);
		update_neighbor(idx_point, i, j, w, W, 8, pimg, i_arr, array_data, array_sum, p_sum, p_n, p_idxn);
		update_neighbor(idx_point, i, j, w, W, 9, pimg, i_arr, array_data, array_sum, p_sum, p_n, p_idxn);
		update_neighbor(idx_point, i, j, w, W, 10, pimg, i_arr, array_data, array_sum, p_sum, p_n, p_idxn);
		update_neighbor(idx_point, i, j, w, W, 11, pimg, i_arr, array_data, array_sum, p_sum, p_n, p_idxn);
		//(W-1)th point
		j = (W - 1);
		idx_point = irow + j;
		p_idxp[(idx_point)*w] = idx_point; // index of pixel
		update_neighbor(idx_point, i, j, w, W, 3, pimg, i_arr, array_data, array_sum, p_sum, p_n, p_idxn);
		update_neighbor(idx_point, i, j, w, W, 4, pimg, i_arr, array_data, array_sum, p_sum, p_n, p_idxn);
		update_neighbor(idx_point, i, j, w, W, 5, pimg, i_arr, array_data, array_sum, p_sum, p_n, p_idxn);
		update_neighbor(idx_point, i, j, w, W, 8, pimg, i_arr, array_data, array_sum, p_sum, p_n, p_idxn);
		update_neighbor(idx_point, i, j, w, W, 9, pimg, i_arr, array_data, array_sum, p_sum, p_n, p_idxn);
		update_neighbor(idx_point, i, j, w, W, 10, pimg, i_arr, array_data, array_sum, p_sum, p_n, p_idxn);
	}
	/***********************************(H-2)th line**************************/
	{
		i = H - 2;
		irow = i*W;
		//First point
		j = 0;
		idx_point = irow + j;
		p_idxp[(idx_point)*w] = idx_point; // index of pixel
		update_neighbor(idx_point, i, j, w, W, 1, pimg, i_arr, array_data, array_sum, p_sum, p_n, p_idxn);
		update_neighbor(idx_point, i, j, w, W, 2, pimg, i_arr, array_data, array_sum, p_sum, p_n, p_idxn);
		update_neighbor(idx_point, i, j, w, W, 5, pimg, i_arr, array_data, array_sum, p_sum, p_n, p_idxn);
		update_neighbor(idx_point, i, j, w, W, 6, pimg, i_arr, array_data, array_sum, p_sum, p_n, p_idxn);
		update_neighbor(idx_point, i, j, w, W, 7, pimg, i_arr, array_data, array_sum, p_sum, p_n, p_idxn);
		// Second point
		j = 1;
		idx_point = irow + j;
		p_idxp[(idx_point)*w] = idx_point; // index of pixel
		update_neighbor(idx_point, i, j, w, W, 1, pimg, i_arr, array_data, array_sum, p_sum, p_n, p_idxn);
		update_neighbor(idx_point, i, j, w, W, 2, pimg, i_arr, array_data, array_sum, p_sum, p_n, p_idxn);
		update_neighbor(idx_point, i, j, w, W, 4, pimg, i_arr, array_data, array_sum, p_sum, p_n, p_idxn);
		update_neighbor(idx_point, i, j, w, W, 5, pimg, i_arr, array_data, array_sum, p_sum, p_n, p_idxn);
		update_neighbor(idx_point, i, j, w, W, 6, pimg, i_arr, array_data, array_sum, p_sum, p_n, p_idxn);
		update_neighbor(idx_point, i, j, w, W, 7, pimg, i_arr, array_data, array_sum, p_sum, p_n, p_idxn);
		//Middle point
		for (j = 2;j < W - 2;j++) {
			idx_point = irow + j;
			p_idxp[(idx_point)*w] = idx_point; // index of pixel
			update_neighbor(idx_point, i, j, w, W, 1, pimg, i_arr, array_data, array_sum, p_sum, p_n, p_idxn);
			update_neighbor(idx_point, i, j, w, W, 2, pimg, i_arr, array_data, array_sum, p_sum, p_n, p_idxn);
			update_neighbor(idx_point, i, j, w, W, 3, pimg, i_arr, array_data, array_sum, p_sum, p_n, p_idxn);
			update_neighbor(idx_point, i, j, w, W, 4, pimg, i_arr, array_data, array_sum, p_sum, p_n, p_idxn);
			update_neighbor(idx_point, i, j, w, W, 5, pimg, i_arr, array_data, array_sum, p_sum, p_n, p_idxn);
			update_neighbor(idx_point, i, j, w, W, 6, pimg, i_arr, array_data, array_sum, p_sum, p_n, p_idxn);
			update_neighbor(idx_point, i, j, w, W, 7, pimg, i_arr, array_data, array_sum, p_sum, p_n, p_idxn);
		}
		//(W-2)th point
		j = (W - 2);
		idx_point = irow + j;
		p_idxp[(idx_point)*w] = idx_point; // index of pixel
		update_neighbor(idx_point, i, j, w, W, 1, pimg, i_arr, array_data, array_sum, p_sum, p_n, p_idxn);
		update_neighbor(idx_point, i, j, w, W, 3, pimg, i_arr, array_data, array_sum, p_sum, p_n, p_idxn);
		update_neighbor(idx_point, i, j, w, W, 4, pimg, i_arr, array_data, array_sum, p_sum, p_n, p_idxn);
		update_neighbor(idx_point, i, j, w, W, 5, pimg, i_arr, array_data, array_sum, p_sum, p_n, p_idxn);
		update_neighbor(idx_point, i, j, w, W, 6, pimg, i_arr, array_data, array_sum, p_sum, p_n, p_idxn);
		//(W-1)th point
		j = (W - 1);
		idx_point = irow + j;
		p_idxp[(idx_point)*w] = idx_point; // index of pixel
		update_neighbor(idx_point, i, j, w, W, 3, pimg, i_arr, array_data, array_sum, p_sum, p_n, p_idxn);
		update_neighbor(idx_point, i, j, w, W, 4, pimg, i_arr, array_data, array_sum, p_sum, p_n, p_idxn);
		update_neighbor(idx_point, i, j, w, W, 5, pimg, i_arr, array_data, array_sum, p_sum, p_n, p_idxn);
	}
	/****************************(H-1)th line/last line**************************/
	{
		i = H - 1;
		irow = i*W;
		//Middle point
		for (j = 0;j < W - 2;j++) {
			idx_point = irow + j;
			p_idxp[(idx_point)*w] = idx_point; // index of pixel
			update_neighbor(idx_point, i, j, w, W, 1, pimg, i_arr, array_data, array_sum, p_sum, p_n, p_idxn);
			update_neighbor(idx_point, i, j, w, W, 2, pimg, i_arr, array_data, array_sum, p_sum, p_n, p_idxn);
		}
		//(W-2)th point
		j = (W - 2);
		idx_point = irow + j;
		p_idxp[(idx_point)*w] = idx_point; // index of pixel
		update_neighbor(idx_point, i, j, w, W, 1, pimg, i_arr, array_data, array_sum, p_sum, p_n, p_idxn);
		//(W-1)th point
		j = (W - 1);
		idx_point = irow + j;
		p_idxp[(idx_point)*w] = idx_point; // index of pixel
	}
	return pixel_info;
}

float e_simi(const Mat * pimg, int row, int tg, int ng)
{
	//uchar* data = (uchar*)pimg->ptr(row);
	float const * data = pimg->ptr<float>(row);

	int H = pimg->rows;
	int W = pimg->cols;
	int delta = ng - tg;
	
	int deltay = ng / W;
	int deltax = abs((ng-deltay*W) - tg);
	//cout << deltax << deltay << endl;
	if (deltax > 2 || deltay > 2)
		cout << "Error" << endl;
	float bt, gt, rt, bn, gn, rn;
	bt = data[tg * 3]; gt = data[tg * 3 + 1];rt = data[tg * 3 + 2];
	bn = data[ng * 3]; gn = data[ng * 3 + 1];rn = data[ng * 3 + 2];
	//bt = bt / (bt + gt + rt);gt = gt / (bt + gt + rt);rt = 1-bt-gt;
	//bn = bn / (bn + gn + rn);gn = gn / (bn + gn + rn);rn = 1 - bn - gn;
	float dcolor = sqrt((bt - bn)*(bt - bn) + (gt - gn)*(gt - gn) + (rt - rn)*(rt - rn));
	float dspatial = sqrt(deltax*deltax + deltay*deltay);
	float val = exp(-dcolor / 10. - (dspatial - 1) / 4.);
	if (val == 0)
		cout << "Error" << endl;
	return val;
}

void update_neighbor(const int &idx_point, const int &i, const int &j, const int &w, const int &W, const int code, const Mat* &pimg, int &i_arr, float* &array_data, float* &array_sum, int* &p_sum, int* &p_n, int* &p_idxn) {
	int idx_conj;
	float val;
	switch (code) {
	case 1: // right 1
		idx_conj = idx_point + 1;
		array_data[i_arr] = val = e_simi(pimg, i, j, j + 1);
		break;
	case 2: // right 2
		idx_conj = idx_point + 2;
		array_data[i_arr] = val = e_simi(pimg, i, j, j + 2);
		break;
	case 3: //down 1 left 2
		idx_conj = idx_point + W - 2;
		array_data[i_arr] = val = e_simi(pimg, i, j, j + W - 2);
		break;
	case 4: //down 1 left 1
		idx_conj = idx_point + W - 1;
		array_data[i_arr] = val = e_simi(pimg, i, j, j + W - 1);
		break;
	case 5: //down 1 
		idx_conj = idx_point + W;
		array_data[i_arr] = val = e_simi(pimg, i, j, j + W);
		break;
	case 6: //down 1 right 1
		idx_conj = idx_point + W + 1;
		array_data[i_arr] = val = e_simi(pimg, i, j, j + W + 1);
		break;
	case 7: //down 1 right 2
		idx_conj = idx_point + W + 2;
		array_data[i_arr] = val = e_simi(pimg, i, j, j + W + 2);
		break;
	case 8: //down 2 left 2
		idx_conj = idx_point + 2 * W - 2;
		array_data[i_arr] = val = e_simi(pimg, i, j, j + 2 * W - 2);
		break;
	case 9: //down 2 left 1
		idx_conj = idx_point + 2 * W - 1;
		array_data[i_arr] = val = e_simi(pimg, i, j, j + 2 * W - 1);
		break;
	case 10: //down 2
		idx_conj = idx_point + 2 * W;
		array_data[i_arr] = val = e_simi(pimg, i, j, j + 2 * W);
		break;
	case 11: //down 2 right 1
		idx_conj = idx_point + 2 * W + 1;
		array_data[i_arr] = val = e_simi(pimg, i, j, j + 2 * W + 1);
		break;
	case 12: //down 2 right 2
		idx_conj = idx_point + 2 * W + 2;
		array_data[i_arr] = val = e_simi(pimg, i, j, j + 2 * W + 2);
		break;
	default:
		printf("update_neighbor: ERROR \n");
		return;
	}
	p_idxn[(idx_point)*w + p_n[idx_point*w] * 2] = idx_conj; // index of neighbor
	p_idxn[(idx_point)*w + p_n[idx_point*w] * 2 + 1] = (int)&array_data[i_arr]; // pointer to data
	p_n[idx_point*w] += 1;// add counter of neighbor to 1
	array_sum[idx_point] += val;// add sum
								//
	p_idxn[idx_conj * w + p_n[idx_conj*w] * 2] = idx_point; // index of neighbor
	p_idxn[idx_conj * w + p_n[idx_conj*w] * 2 + 1] = (int)&array_data[i_arr]; // pointer to data
	p_n[idx_conj* w] += 1;// add counter of neighbor to 1
	array_sum[idx_conj] += val;// add sum
	//cout << "Address: " << (int)(array_data+i_arr) << endl;
	//float* p_simi = (float*)p_idxn[idx_point * w + (p_n[idx_point*w]-1) * 2 + 1];
	//float val2 = *p_simi;
	//if (val != val2) cout << "Error comparaison" << endl;

	i_arr++;
}

void verify_data(const Mat &pixel_info, float* &sum_rate, const int &W) {
	int* p_n = (int*)pixel_info.data + 2; //3rd column // number of neighbor
	int* p_idxp = (int*)pixel_info.data;//1st column pixel index
	int* p_idxn = (int*)pixel_info.data + 3;//begin of neighbor 4th column
	for (int i = 24 * W - 5;i < 24 * W;i++)
	{
		int n = p_n[i * 51];
		float w_sum = 0;
		printf("Pixel(%d,%d):\n", p_idxp[i * 51] / W, p_idxp[i * 51] % W);
		for (int j = 0;j < n;j++)
		{
			float* p_simi = (float*)p_idxn[i * 51 + j * 2 + 1];
			int idx_n = p_idxn[i * 51 + j * 2];
			w_sum += *p_simi;
			printf("Neighbor(%d,%d):%f	", idx_n / W, idx_n%W, *p_simi);
		}
		printf("\n");
		printf("SUM WEIGHT: %f \n", sum_rate[i]);
		printf("SUM WEIGHT calculated: %f \n", w_sum);
	}
}

Mat display_flow_quiver(Mat img_display, Mat &flow) {
	Mat img;
	img_display.copyTo(img);
	float* Udata = (float*)flow.data;
	int i, j;
	int H = flow.rows;
	int W = flow.cols;
	for (i = 5;i < H;i = i + 10)
		for (j = 5;j < W;j = j + 10)
		{
			int u = round(Udata[i*W*2+j*2])*2;
			int v = round(Udata[i*W * 2 + j * 2+1])*2;
			if ((u != 0) || (v != 0))
				cvQuiver(img, j, i, u, v, Scalar(0, 255, 0), 5, 1);

		}
	//imshow("display_quiver", img);
	//cv::moveWindow("display_quiver", -1680, 2 * H);
	return img;
}

void cvQuiver(Mat& Image, int x, int y, int u, int v, Scalar Color, int Size, int Thickness) {
	cv::Point pt1, pt2;
	int row = Image.rows;
	int col = Image.cols;
#define border (pt1.x > col || pt1.x <0 || pt1.y <0 || pt1.y >row ||pt2.x > col || pt2.x <0 || pt2.y <0 || pt2.y >row)
	double Theta;
	double PI = 3.1416;
	if (u == 0)
		Theta = PI / 2;
	else
		Theta = atan2(double(v), (double)(u));
	if (border) return;
	pt1.x = x;
	pt1.y = y;

	pt2.x = x + u;
	pt2.y = y + v;

	cv::line(Image, pt1, pt2, Color, Thickness, 8);  //Draw Line


	Size = (int)(Size*0.707);


	if (Theta == PI / 2 && pt1.y > pt2.y)
	{
		pt1.x = (int)(Size*cos(Theta) - Size*sin(Theta) + pt2.x);
		pt1.y = (int)(Size*sin(Theta) + Size*cos(Theta) + pt2.y);
		if (border) return;
		cv::line(Image, pt1, pt2, Color, Thickness, 8);  //Draw Line

		pt1.x = (int)(Size*cos(Theta) + Size*sin(Theta) + pt2.x);
		pt1.y = (int)(Size*sin(Theta) - Size*cos(Theta) + pt2.y);
		if (border) return;
		cv::line(Image, pt1, pt2, Color, Thickness, 8);  //Draw Line
	}
	else {
		pt1.x = (int)(-Size*cos(Theta) - Size*sin(Theta) + pt2.x);
		pt1.y = (int)(-Size*sin(Theta) + Size*cos(Theta) + pt2.y);
		if (border) return;
		cv::line(Image, pt1, pt2, Color, Thickness, 8);  //Draw Line

		pt1.x = (int)(-Size*cos(Theta) + Size*sin(Theta) + pt2.x);
		pt1.y = (int)(-Size*sin(Theta) - Size*cos(Theta) + pt2.y);
		if (border) return;
		cv::line(Image, pt1, pt2, Color, Thickness, 8);  //Draw Line
	}

}