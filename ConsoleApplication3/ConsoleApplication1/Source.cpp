#include "cv.h"
#include "highgui.h"
#include <cv.h>
#include <highgui.h>
#include <stdio.h>
#include <stdlib.h>
#include "math.h"
#include <iostream>
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

void algo_plane(IplImage*);
CvMemStorage* storage = NULL;
IplImage* img1 = NULL;
IplImage* img2 = NULL;
IplImage* img3 = NULL;
IplImage* img4 = NULL;
IplImage* img5 = NULL;

CvSeq* findSquares4(IplImage*, CvMemStorage*);
CvSeq* square_result = NULL;

void drawSquares(IplImage*, CvSeq*);
CvPoint fp[5];
int pn = 0;


int calculate_model();
int pm, pb, pt, pl, pr;
int cb, ct, cm, cl, cr;


void process_model(int);
vector<Point2f> obj(4);
vector<Point2f> scene(4);
vector<Point2f> plane(4);
vector<Point2f> draw(4);
int model_num = 0;

void model1();
void model2();
void model3();
void model4();
void model5();
void model6();
void model7();
void findH();



int judgeRGB(IplImage*, int, int);


int main(int argc, char** argv) {
	cvNamedWindow("Origin Video", CV_WINDOW_AUTOSIZE);

	cvNamedWindow("H image", 1);

	CvCapture* capture;
	if (1) {
		capture = cvCreateCameraCapture(0);
	}
	else {
		capture = cvCreateFileCapture(argv[1]);
	}
	assert(capture != NULL);

	IplImage* frame;
	while (1) {
		frame = cvQueryFrame(capture);
		if (!frame) break;
		cvShowImage("Origin Video", frame);

		algo_plane(frame);

		char c = cvWaitKey(10);
		if (c == 27) break;
	}
	cvReleaseCapture(&capture);
	cvDestroyWindow("Origin Video");
}

void algo_plane(IplImage* ori_img){
	int i, c;
	


	storage = cvCreateMemStorage(0);

	img1 = cvCloneImage(ori_img);
	/*cvNamedWindow("origin image", 1);
	cvShowImage("origin image",img1);*/

	img2 = cvCreateImage(cvGetSize(img1), IPL_DEPTH_8U, 1);
	cvCvtColor(img1, img2, CV_BGR2GRAY);
	/*cvNamedWindow("grey image", 1);
	cvShowImage("grey image", img2);*/

	img3 = cvCreateImage(cvGetSize(img1), IPL_DEPTH_8U, 1);
	cvThreshold(img2, img3, 80, 255, CV_THRESH_BINARY);
	/*cvNamedWindow("binary image", 1);
	cvShowImage("binary image", img3);*/

	img4 = cvCreateImage(cvGetSize(img1), IPL_DEPTH_8U, 1);
	cvDilate(img3, img4, NULL, 3);
	cvSmooth(img4, img4, CV_GAUSSIAN, 5, 5);
	/*cvNamedWindow("dilate image", 1);
	cvShowImage("dilate image", img4);*/

	img5 = cvCloneImage(img1);
	//cvCvtColor(img4, img5, CV_GRAY2RGB);

	square_result = findSquares4(img4, storage);
	drawSquares(img5, square_result);

	int model = calculate_model();
	process_model(model);


	cvReleaseImage(&img1);
	cvReleaseImage(&img2);
	cvReleaseImage(&img3);
	cvReleaseImage(&img4);
	cvReleaseImage(&img5);
	cvClearMemStorage(storage);
}

CvSeq* findSquares4(IplImage* img, CvMemStorage* storage)
{
	CvSeq* contours;
	int i, c, l, N = 11;
	CvSeq* result;
	double s[4], t;
	// 创建一个空序列用于存储轮廓角点
	CvSeq* squares = cvCreateSeq(0, sizeof(CvSeq), sizeof(CvPoint), storage);


	// 找到所有轮廓并且存储在序列中
	cvFindContours(img, storage, &contours, sizeof(CvContour),
		CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE, cvPoint(0, 0));

	IplImage* pBinary = cvCreateImage(cvGetSize(img), IPL_DEPTH_8U, 1);
	cvDrawContours(pBinary, contours, CV_RGB(255, 255, 255), CV_RGB(255, 255, 255), 2);
	cvNamedWindow("contours image", 1);
	cvShowImage("contours image", pBinary);


	// 遍历找到的每个轮廓contours
	while (contours)
	{
		//用指定精度逼近多边形曲线
		result = cvApproxPoly(contours, sizeof(CvContour), storage,
			CV_POLY_APPROX_DP, cvContourPerimeter(contours)*0.02, 0);


		if (result->total == 4 &&
			fabs(cvContourArea(result, CV_WHOLE_SEQ)) > 100 &&
			fabs(cvContourArea(result, CV_WHOLE_SEQ)) < 100000 &&
			cvCheckContourConvexity(result))
		{

			for (i = 0; i < 4; i++)
				cvSeqPush(squares,
				(CvPoint*)cvGetSeqElem(result, i));
		}

		// 继续查找下一个轮廓
		contours = contours->h_next;
	}

	return squares;
}

void drawSquares(IplImage* img, CvSeq* squares)
{
	CvSeqReader reader;
	IplImage* cpy = cvCloneImage(img);
	int i;
	cvStartReadSeq(squares, &reader, 0);

	// read 4 sequence elements at a time (all vertices of a square)
	for (i = 0; i < squares->total; i += 4)
	{
		CvPoint pt[4], *rect = pt, tp;
		int count = 4;

		// read 4 vertices
		CV_READ_SEQ_ELEM(pt[0], reader);
		CV_READ_SEQ_ELEM(pt[1], reader);
		CV_READ_SEQ_ELEM(pt[2], reader);
		CV_READ_SEQ_ELEM(pt[3], reader);

		tp.x = (pt[0].x + pt[1].x + pt[2].x + pt[3].x) / 4;
		tp.y = (pt[0].y + pt[1].y + pt[2].y + pt[3].y) / 4;


		// draw the square as a closed polyline
		cvPolyLine(cpy, &rect, &count, 1, 1, CV_RGB(0, 255, 0), 2, CV_AA, 0);

		int color = judgeRGB(img, tp.y, tp.x);

		if (color == 1){
			cvCircle(cpy, tp, 2, cv::Scalar(0, 0, 255));
		}
		else if (color == 2){
			cvCircle(cpy, tp, 2, cv::Scalar(0, 255, 0));
		}
		else if (color == 3){
			cvCircle(cpy, tp, 2, cv::Scalar(255, 0, 0));
		}
		else{
			printf("not my point\n");
		}

		if (color >= 1 && color <= 3){
			if (pn<5){
				fp[pn].x = tp.x;
				fp[pn].y = tp.y;
				pn++;
				printf("point %d at y:%d x:%d\n", pn, fp[pn - 1].y, fp[pn - 1].x);
			}
			else{
				printf("can not identify\n");
				break;
			}
		}

	}
	/*cvNamedWindow("square image", 1);
	cvShowImage("square image", cpy);*/
	cvReleaseImage(&cpy);
}

int judgeRGB(IplImage* img, int y, int x){
	int ans = 0;

	CvPoint *tar_point;

	double tmpb = cvGet2D(img, y, x).val[0];
	double tmpg = cvGet2D(img, y, x).val[1];
	double tmpr = cvGet2D(img, y, x).val[2];

	if (tmpr > tmpg){
		if (tmpr > tmpb)
			ans = 1;
		else
			ans = 3;
	}
	else{
		if (tmpg > tmpb)
			ans = 2;
		else
			ans = 3;
	}
	int X = 10;
	if (ans == 1){ // R
		if (tmpg > tmpr - X || tmpb > tmpr - X)
			ans = 0;
	}
	else if (ans == 2){ // G
		if (tmpr > tmpg - X || tmpb > tmpg - X)
			ans = 0;
	}
	else if (ans == 3){ // B
		if (tmpg >tmpb - X || tmpr >tmpb - X)
			ans = 0;
	}


	return ans;
}

int calculate_model(){
	int model = 0;

	if (pn<3 || pn>5){
		return model;
	}

	int mx = 0, my = 0;
	for (int i = 0; i < pn; i++){
		mx += fp[i].x;
		my += fp[i].y;
	}
	mx /= pn;
	my /= pn;

	pm = 0;
	int dis = (mx - fp[0].x)*(mx - fp[0].x) + (my - fp[0].y)*(my - fp[0].y);
	int tempd = 0;
	for (int i = 1; i < pn; i++){
		tempd = (mx - fp[i].x)*(mx - fp[i].x) + (my - fp[i].y)*(my - fp[i].y);
		if (tempd < dis){
			dis = tempd;
			pm = i;
		}
	}

	cm = judgeRGB(img1, fp[pm].y, fp[pm].x);

	double miny = fp[0].y;
	pt = 0;
	for (int i = 1; i < pn; i++){
		if (fp[i].y < miny){
			miny = fp[i].y;
			pt = i;
		}
	}
	double maxy = fp[0].y;
	pb = 0;
	for (int i = 1; i < pn; i++){
		if (fp[i].y > maxy){
			maxy = fp[i].y;
			pb = i;
		}
	}

	double minx = fp[0].x;
	pr = 0;
	for (int i = 1; i < pn; i++){
		if (fp[i].x < minx){
			minx = fp[i].x;
			pr = i;
		}
	}
	double maxx = fp[0].x;
	pl = 0;
	for (int i = 1; i < pn; i++){
		if (fp[i].x > maxx){
			maxx = fp[i].x;
			pl = i;
		}
	}


	cb = judgeRGB(img1, fp[pb].y, fp[pb].x);
	ct = judgeRGB(img1, fp[pt].y, fp[pt].x);
	cl = judgeRGB(img1, fp[pl].y, fp[pl].x);
	cr = judgeRGB(img1, fp[pr].y, fp[pr].x);


	// cl cr

	printf("Color: mid: %d, top: %d, bot: %d\n", cm, ct, cb);


	if (pn == 5){
		if (cm == 3){
			model = 1;
		}
		else if (cb == 3){
			model = 2;
		}
		else if (ct == 3){
			model = 3;
		}

	}
	else if (pn == 4){
		if (ct == cb){
			model = 4;
		}
		else{
			model = 5;
		}
	}
	else if (pn == 3){
		if (cb == 3){
			model = 6;
		}
		else if (ct == 3){
			model = 7;
		}

	}

	printf("Model: %d\n", model);
	return model;
}


void process_model(int model){
	if (model == 1){
		model1();
	}
	else if (model == 2){
		model2();
	}
	else if (model == 3){
		model2();
	}
	else if (model == 4){
		model4();
	}
	else if (model == 5){
		model5();
	}
	else if (model == 6){
		model6();
	}
	else if (model == 7){
		model7();
	}
}

void model1(){
	model_num = 4;

	obj[0] = cvPoint(40, 100);
	obj[1] = cvPoint(80, 100);
	obj[2] = cvPoint(80, 160);
	obj[3] = cvPoint(40, 160);

	plane[0] = cvPoint(0, 70);
	plane[1] = cvPoint(120, 70);
	plane[2] = cvPoint(120, 160);
	plane[3] = cvPoint(0, 160);

	int k = 0;
	for (int i = 0; i < 5; i++){
		if (i == pm){
			continue;
		}
		if (fp[i].x < fp[pm].x&&fp[i].y < fp[pm].y){
			k = 0;
		}
		else if (fp[i].x > fp[pm].x&&fp[i].y < fp[pm].y){
			k = 1;
		}
		else if (fp[i].x > fp[pm].x&&fp[i].y > fp[pm].y){
			k = 2;
		}
		else if (fp[i].x < fp[pm].x&&fp[i].y > fp[pm].y){
			k = 3;
		}
		else{
			printf("error in model 2\n");
		}
		scene[k] = cvPoint(fp[i].x, fp[i].y);
	}

	findH();
}

void model2(){
	model_num = 4;

	obj[0] = cvPoint(40, 0);
	obj[1] = cvPoint(80, 0);
	obj[2] = cvPoint(80, 60);
	obj[3] = cvPoint(40, 60);

	plane[0] = cvPoint(0, 70);
	plane[1] = cvPoint(120, 70);
	plane[2] = cvPoint(120, 110);
	plane[3] = cvPoint(0, 110);

	int k = 0;
	for (int i = 0; i < 5; i++){
		if (i == pm){
			continue;
		}
		if (fp[i].x < fp[pm].x&&fp[i].y < fp[pm].y){
			k = 0;
		}
		else if (fp[i].x > fp[pm].x&&fp[i].y < fp[pm].y){
			k = 1;
		}
		else if (fp[i].x > fp[pm].x&&fp[i].y > fp[pm].y){
			k = 2;
		}
		else if (fp[i].x < fp[pm].x&&fp[i].y > fp[pm].y){
			k = 3;
		}
		else{
			printf("error in model 2\n");
		}
		scene[k] = cvPoint(fp[i].x, fp[i].y);
	}

	findH();
}

void model3(){
	model_num = 4;

	obj[0] = cvPoint(40, 70);
	obj[1] = cvPoint(80, 70);
	obj[2] = cvPoint(80, 130);
	obj[3] = cvPoint(40, 130);

	plane[0] = cvPoint(0, 70);
	plane[1] = cvPoint(120, 70);
	plane[2] = cvPoint(120, 160);
	plane[3] = cvPoint(0, 160);

	int k = 0;
	for (int i = 0; i < 5; i++){
		if (i == pm){
			continue;
		}
		if (fp[i].x < fp[pm].x&&fp[i].y < fp[pm].y){
			k = 0;
		}
		else if (fp[i].x > fp[pm].x&&fp[i].y < fp[pm].y){
			k = 1;
		}
		else if (fp[i].x > fp[pm].x&&fp[i].y > fp[pm].y){
			k = 2;
		}
		else if (fp[i].x < fp[pm].x&&fp[i].y > fp[pm].y){
			k = 3;
		}
		else{
			printf("error in model 2\n");
		}
		scene[k] = cvPoint(fp[i].x, fp[i].y);
	}

	findH();
}

void model5(){

	model_num = 4;

	obj[0] = cvPoint(60, 80);
	obj[1] = cvPoint(20, 110);
	obj[3] = cvPoint(100, 110);
	obj[2] = cvPoint(60, 140);

	plane[0] = cvPoint(0, 70);
	plane[1] = cvPoint(120, 70);
	plane[2] = cvPoint(120, 150);
	plane[3] = cvPoint(0, 150);

	int k = 0;

	scene[0].x = fp[pt].x;
	scene[0].y = fp[pt].y;
	scene[1].x = fp[pl].x;
	scene[1].y = fp[pl].y;
	scene[3].x = fp[pr].x;
	scene[3].y = fp[pr].y;
	scene[2].x = fp[pb].x;
	scene[2].y = fp[pb].y;

	findH();
}

void model4(){

	model_num = 4;

	obj[0] = cvPoint(60, 0);
	obj[1] = cvPoint(20, 30);
	obj[3] = cvPoint(100, 30);
	obj[2] = cvPoint(60, 60);

	plane[0] = cvPoint(0, 70);
	plane[1] = cvPoint(120, 70);
	plane[2] = cvPoint(120, 110);
	plane[3] = cvPoint(0, 110);

	int k = 0;

	scene[0].x = fp[pt].x;
	scene[0].y = fp[pt].y;
	scene[1].x = fp[pl].x;
	scene[1].y = fp[pl].y;
	scene[3].x = fp[pr].x;
	scene[3].y = fp[pr].y;
	scene[2].x = fp[pb].x;
	scene[2].y = fp[pb].y;

	findH();
}

void model6(){

	model_num = 3;


	obj[0] = cvPoint(20, 0);
	obj[1] = cvPoint(60, 60);
	obj[2] = cvPoint(100, 0);
	obj[3] = cvPoint(60, 20);

	plane[0] = cvPoint(0, 100);
	plane[1] = cvPoint(120, 100);
	plane[2] = cvPoint(120, 140);
	plane[3] = cvPoint(0, 140);

	int k = 0;

	scene[0].x = fp[pl].x;
	scene[0].y = fp[pl].y;
	scene[1].x = fp[pb].x;
	scene[1].y = fp[pb].y;
	scene[2].x = fp[pr].x;
	scene[2].y = fp[pr].y;
	scene[3].x = (scene[0].x + scene[1].x + scene[2].x) / 3;
	scene[3].y = (scene[0].y + scene[1].y + scene[2].y) / 3;

	findH();
}

void model7(){

	model_num = 3;

	obj[0] = cvPoint(60, 0);
	obj[1] = cvPoint(20, 60);
	obj[2] = cvPoint(100, 60);
	obj[3] = cvPoint(60, 40);

	plane[0] = cvPoint(0, 70);
	plane[1] = cvPoint(120, 70);
	plane[2] = cvPoint(120, 110);
	plane[3] = cvPoint(0, 110);

	int k = 0;

	scene[0].x = fp[pt].x;
	scene[0].y = fp[pt].y;
	scene[1].x = fp[pl].x;
	scene[1].y = fp[pl].y;
	scene[2].x = fp[pr].x;
	scene[2].y = fp[pr].y;
	scene[3].x = (scene[0].x + scene[1].x + scene[2].x) / 3;
	scene[3].y = (scene[0].y + scene[1].y + scene[2].y) / 3;

	findH();
}

void findH(){
	printf("%d\n", model_num);
	for (int i = 0; i < 4; i++){
		printf("%f %f %f %f\n", obj[i].y, obj[i].x, scene[i].y, scene[i].x);
	}


	Mat H = findHomography(obj, scene);

	perspectiveTransform(plane, draw, H);
	IplImage* cpy = cvCloneImage(img1);


	cvLine(cpy, obj[0], obj[1], Scalar(0, 255, 0), 4);
	cvLine(cpy, obj[1], obj[2], Scalar(0, 255, 0), 4);

	if (model_num == 4){
		cvLine(cpy, obj[2], obj[3], Scalar(0, 255, 0), 4);
		cvLine(cpy, obj[3], obj[0], Scalar(0, 255, 0), 4);
	}
	else{
		cvLine(cpy, obj[2], obj[0], Scalar(0, 255, 0), 4);
	}


	cvLine(cpy, plane[0], plane[1], Scalar(0, 255, 0), 4);
	cvLine(cpy, plane[1], plane[2], Scalar(0, 255, 0), 4);
	cvLine(cpy, plane[2], plane[3], Scalar(0, 255, 0), 4);
	cvLine(cpy, plane[3], plane[0], Scalar(0, 255, 0), 4);

	cvLine(cpy, draw[0], draw[1], Scalar(0, 255, 0), 4);
	cvLine(cpy, draw[1], draw[2], Scalar(0, 255, 0), 4);
	cvLine(cpy, draw[2], draw[3], Scalar(0, 255, 0), 4);
	cvLine(cpy, draw[3], draw[0], Scalar(0, 255, 0), 4);

	
	cvShowImage("H image", cpy);



	CvMat* intrinsic_matrix = (CvMat*)cvLoad("Intrinsics.xml");
	CvMat* distortion_coeffs = (CvMat*)cvLoad("Distortion.xml");

	CvMat *objm, *pojm, *ro, *tr;

	objm = cvCreateMat(3, 4, CV_32F);
	pojm = cvCreateMat(2, 4, CV_32F);
	ro = cvCreateMat(3, 1, CV_32F);
	tr = cvCreateMat(3, 1, CV_32F);

	for (int i = 0; i < 4; i++){
		CV_MAT_ELEM(*objm, float, 0, i) = obj[i].x;
		CV_MAT_ELEM(*objm, float, 1, i) = obj[i].y;
		CV_MAT_ELEM(*objm, float, 2, i) = 100;

		CV_MAT_ELEM(*pojm, float, 0, i) = scene[i].x;
		CV_MAT_ELEM(*pojm, float, 1, i) = scene[i].y;

		//printf("%.2f %.2f\n", CV_MAT_ELEM(*obj, float, 0, i), CV_MAT_ELEM(*obj, float, 1, i));
		//printf("%.2f %.2f\n", CV_MAT_ELEM(*poj, float, 0, i), CV_MAT_ELEM(*poj, float, 1, i));
	}

	cvFindExtrinsicCameraParams2(objm, pojm, intrinsic_matrix, distortion_coeffs, ro, tr);

	double tO[6];
	tO[0] = CV_MAT_ELEM(*tr, float, 0, 0);
	tO[1] = CV_MAT_ELEM(*tr, float, 1, 0);
	tO[2] = CV_MAT_ELEM(*tr, float, 2, 0);

	tO[3] = CV_MAT_ELEM(*ro, float, 0, 0);
	tO[4] = CV_MAT_ELEM(*ro, float, 1, 0);
	tO[5] = CV_MAT_ELEM(*ro, float, 2, 0);

	double roll = 0;
	for (int i = 3; i < 6; i++){
		roll += tO[i] * tO[i];
	}
	roll = sqrt(roll) / 3.1415926 * 180;

	double yaw = cvFastArctan(tO[5], tO[3]);
	double pitch = cvFastArctan(tO[4], sqrt(tO[5] * tO[5] + tO[3] * tO[3]));

	printf("roll %.2f, pitch: %.2f, yaw: %.2f\n", roll, pitch, yaw);


	printf("x:%.2f y:%.2f z:%.2f"
		, tO[0], tO[1], tO[2]);
}