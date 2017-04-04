////正方形检测源码
////载入数张包含各种形状的图片，检测出其中的正方形 
////http://blog.csdn.net/qq_15947787/article/details/51085352
//#include "cv.h"
//#include "highgui.h"
//#include <stdio.h>
//#include <math.h>
//#include <string.h>
//#include <iostream>
//
//
//
//int thresh = 50;
//
//CvMemStorage* storage = NULL;
//const char * wndname = "正方形检测 demo";
//
////angle函数用来返回（两个向量之间找到角度的余弦值）
//double angle(CvPoint* pt1, CvPoint* pt2, CvPoint* pt0)
//{
//	double dx1 = pt1->x - pt0->x;
//	double dy1 = pt1->y - pt0->y;
//	double dx2 = pt2->x - pt0->x;
//	double dy2 = pt2->y - pt0->y;
//	return (dx1*dx2 + dy1*dy2) / sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10);
//}
//
//// 返回图像中找到的所有轮廓序列，并且序列存储在内存存储器中
//
//CvSeq* findSquares4(IplImage* img, CvMemStorage* storage)
//{
//	CvSeq* contours;
//	int i, c, l, N = 11;
//	CvSeq* result;
//	double s[4], t;
//	// 创建一个空序列用于存储轮廓角点
//	CvSeq* squares = cvCreateSeq(0, sizeof(CvSeq), sizeof(CvPoint), storage);
//
//
//	// 找到所有轮廓并且存储在序列中
//	cvFindContours(img, storage, &contours, sizeof(CvContour),
//		CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE, cvPoint(0, 0));
//
//	IplImage* pBinary = cvCreateImage(cvGetSize(img), IPL_DEPTH_8U, 1);
//	cvDrawContours(pBinary, contours, CV_RGB(255, 255, 255), CV_RGB(255, 255, 255), 2);
//	cvNamedWindow("contours image", 1);
//	cvShowImage("contours image", pBinary);
//
//
//	// 遍历找到的每个轮廓contours
//	while (contours)
//	{
//		//用指定精度逼近多边形曲线
//		result = cvApproxPoly(contours, sizeof(CvContour), storage,
//			CV_POLY_APPROX_DP, cvContourPerimeter(contours)*0.02, 0);
//
//
//		if (result->total == 4 &&
//			fabs(cvContourArea(result, CV_WHOLE_SEQ)) > 100 &&
//			fabs(cvContourArea(result, CV_WHOLE_SEQ)) < 100000 &&
//			cvCheckContourConvexity(result))
//		{
//
//			printf("coutours:\t");
//			for (i = 0; i <4; i++)
//			{
//				s[i] = fabs(angle(
//					(CvPoint*)cvGetSeqElem(result, i),
//					(CvPoint*)cvGetSeqElem(result, (i + 1) % 4),
//					(CvPoint*)cvGetSeqElem(result, (i + 2) % 4)));
//				if (s[i])
//					printf("%f\t", s[i]);
//			}
//			printf("\n");
//
//			for (i = 0; i < 4; i++)
//				cvSeqPush(squares,
//				(CvPoint*)cvGetSeqElem(result, i));
//		}
//
//		// 继续查找下一个轮廓
//		contours = contours->h_next;
//	}
//
//	return squares;
//}
//
//
//
//
//int judgeRGB(IplImage* img, int y, int x){
//	int ans = 0;
//
//	CvPoint *tar_point;
//
//	double tmpb = cvGet2D(img, y, x).val[0];
//	double tmpg = cvGet2D(img, y, x).val[1];
//	double tmpr = cvGet2D(img, y, x).val[2];
//
//	if (tmpr > tmpg){
//		if (tmpr > tmpb)
//			ans = 1;
//		else
//			ans = 3;
//	}
//	else{
//		if (tmpg > tmpb)
//			ans = 2;
//		else
//			ans = 3;
//	}
//
//	if (ans == 1){ // R
//		if (tmpg > 50 || tmpb > 50)
//			ans = 0;
//	}
//	else if (ans == 2){ // G
//		if (tmpr > 50 || tmpb > 50)
//			ans = 0;
//	}
//	else if (ans == 3){ // B
//		if (tmpg > 50 || tmpr > 50)
//			ans = 0;
//	}
//
//
//	return ans;
//}
//
//CvPoint pointRGB[3][2];
//int point_num[3];
//
//
////drawSquares函数用来画出在图像中找到的所有正方形轮廓
//void drawSquares(IplImage* img, CvSeq* squares)
//{
//	CvSeqReader reader;
//	IplImage* cpy = cvCloneImage(img);
//	int i;
//	cvStartReadSeq(squares, &reader, 0);
//
//	// read 4 sequence elements at a time (all vertices of a square)
//	for (i = 0; i < squares->total; i += 4)
//	{
//		CvPoint pt[4], *rect = pt, tp;
//		int count = 4;
//
//		// read 4 vertices
//		CV_READ_SEQ_ELEM(pt[0], reader);
//		CV_READ_SEQ_ELEM(pt[1], reader);
//		CV_READ_SEQ_ELEM(pt[2], reader);
//		CV_READ_SEQ_ELEM(pt[3], reader);
//
//		tp.x = (pt[0].x + pt[1].x + pt[2].x + pt[3].x) / 4;
//		tp.y = (pt[0].y + pt[1].y + pt[2].y + pt[3].y) / 4;
//
//
//		// draw the square as a closed polyline
//		cvPolyLine(cpy, &rect, &count, 1, 1, CV_RGB(0, 255, 0), 2, CV_AA, 0);
//
//		int color = judgeRGB(img, tp.y, tp.x);
//
//		if (color == 1){
//			cvCircle(cpy, tp, 2, cv::Scalar(0, 0, 255));
//		}
//		else if (color == 2){
//			cvCircle(cpy, tp, 2, cv::Scalar(0, 255, 0));
//		}
//		else if (color == 3){
//			cvCircle(cpy, tp, 2, cv::Scalar(255, 0, 0));
//		}
//		else{
//			printf("not my point\n");
//		}
//
//		if (color >= 1 && color <= 3){
//			if (point_num[color - 1] < 2){
//				pointRGB[color - 1][point_num[color - 1]].x = tp.x;
//				pointRGB[color - 1][point_num[color - 1]].y = tp.y;
//				point_num[color - 1]++;
//			}
//			else{
//				printf("can not identify\n");
//				break;
//			}
//		}
//
//	}
//
//
//	cvNamedWindow("square image", 1);
//	cvShowImage("square image", cpy);
//	cvReleaseImage(&cpy);
//}
//
//
//char* pic_name = "birdseye//p5.jpg";
//IplImage* img5 = NULL;
//IplImage* img4 = NULL;
//IplImage* img3 = NULL;
//IplImage* img2 = NULL;
//IplImage* img1 = NULL;
//IplImage* img0 = NULL;
//
//
//int isCross(float x1, float y1, float x2, float y2, float x3, float y3, float  x4, float y4){
//	return 1;
//}
//
//void calculate_RGB(){
//
//	if (point_num[0] == 2 && point_num[1] == 2){
//
//
//	}
//	else if (point_num[1] == 2 && point_num[2] == 2){
//
//	}
//	else if (point_num[2] == 2 && point_num[0] == 2){
//
//	}
//	else if (point_num[0] == 1 && point_num[1] == 1 && point_num[2] == 1){
//		printf("model 111");
//
//	}
//	else{
//		printf("color num error\n");
//	}
//}
//
//int main(int argc, char** argv)
//{
//	int i, c;
//	storage = cvCreateMemStorage(0);
//
//
//	point_num[0] = 0; //R
//	point_num[1] = 0; //G
//	point_num[2] = 0; //B
//
//
//	img0 = cvLoadImage(pic_name, 1);
//
//	if (!img0)
//	{
//		printf("不能载入");
//		exit(1);
//	}
//
//	img1 = cvCloneImage(img0);
//
//
//	cvNamedWindow("origin image", 1);
//	cvShowImage("origin image", img1);
//
//	img2 = cvCreateImage(cvGetSize(img1), IPL_DEPTH_8U, 1);
//	cvCvtColor(img1, img2, CV_BGR2GRAY);
//
//	cvNamedWindow("grey image", 1);
//	cvShowImage("grey image", img2);
//
//	img3 = cvCreateImage(cvGetSize(img1), IPL_DEPTH_8U, 1);
//	cvThreshold(img2, img3, 80, 255, CV_THRESH_BINARY);
//
//	cvNamedWindow("binary image", 1);
//	cvShowImage("binary image", img3);
//
//	img4 = cvCreateImage(cvGetSize(img1), IPL_DEPTH_8U, 1);
//	cvDilate(img3, img4, NULL, 3);
//
//	cvSmooth(img4, img4, CV_GAUSSIAN, 5, 5);
//
//	cvNamedWindow("dilate image", 1);
//	cvShowImage("dilate image", img4);
//
//	img5 = cvCloneImage(img1);
//	//cvCvtColor(img4, img5, CV_GRAY2RGB);
//
//	drawSquares(img5, findSquares4(img4, storage));
//
//	calculate_RGB();
//
//
//
//	c = cvWaitKey(0);
//
//	cvReleaseImage(&img1);
//	cvReleaseImage(&img0);
//
//	cvClearMemStorage(storage);
//
//	cvDestroyWindow(wndname);
//	return 0;
//}
//
