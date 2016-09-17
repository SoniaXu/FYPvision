//#include <cv.h>
//#include <highgui.h>
//#include <stdio.h>
//#include <stdlib.h>
//#include "math.h"
//#include <iostream>
//#include "opencv2/nonfree/nonfree.hpp"
//#include "opencv2/opencv.hpp"
//
//
//using namespace std;
//using namespace cv;
//
//// chooe image
//IplImage* tar_img = 0;
//IplImage* tar_img_temp = 0;
//IplImage* tar_img_pre = 0;
//IplImage* tar_img_data = 0;
//IplImage* tar_img_roi = 0;
//
//IplImage* fea_img_data = 0;
//IplImage* fea_img_display = 0;
//IplImage* fea_img_roi = 0;
//IplImage* fea_img = 0;
//IplImage* fea_img_temp = 0;
//
//CvScalar col_fea_unselect;
//CvScalar col_fea_select;
//CvScalar col_fea_pen;
//
//int g_slider_position = 0;
//int g_slider_erode_object = 0;
//int g_slider_dilate_object = 0;
//int g_slider_erode_image = 0;
//int g_slider_dilate_image = 0;
//
//int g_slider_video = 0;
//int g_slider_catch = 0;
//
//// text
//CvMat* txt_show;
//CvFont font;
//char tempStr[256];
//
//CvScalar col_txt_blk;
//
//// sift
//SIFT sift1, sift2;
//Mat descriptors1, descriptors2, mascara;
//double clo_begin = 0;
//double clo_end = 0;
//vector<KeyPoint> keypointsObject, keypointsScene;
//
//// camera
//CvCapture* capture;
//
//int updateText(char* str){
//
//	int p1 = 0;
//	int p2 = 0;
//	int line = 0;
//
//
//	cvSet(txt_show, col_txt_blk);
//
//	while (str[p1]){
//		if (str[p1] != ' '){
//			tempStr[p2] = str[p1];
//			p2++;
//		}
//		else{
//			tempStr[p2] = 0;
//			p2 = 0;
//			cvPutText(txt_show, tempStr, cvPoint(20, 20 + line * 20), &font, CV_RGB(255, 255, 255));
//			line++;
//		}
//		p1++;
//	}
//	tempStr[p2] = 0;
//	p2 = 0;
//	cvPutText(txt_show, tempStr, cvPoint(20, 20 + line * 20), &font, CV_RGB(255, 255, 255));
//	cvShowImage("NOTE", txt_show);
//
//	return 0;
//}
//
//void dealImage(){
//	cvErode(tar_img_temp, tar_img, NULL, g_slider_erode_image);
//	cvDilate(tar_img, tar_img, NULL, g_slider_dilate_image);
//}
//
//
//void matchImageRect(){
//	dealImage();
//
//	//cvXor(tar_img_temp, tar_img_pre, tar_img_data);
//
//	int tempK = 0;
//	/*int tempQ[6] = { 0, 0, 0,0,0,0};*/
//	int X = 20;
//
//	for (int i = 0; i < tar_img->height / X; i++){
//		for (int j = 0; j < tar_img->width /X; j++){
//			tempK = 0;
//			/*tempQ[3] = tempQ[0];
//			tempQ[4] = tempQ[1];
//			tempQ[5] = tempQ[2];
//			tempQ[0] = 0;
//			tempQ[1] = 0;
//			tempQ[2] = 0;*/
//			for (int y = 0; y < X; y++) {
//				uchar* ptr = (uchar*)(
//					tar_img_pre->imageData + (i * X+ y) * tar_img_pre->widthStep + j * X*3
//					);
//				uchar* ptr2 = (uchar*)(
//					tar_img_data->imageData + (i * X + y) * tar_img_data->widthStep + j * X*3
//					);
//				uchar* ptr3 = (uchar*)(
//					tar_img_temp->imageData + (i *X + y) * tar_img_temp->widthStep + j * X*3
//					);
//				uchar* ptr4 = (uchar*)(
//					tar_img->imageData + (i * X + y) * tar_img->widthStep + j * X*3
//					);
//				for (int x = 0; x < X; x++) {
//					tempK += abs(ptr3[3 * x + 0] - ptr[3 * x + 0]);
//					tempK += abs(ptr3[3 * x + 1] - ptr[3 * x + 1]);
//					tempK += abs(ptr3[3 * x + 2] - ptr[3 * x + 2]);
//					/*tempQ[0] += ptr3[3 * x + 0];
//					tempQ[1] += ptr3[3 * x + 1];
//					tempQ[2] += ptr3[3 * x + 2];*/
//				}
//			}
//			/*tempQ[0] /= X*X;
//			tempQ[1] /= X*X;
//			tempQ[2] /= X*X;
//
//			tempQ[0] = tempQ[0] / 4 + 3 * tempQ[4] / 4;
//			tempQ[1] = tempQ[1] / 4 + 3 * tempQ[5] / 4;
//			tempQ[2] = tempQ[2] / 4 + 3 * tempQ[6] / 4;*/
//			
//			for (int y = 0; y < X; y++) {
//				uchar* ptr = (uchar*)(
//					tar_img_pre->imageData + (i * X + y) * tar_img_pre->widthStep + j * X * 3
//					);
//				uchar* ptr2 = (uchar*)(
//					tar_img_data->imageData + (i * X + y) * tar_img_data->widthStep + j * X * 3
//					);
//				uchar* ptr3 = (uchar*)(
//					tar_img_temp->imageData + (i * X + y) * tar_img_temp->widthStep + j * X * 3
//					);
//				uchar* ptr4 = (uchar*)(
//					tar_img->imageData + (i * X + y) * tar_img->widthStep + j * X * 3
//					);
//				Vec3b pixel;
//
//				for (int x = 0; x < X; x++) {
//
//					if (tempK>10000){
//						/*ptr2[3 * x + 0] = ptr4[3 * x + 0];
//						ptr2[3 * x + 1] = ptr4[3 * x + 1];
//						ptr2[3 * x + 2] = ptr4[3 * x + 2];*/
//
//						ptr2[3 * x + 0] = 255;
//						ptr2[3 * x + 1] = 255;
//						ptr2[3 * x + 2] = 255;
//
//
//					}
//					else{
//						ptr2[3 * x + 0] = 0;
//						ptr2[3 * x + 1] = 0;
//						ptr2[3 * x + 2] = 0;
//					}
//
//				}
//			}
//		}
//	}
//
//	
//
//
//	clo_begin = clock();
//
//	Mat imgScene = tar_img;
//
//	
//	cvCvtColor(tar_img_data,tar_img_roi, CV_BGR2GRAY);
//	cvShowImage("tar_img3", tar_img_roi);
//
//	Mat img_mask = tar_img_roi;
//
//	sift2(imgScene, img_mask, keypointsScene, descriptors2);
//
//	clo_end = clock();
//	printf("find features on scene: %.1f\n", (clo_end - clo_begin));
//	clo_begin = clo_end;
//
//	
//
//
//	FlannBasedMatcher matcher;
//	vector< DMatch > allMatches;
//	matcher.match(descriptors1, descriptors2, allMatches);
//
//	clo_end = clock();
//	printf("match features: %.1f\n", (clo_end - clo_begin));
//	clo_begin = clo_end;
//
//	double maxDist = 0;
//	double minDist = 100;
//	double avgDist = 0;
//	for (int i = 0; i < descriptors1.rows; i++)
//	{
//		double dist = allMatches[i].distance;
//		if (dist < minDist)
//			minDist = dist;
//		if (dist > maxDist)
//			maxDist = dist;
//		avgDist += dist;
//	}
//	avgDist /= descriptors1.rows;
//
//	printf("avgDist: %f\n", avgDist);
//
//	vector< DMatch > goodMatches;
//	for (int i = 0; i < descriptors1.rows; i++)
//	{
//		if (allMatches[i].distance < avgDist*1.2)
//			goodMatches.push_back(allMatches[i]);
//	}
//
//
//	printf("good matches: %d\n", goodMatches.size());
//
//
//	
//
//	if (goodMatches.size() < 4)
//		return;
//
//
//	vector<Point2f> object;
//	vector<Point2f> scene;
//
//	CvMat* intrinsic_matrix = (CvMat*)cvLoad("Intrinsics.xml");
//	CvMat* distortion_coeffs = (CvMat*)cvLoad("Distortion.xml");
//	CvMat *obj, *poj, *ro, *tr;
//
//	obj = cvCreateMat(3, 4, CV_32F);
//	poj = cvCreateMat(2, 4, CV_32F);
//	ro = cvCreateMat(3, 1, CV_32F);
//	tr = cvCreateMat(3, 1, CV_32F);
//
//	for (size_t i = 0; i < goodMatches.size(); i++)
//	{
//		//-- 从好的匹配中获取关键点: 匹配关系是关键点间具有的一 一对应关系，可以从匹配关系获得关键点的索引
//		//-- e.g. 这里的goodMatches[i].queryIdx和goodMatches[i].trainIdx是匹配中一对关键点的索引
//		object.push_back(keypointsObject[goodMatches[i].queryIdx].pt);
//		scene.push_back(keypointsScene[goodMatches[i].trainIdx].pt);
//
//
//
//	}
//
//	Mat H = findHomography(object, scene, CV_RANSAC);
//
//
//
//
//
//
//	clo_end = clock();
//	printf("math calculation: %.1f\n", (clo_end - clo_begin));
//	clo_begin = clo_end;
//
//	Mat img = fea_img;
//	std::vector<Point2f> objCorners(4);
//	objCorners[0] = cvPoint(0, 0);
//	objCorners[1] = cvPoint(img.cols, 0);
//	objCorners[2] = cvPoint(img.cols, img.rows);
//	objCorners[3] = cvPoint(0, img.rows);
//	std::vector<Point2f> sceneCorners(4);
//	perspectiveTransform(objCorners, sceneCorners, H);
//
//	cvLine(tar_img, sceneCorners[0], sceneCorners[1], Scalar(0, 255, 0), 4);
//	cvLine(tar_img, sceneCorners[1], sceneCorners[2], Scalar(0, 255, 0), 4);
//	cvLine(tar_img, sceneCorners[2], sceneCorners[3], Scalar(0, 255, 0), 4);
//	cvLine(tar_img, sceneCorners[3], sceneCorners[0], Scalar(0, 255, 0), 4);
//	Mat show_img = tar_img;
//	drawKeypoints(tar_img,     //输入图像
//		keypointsScene,      //特征点矢量
//		show_img,      //输出图像
//		Scalar::all(-1),      //绘制特征点的颜色，为随机
//		//以特征点为中心画圆，圆的半径表示特征点的大小，直线表示特征点的方向
//		DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
//
//
//	for (int i = 0; i < 4; i++){
//		CV_MAT_ELEM(*obj, float, 0, i) = objCorners[i].x;
//		CV_MAT_ELEM(*obj, float, 1, i) = objCorners[i].y;
//		CV_MAT_ELEM(*obj, float, 2, i) = 100;
//
//		CV_MAT_ELEM(*poj, float, 0, i) = sceneCorners[i].x;
//		CV_MAT_ELEM(*poj, float, 1, i) = sceneCorners[i].y;
//
//		//printf("%.2f %.2f\n", CV_MAT_ELEM(*obj, float, 0, i), CV_MAT_ELEM(*obj, float, 1, i));
//		//printf("%.2f %.2f\n", CV_MAT_ELEM(*poj, float, 0, i), CV_MAT_ELEM(*poj, float, 1, i));
//	}
//
//	cvFindExtrinsicCameraParams2(obj, poj, intrinsic_matrix, distortion_coeffs, ro, tr);
//
//	cvSave("transportation.xml", tr);
//	cvSave("rotation.xml", ro);
//
//	double tO[6];
//	tO[0] = CV_MAT_ELEM(*tr, float, 0, 0);
//	tO[1] = CV_MAT_ELEM(*tr, float, 1, 0);
//	tO[2] = CV_MAT_ELEM(*tr, float, 2, 0);
//	tO[3] = CV_MAT_ELEM(*ro, float, 0, 0);
//	tO[4] = CV_MAT_ELEM(*ro, float, 1, 0);
//	tO[5] = CV_MAT_ELEM(*ro, float, 2, 0);
//
//	double an = 0;
//	for (int i = 3; i < 6; i++){
//		an += tO[i] * tO[i];
//	}
//	an = sqrt(an) / 3.1415926 * 180;
//
//
//	sprintf(tempStr, "x:%.2f y:%.2f z:%.2f tx:%.2f ty:%.2f tz:%.2f angle:%f goodMatches:%d"
//		, tO[0], tO[1], tO[2], tO[3], tO[4], tO[5], an, goodMatches.size());
//	updateText(tempStr);
//
//
//
//
//
//	cvShowImage("tar_img", tar_img);
//}
//
//
//void on_mouse(int event, int x, int y, int flags, void* ustc)
//{
//	static CvPoint pre_pt = { -1, -1 };
//	
//	
//	static CvPoint cur_pt = { -1, -1 };
//	CvFont font;
//	cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX, 0.5, 0.5, 0, 1, CV_AA);
//	char temp[16];
//
//	if (event == CV_EVENT_LBUTTONDOWN)
//	{
//		dealImage();
//		sprintf(temp, "(%d,%d)", x, y);
//		cur_pt = cvPoint(x, y);
//		cvPutText(tar_img, temp, cur_pt, &font, col_fea_pen);
//
//		cvCircle(tar_img, cur_pt, 20, col_fea_pen, 1, CV_AA, 0);
//		cvCircle(fea_img_data, cur_pt, 20, col_fea_select, CV_FILLED, CV_AA, 0);
//		
//		cvAdd(tar_img, fea_img_data, fea_img_display);
//		cvShowImage("tar_img", fea_img_display);
//	}
//	else if (event == CV_EVENT_MOUSEMOVE && !(flags & CV_EVENT_FLAG_LBUTTON))
//	{
//		dealImage();
//		sprintf(temp, "(%d,%d)", x, y);
//		cur_pt = cvPoint(x, y);
//		cvPutText(tar_img, temp, cur_pt, &font, col_fea_pen);
//
//		cvCircle(tar_img, cur_pt, 20, col_fea_pen, 1, CV_AA, 0);
//
//		cvAdd(tar_img, fea_img_data, fea_img_display);
//		cvShowImage("tar_img", fea_img_display);
//	}
//	else if (event == CV_EVENT_MOUSEMOVE && (flags & CV_EVENT_FLAG_LBUTTON))
//	{
//		dealImage();
//		sprintf(temp, "(%d,%d)", x, y);
//		cur_pt = cvPoint(x, y);
//		cvPutText(tar_img, temp, cur_pt, &font, cvScalar(0, 0, 0, 255));
//
//		cvCircle(tar_img, cur_pt, 20, col_fea_pen, 1, CV_AA, 0);
//		cvCircle(fea_img_data, cur_pt, 20, col_fea_select, CV_FILLED, CV_AA, 0);
//
//		cvAdd(tar_img, fea_img_data, fea_img_display);
//		cvShowImage("tar_img", fea_img_display);
//	}/*
//	else if (event == CV_EVENT_LBUTTONUP)
//	{
//		sprintf(temp, "(%d,%d)", x, y);
//		cur_pt = cvPoint(x, y);
//		cvPutText(tar_img, temp, cur_pt, &font, cvScalar(0, 0, 0, 255));
//		cvCircle(tar_img, cur_pt, 3, cvScalar(255, 0, 0, 0), CV_FILLED, CV_AA, 0);
//		cvRectangle(tar_img, pre_pt, cur_pt, cvScalar(0, 255, 0, 0), 1, 8, 0);
//		cvShowImage("tar_img", tar_img);
//		cvCopy(tar_img, tar_img_temp);
//	}*/
//
//}
//
//void onTrackbarSlideErodeO(int pos){
//	g_slider_erode_object = pos;
//
//	cvErode(fea_img_temp, fea_img, NULL, g_slider_erode_object);
//	cvDilate(fea_img, fea_img, NULL, g_slider_dilate_object);
//	cvShowImage("tar_img2", fea_img);
//}
//
//void onTrackbarSlideDilateO(int pos){
//	g_slider_dilate_object = pos;
//
//	cvErode(fea_img_temp, fea_img, NULL, g_slider_erode_object);
//	cvDilate(fea_img, fea_img, NULL, g_slider_dilate_object);
//	cvShowImage("tar_img2", fea_img);
//}
//
//void onTrackbarSlideErodeI(int pos){
//	g_slider_erode_image = pos;
//
//	dealImage();
//	cvShowImage("tar_img", tar_img);
//	
//}
//
//void onTrackbarSlideDilateI(int pos){
//	g_slider_dilate_image = pos;
//
//	dealImage();
//	cvShowImage("tar_img", tar_img);
//}
//
//
//
//void onTrackbarSlideVideo(int pos){
//	g_slider_video = pos;
//}
//
//
//
//void onTrackbarSlideCatch(int pos){
//	g_slider_catch = pos;
//
//}
//
//void onTrackbarSlide(int pos){
//	if (pos == 0){
//		cvSetMouseCallback("tar_img", on_mouse, 0);
//		cvSet(fea_img_data, col_fea_unselect);
//	}
//	if (pos == 1){
//
//		int top_x=-1, top_y=-1,bot_x=-1,bot_y=-1;
//
//		fea_img_roi = cvCreateImage(cvGetSize(tar_img), tar_img->depth, tar_img->nChannels);
//		cvSet(fea_img_roi, cvScalar(255, 255, 255));
//
//		for (int y = 0; y < tar_img->height; y++) {
//			uchar* ptr = (uchar*)(
//				tar_img->imageData + y * tar_img->widthStep
//				);
//			uchar* ptr2 = (uchar*)(
//				fea_img_data->imageData + y * fea_img_data->widthStep
//				);
//			uchar* ptr3 = (uchar*)(
//				fea_img_roi->imageData + y * fea_img_roi->widthStep
//				);
//			for (int x = 0; x < tar_img->width; x++) {
//				if (ptr2[3 * x + 0] < 100){
//					ptr3[3 * x + 0] = ptr[3 * x + 0];
//					ptr3[3 * x + 1] = ptr[3 * x + 1];
//					ptr3[3 * x + 2] = ptr[3 * x + 2];
//					if (top_x == -1 || x < top_x){
//						top_x = x;
//					}
//					if (bot_x == -1 || x> bot_x){
//						bot_x = x;
//					}
//					if (top_y == -1 || y < top_y){
//						top_y = y;
//					}
//					if (bot_y == -1 || y> bot_y){
//						bot_y = y;
//					}
//				}
//			}
//		}
//
//		cvSetImageROI(fea_img_roi, cvRect(top_x, top_y, bot_x-top_x, bot_y-top_y));
//		fea_img = cvCloneImage(fea_img_roi);
//		cvResetImageROI(fea_img_roi);
//		fea_img_temp = cvCloneImage(fea_img);
//		cvShowImage("tar_img2", fea_img);
//		cvSetMouseCallback("tar_img", NULL, 0);
//	}
//	if (pos == 2){
//		Mat img = fea_img;
//		Mat output_img;
//		sift1(img, mascara, keypointsObject, descriptors1);
//		drawKeypoints(img,     //输入图像
//			keypointsObject,      //特征点矢量
//			output_img,      //输出图像
//			Scalar::all(-1),      //绘制特征点的颜色，为随机
//			//以特征点为中心画圆，圆的半径表示特征点的大小，直线表示特征点的方向
//			DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
//		namedWindow("SIFT");
//		imshow("SIFT", output_img);
//	}
//}
//
//int initText(){
//	txt_show = cvCreateMat(200, 200, CV_32F);
//	cvInitFont(&font, CV_FONT_VECTOR0, 0.5, 0.5, 0, 1, 8);
//	col_txt_blk = cvScalar(0, 0, 0);
//
//	//开辟一个窗口
//	cvNamedWindow("NOTE", CV_WINDOW_AUTOSIZE);
//
//	return 0;
//}
//
//
//
//int main()
//{
//
//	initText();
//	updateText("X:120 Y:120 Z:200");
//
//
//	col_fea_unselect = cvScalar(100, 0, 0);
//	col_fea_select = cvScalar(0, 0, 255);
//
//	capture = cvCreateCameraCapture(0);
//	tar_img = cvQueryFrame(capture);
//	tar_img_temp = cvCloneImage(tar_img);
//	tar_img_pre = cvCloneImage(tar_img);
//
//
//	tar_img_data = cvCloneImage(tar_img);
//	tar_img_roi = cvCreateImage(cvGetSize(tar_img_data), tar_img_data->depth, 1);
//
//
//	cvSet(tar_img_roi, cvScalar(0,0,0));
//	
//	fea_img_display = cvCreateImage(cvGetSize(tar_img), tar_img->depth, tar_img->nChannels);
//
//	fea_img_data = cvCreateImage(cvGetSize(tar_img), tar_img->depth, tar_img->nChannels);
//	cvSet(fea_img_data, col_fea_unselect);
//	
//	
//	
//
//	cvNamedWindow("tar_img", 1);
//	cvNamedWindow("tar_img2", 1);
//	cvNamedWindow("tar_img3", 1);
//	cvNamedWindow("control", 1);
//	cvSetMouseCallback("tar_img", on_mouse, 0);
//
//	cvCreateTrackbar(
//		"Position",
//		"control",
//		&g_slider_position,
//		2,
//		onTrackbarSlide
//		);
//
//	cvCreateTrackbar(
//		"ErodeOject",
//		"control",
//		&g_slider_erode_object,
//		5,
//		onTrackbarSlideErodeO
//		);
//
//	cvCreateTrackbar(
//		"DilateOject",
//		"control",
//		&g_slider_dilate_object,
//		5,
//		onTrackbarSlideDilateO
//		);
//	cvCreateTrackbar(
//		"ErodeImage",
//		"control",
//		&g_slider_erode_image,
//		5,
//		onTrackbarSlideErodeI
//		);
//
//	cvCreateTrackbar(
//		"DilateImage",
//		"control",
//		&g_slider_dilate_image,
//		5,
//		onTrackbarSlideDilateI
//		);
//
//	cvCreateTrackbar(
//		"VideoPlay",
//		"control",
//		&g_slider_video,
//		1,
//		onTrackbarSlideVideo
//		);
//
//	cvCreateTrackbar(
//		"Catch",
//		"control",
//		&g_slider_catch,
//		1,
//		onTrackbarSlideCatch
//		);
//
//	cvShowImage("tar_img", tar_img);
//
//	while (1) {
//		if (g_slider_video == 0){
//			
//			tar_img = cvQueryFrame(capture);
//			cvFlip(tar_img,tar_img,1);
//			if (!tar_img) break;
//			cvCopy(tar_img_temp, tar_img_pre);
//			cvCopy(tar_img, tar_img_temp);
//
//			dealImage();
//
//			
//			if (g_slider_catch == 1){
//				matchImageRect();
//			}
//			else{
//				cvShowImage("tar_img", tar_img);
//			}
//			
//		}
//		char c = cvWaitKey(10);
//		if (c == 27) break;
//	}
//
//	cvReleaseCapture(&capture);
//	cvDestroyAllWindows();
//	
//
//	/*cvReleaseImage(&tar_img);
//	cvReleaseImage(&tar_img_temp);
//
//	cvReleaseImage(&fea_img);
//	cvReleaseImage(&fea_img_temp);
//	cvReleaseImage(&fea_img_data);
//	cvReleaseImage(&fea_img_display);
//	cvReleaseImage(&fea_img_roi);*/
//
//	return 0;
//}