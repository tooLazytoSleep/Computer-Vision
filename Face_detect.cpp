#include "opencv2/objdetect.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

/** 定义函数 */
void detectAndDisplay(Mat frame);
/** 全局变量 */
String face_cascade_name = "haarcascade_frontalface_alt.xml";
CascadeClassifier face_cascade;
String window_name = "人脸识别及心率检测";

/** function main */
int main(void)
{
	cv::namedWindow(window_name, CV_WINDOW_NORMAL);
	cv::namedWindow("心率测试-RGB三通道", CV_WINDOW_NORMAL);
	cv::namedWindow("心率测试-R通道", CV_WINDOW_NORMAL);
	cv::namedWindow("心率测试-G通道", CV_WINDOW_NORMAL);
	cv::namedWindow("心率测试-B通道", CV_WINDOW_NORMAL);
	VideoCapture capture;
	Mat frame;

	//-- 加载训练集
	if (!face_cascade.load(face_cascade_name)) { printf("--加载脸部识别训练集失败！\n"); return -1; };

	capture.open(0);
	if (!capture.isOpened()) { printf("--打开摄像头失败！\n"); return -1; }
	capture.set(CV_CAP_PROP_FRAME_WIDTH, 400);
	capture.set(CV_CAP_PROP_FRAME_HEIGHT, 400);

	cout << "Frame Width: " << capture.get(CV_CAP_PROP_FRAME_WIDTH) << endl;
	cout << "Frame Height: " << capture.get(CV_CAP_PROP_FRAME_HEIGHT) << endl;
	while (capture.read(frame))
	{
		if (frame.empty())
		{
			printf(" --没有图像帧数，结束！");
			break;
		}
		//-- 3. 在每一帧上应用分类器
		detectAndDisplay(frame);

		char c = (char)waitKey(33);
		if (c == 27) { break; } // 跳出
	}
	system("pause");
	destroyAllWindows();

}

/** 检测及显示的方法**/
void detectAndDisplay(Mat frame)
{
	std::vector<Rect> faces;
	Mat frame_gray;

	cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);

	//检测脸部
	face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));

	for (size_t i = 0; i < faces.size(); i++)
	{
		Point center(faces[i].x + faces[i].width / 2, faces[i].y + faces[i].height / 2); //寻找脸部中心部位
		ellipse(frame, center, Size(faces[i].width / 2, faces[i].height / 2), 0, 0, 360, Scalar(0, 255, 0), 4, 8, 0);//添加识别脸部标记为椭圆绿色圈

		Mat faceROI = frame_gray(faces[i]);
		std::vector<Rect> eyes;
	}
	int p = 0;
	for (vector<Rect>::const_iterator r = faces.begin(); r != faces.end(); r++, p++)//截取人脸的方法
	{
		Mat temp = frame(Rect(r->x, r->y, faces[p].width, faces[p].height));// 
		Mat tempr = frame(Rect(r->x, r->y, faces[p].width, faces[p].height));//
		Mat tempg = frame(Rect(r->x, r->y, faces[p].width, faces[p].height));//
		Mat tempb = frame(Rect(r->x, r->y, faces[p].width, faces[p].height));//

																			 //转为彩虹图
		Mat img_color(temp.rows, temp.cols, CV_8UC3);//构造RGB图像 
		Mat img_colorR(tempr.rows, tempr.cols, CV_8UC3);//构造RGB图像
		Mat img_colorG(tempg.rows, tempg.cols, CV_8UC3);//构造RGB图像
		Mat img_colorB(tempb.rows, tempb.cols, CV_8UC3);//构造RGB图像
#define IMG_B(img,y,x) img.at<Vec3b>(y,x)[0]  
#define IMG_G(img,y,x) img.at<Vec3b>(y,x)[1]  
#define IMG_R(img,y,x) img.at<Vec3b>(y,x)[2]  
		uchar tmp2 = 0;

		for (int y = 0; y < temp.rows; y++)
		{
			for (int x = 0; x < temp.cols; x++)
			{
				tmp2 = temp.at<uchar>(y, x);
				if (tmp2 <= 51)
				{
					IMG_B(img_color, y, x) = 255;
					IMG_G(img_color, y, x) = tmp2 * 5;
					IMG_R(img_color, y, x) = 0;
				}
				else if (tmp2 <= 102)
				{
					tmp2 -= 51;
					IMG_B(img_color, y, x) = 255 - tmp2 * 5;
					IMG_G(img_color, y, x) = 255;
					IMG_R(img_color, y, x) = 0;
				}
				else if (tmp2 <= 153)
				{
					tmp2 -= 102;
					IMG_B(img_color, y, x) = 0;
					IMG_G(img_color, y, x) = 255;
					IMG_R(img_color, y, x) = tmp2 * 5;
				}
				else if (tmp2 <= 204)
				{
					tmp2 -= 153;
					IMG_B(img_color, y, x) = 0;
					IMG_G(img_color, y, x) = 255 - uchar(128.0*tmp2 / 51.0 + 0.5);
					IMG_R(img_color, y, x) = 255;
				}
				else
				{
					tmp2 -= 204;
					IMG_B(img_color, y, x) = 0;
					IMG_G(img_color, y, x) = 127 - uchar(127.0*tmp2 / 51.0 + 0.5);
					IMG_R(img_color, y, x) = 255;
				}
			}
		}
		//输出视频中人脸
		cv::imshow("心率测试-RGB三通道", img_color);

		uchar tmp3 = 0;

		for (int y = 0; y < tempr.rows; y++)
		{
			for (int x = 0; x < tempr.cols; x++)
			{
				tmp3 = tempr.at<uchar>(y, x);
				if (tmp3 <= 51)
				{
					IMG_B(img_colorR, y, x) = 0;
					IMG_G(img_colorR, y, x) = 0;
					IMG_R(img_colorR, y, x) = 0;
				}
				else if (tmp3 <= 102)
				{
					tmp3 -= 51;
					IMG_B(img_colorR, y, x) = 0;
					IMG_G(img_colorR, y, x) = 0;
					IMG_R(img_colorR, y, x) = 0;
				}
				else if (tmp3 <= 153)
				{
					tmp3 -= 102;
					IMG_B(img_colorR, y, x) = 0;
					IMG_G(img_colorR, y, x) = 0;
					IMG_R(img_colorR, y, x) = tmp3 * 5;
				}
				else if (tmp3 <= 204)
				{
					tmp3 -= 153;
					IMG_B(img_colorR, y, x) = 0;
					IMG_G(img_colorR, y, x) = 0;
					IMG_R(img_colorR, y, x) = 255;
				}
				else
				{
					tmp3 -= 204;
					IMG_B(img_colorR, y, x) = 0;
					IMG_G(img_colorR, y, x) = 0;
					IMG_R(img_colorR, y, x) = 255;
				}
			}
		}
		//输出视频中人脸
		cv::imshow("心率测试-R通道", img_colorR);

		uchar tmp4 = 0;

		for (int y = 0; y < tempg.rows; y++)
		{
			for (int x = 0; x < tempg.cols; x++)
			{
				tmp4 = tempg.at<uchar>(y, x);
				if (tmp4 <= 51)
				{
					IMG_B(img_colorG, y, x) = 0;
					IMG_G(img_colorG, y, x) = tmp2 * 5;
					IMG_R(img_colorG, y, x) = 0;
				}
				else if (tmp4 <= 102)
				{
					tmp4 -= 51;
					IMG_B(img_colorG, y, x) = 0;
					IMG_G(img_colorG, y, x) = 255;
					IMG_R(img_colorG, y, x) = 0;
				}
				else if (tmp4 <= 153)
				{
					tmp4 -= 102;
					IMG_B(img_colorG, y, x) = 0;
					IMG_G(img_colorG, y, x) = 255;
					IMG_R(img_colorG, y, x) = 0;
				}
				else if (tmp4 <= 204)
				{
					tmp4 -= 153;
					IMG_B(img_colorG, y, x) = 0;
					IMG_G(img_colorG, y, x) = 255 - uchar(128.0*tmp2 / 51.0 + 0.5);
					IMG_R(img_colorG, y, x) = 0;
				}
				else
				{
					tmp4 -= 204;
					IMG_B(img_colorG, y, x) = 0;
					IMG_G(img_colorG, y, x) = 127 - uchar(127.0*tmp2 / 51.0 + 0.5);
					IMG_R(img_colorG, y, x) = 0;
				}
			}
		}
		//输出视频中人脸
		cv::imshow("心率测试-G通道", img_colorG);
		uchar tmp5 = 0;
		for (int y = 0; y < tempb.rows; y++)
		{
			for (int x = 0; x < tempb.cols; x++)
			{
				tmp5 = tempb.at<uchar>(y, x);
				if (tmp5 <= 51)
				{
					IMG_B(img_colorB, y, x) = 255;
					IMG_G(img_colorB, y, x) = 0;
					IMG_R(img_colorB, y, x) = 0;
				}
				else if (tmp5 <= 102)
				{
					tmp5 -= 51;
					IMG_B(img_colorB, y, x) = 255 - tmp2 * 5;
					IMG_G(img_colorB, y, x) = 0;
					IMG_R(img_colorB, y, x) = 0;
				}
				else if (tmp5 <= 153)
				{
					tmp5 -= 102;
					IMG_B(img_colorB, y, x) = 0;
					IMG_G(img_colorB, y, x) = 0;
					IMG_R(img_colorB, y, x) = 0;
				}
				else if (tmp5 <= 204)
				{
					tmp5 -= 153;
					IMG_B(img_colorB, y, x) = 0;
					IMG_G(img_colorB, y, x) = 0;
					IMG_R(img_colorB, y, x) = 0;
				}
				else
				{
					tmp5 -= 204;
					IMG_B(img_colorB, y, x) = 0;
					IMG_G(img_colorB, y, x) = 0;
					IMG_R(img_colorB, y, x) = 0;
				}
			}
		}
		//输出视频中人脸
		cv::imshow("心率测试-B通道", img_colorB);
	}
	cv::imshow(window_name, frame);
}
