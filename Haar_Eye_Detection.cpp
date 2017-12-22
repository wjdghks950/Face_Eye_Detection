/*
The following program detects a human face, eye and iris using OpenCv 3.3.1.
Harr cascade is used to detect the features, and connected component analysis along with morphological operations
is used to detect the iris within the detected eye feature.
The haar cascade feature for eyes is invariant to glasses.

By Jeonghwan Kim
*/

#include "cv.hpp"
#include <iostream>

#define minCr 140
#define maxCr 173
#define minCb 77
#define maxCb 127

using namespace cv;

const std::string face_cascade = "C:/opencv/sources/data/haarcascades/haarcascade_frontalface_alt.xml";
const std::string eye_cascade = "C:/opencv/sources/data/haarcascades/haarcascade_eye_tree_eyeglasses.xml";

CascadeClassifier face;
CascadeClassifier eye;

int global_cnt = 0;
bool tracking = false;

struct CallbackParam
{
	Mat frame;
	Point pt1, pt2;
	Rect roi;
	bool updated;
};

void detectEyes_Faces(Mat frame);

int main()
{
	VideoCapture cap(0); //Load webcam
	Mat frame, matGray;

	//frame = imread("person.png");
	while (1)
	{
		if (!cap.read(frame))
		{
			std::cout << "Error!" << std::endl;
			return -1;
		}
		if (frame.empty())
		{
			std::cout << "Error! No frame received!" << std::endl;
			return -1;
		}
		if (!face.load(face_cascade) || !eye.load(eye_cascade)) //load the trained xml files
		{ 
			std::cout << "Cascade file could not be opened!" << std::endl;
			return -1;
		}
		detectEyes_Faces(frame);

		waitKey(33);
	}
	return 0;
}

void detectEyes_Faces(Mat frame)
{
	std::vector<Rect> faces;
	Mat matGray, m_backproj, m_backproj_eye, hsv_eye, hsv;
	CallbackParam param_face, param_eye;
	static MatND m_model3d, m_model3d_eye; //m_model3d histogram should be saved even when param_update != false
	param_face.updated = false;

	static Rect m_rc, m_rc_eye;
	
	float hrange[] = { 0, 180 };//{ 140,173 };
	float vrange[] = { 0, 255 };//{ 77,127 };
	const float* ranges[] = { hrange, vrange, vrange };	// hue, saturation, brightness
	int channels_rgb[] = { 0, 1, 2};
	int hist_sizes[] = { 16, 16, 16 };

	cvtColor(frame, matGray, CV_BGR2GRAY);
	cvtColor(frame, hsv, CV_BGR2HSV);

	face.detectMultiScale(matGray, faces, 1.1, 3, 0 | CV_HAAR_SCALE_IMAGE, Size(50, 50));

	param_face.frame = frame;

	for (int i = 0; i < faces.size(); i++)
	{
		global_cnt++;
		std::cout << global_cnt << std::endl;

		//Initial face detection
		if (global_cnt < 10 && faces.size() > 0)
		{
			param_face.roi = faces[i];
			param_face.updated = true;
		}
		//Every 20 frames, update the face feature(re-detect face using Haar cascade)
		if ((tracking == false && faces.size() > 0) || global_cnt % 20 == 0)
		{
			param_face.roi = faces[i];
			param_face.updated = true;
		}

		if (param_face.updated)
		{
			Rect rc = param_face.roi;
			Mat mask_face = Mat::zeros(rc.height, rc.width, CV_8U);
			ellipse(mask_face, Point(rc.width / 2, rc.height / 2), Size(rc.width / 2, rc.height / 2), 0, 0, 360, 255, CV_FILLED);

			//imshow("mask_face", mask_face);

			Mat roi_face(hsv, rc);
			calcHist(&roi_face, 1, channels_rgb, mask_face, m_model3d, 3, hist_sizes, ranges);

			m_rc = rc;
			param_face.updated = false;
			tracking = true;
		}

		if (tracking && !param_face.updated) {
			//histogram backprojection
			calcBackProject(&hsv, 1, channels_rgb, m_model3d, m_backproj, ranges);
			imshow("Back projection", m_backproj);
			//tracking
			meanShift(m_backproj, m_rc, TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 10, 1));

			rectangle(frame, m_rc, Scalar(0, 255, 0), 2, CV_AA);
		}

		//rectangle(frame, faces[i], Scalar(0, 255, 0), 2);
		Mat faceROI = matGray(m_rc);
		//Mat faceROI = matGray(faces[i]);

		std::vector<Rect> eyes;
		eye.detectMultiScale(faceROI, eyes, 1.1, 3, 0 | CV_HAAR_SCALE_IMAGE, Size(3, 3));

		for (int j = 0; j < eyes.size(); j++)
		{
			rectangle(frame, Rect(m_rc.x + eyes[j].x, m_rc.y + eyes[j].y, eyes[j].width, eyes[j].height), Scalar(0, 0, 255), 2);

			Mat eyeROI = matGray(Rect(m_rc.x + eyes[j].x, m_rc.y + eyes[j].y, eyes[j].width, eyes[j].height));
			Mat eyeRGB = frame(Rect(m_rc.x + eyes[j].x, m_rc.y + eyes[j].y, eyes[j].width, eyes[j].height));
			Mat eyeYCrCb;
			Mat mask(eyeRGB.size(), CV_8U, Scalar(0));
			cvtColor(eyeRGB, eyeYCrCb, CV_BGR2YCrCb);

			std::vector <Mat> channels(3);
			Mat filtered;
			bilateralFilter(eyeYCrCb, filtered, 13, 175, 175); //Use bilateral filter to blur out the rest while preserving edges
			imshow("FILTERED", filtered);
			split(filtered, channels);
			int rows = eyeYCrCb.rows;
			int cols = eyeYCrCb.cols;
			for (int a = 0; a < rows; a++)
			{
				for (int b = 0; b < cols; b++)
				{
					uchar * Cr = channels[1].ptr <uchar>(a);
					uchar * Cb = channels[2].ptr <uchar>(a);
					int cb = channels[2].at<uchar>(j, i);
					if ((minCr > Cr[b]) || (Cr[b] > maxCr) ||
						(minCb > Cb[b]) || (Cb[b] > maxCb))
					{ //Any pixel not within the skin-color pixel to 255
						mask.at<uchar>(a, b) = 255;
					}
				}
			}
			
			Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
			Mat canny;

			//erode(mask, mask, kernel, Point(-1, -1), 1); //Erode more
			Canny(mask, canny, 45, 100); //Edge detection to isolate iris

			std::vector<std::vector<Point>> contours;
			std::vector<Vec4i>hierarchy;

			//Find the contour for the iris within the masked binary image of eyeROI
			findContours(mask, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
			std::vector<Rect2d> rect(contours.size());

			//Check to see if the iris contour is within the eyeROI
			for (int k = 0; k < contours.size(); k++)
			{
				rect[k] = boundingRect(Mat(contours[k])); //Find a boundingRect on the detect iris

			}

			//Put a marker in the center (pupil)
			for (int k = 0; k < contours.size(); k++)
			{
				drawMarker(frame(Rect(m_rc.x + eyes[j].x, m_rc.y + eyes[j].y, eyes[j].width, eyes[j].height)), Point(rect[k].x + rect[k].width * 0.5, rect[k].y + rect[k].height * 0.5), Scalar(0, 0, 255), MARKER_CROSS, 10, 1);
				circle(frame(Rect(m_rc.x + eyes[j].x, m_rc.y + eyes[j].y, eyes[j].width, eyes[j].height)), Point(rect[k].x + rect[k].width * 0.5, rect[k].y + rect[k].height * 0.5), 8, Scalar(0,0,255));
				std::cout << "Coordinates (x, y): " << m_rc.x + eyes[j].x << m_rc.y + eyes[j].y <<std::endl;
			}

			imshow("EYE_ROI", eyeROI);
			imshow("SKIN COLOR", eyeYCrCb);
			imshow("MASK", mask);
			imshow("CANNY", canny);
		}
	}

	imshow("DETECTED", frame);
}