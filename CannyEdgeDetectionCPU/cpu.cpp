//
//  cpu.cpp
//  OpenCV Benchmarks
//
//  Created by heshuimu on 4/3/17.
//  Copyright © 2017 雪竜. All rights reserved.
//

#include <iostream>

#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core/ocl.hpp"

using namespace cv;
using namespace std;

int main(int a, char** b)
{
	Mat source, grayscale, blurred, canny;
	VideoCapture captureDevice(b[1]);
	
	if(ocl::useOpenCL())
	{
		cout << "Using OpenCL, disabled for even footing. \n";
		//ocl::setUseOpenCL(false);
	}
	
	if(useOptimized())
	{
		cout << "Using Optimized (SSE2 etc.). Disabled. \n";
		setUseOptimized(false);
	}
	
	setNumThreads(0);
	
	unsigned long frameCount = 0;
	
	cout << "Frame,Read Time,Upload Time,Change Color Time,Blur Time,Edge Detetion Time, Download Time,UI Time\n";
	
	while (captureDevice.isOpened())
	{
	
		frameCount++;
		
		unsigned long tstart, tread, tup, tcolor, tblur, tdetect,tdown, tui;
		
		tstart = cv::getTickCount();
		
		captureDevice.read(source);
		
		tread = cv::getTickCount();
		
		if(source.empty()) exit(0);
		
		tup = cv::getTickCount();
		
		cvtColor(source, grayscale, CV_BGR2GRAY);//OpenCV API calll
		
		tcolor = cv::getTickCount();
		
		blur(grayscale, blurred, Size(3,3));//OpenCV API calll
		
		tblur = cv::getTickCount();
		
		Canny(blurred, canny, 30, 90);
		
		tdetect = cv::getTickCount();
		
		tdown = cv::getTickCount();
		
		//imshow("Canny edge video CPU", canny);
		
		tui = cv::getTickCount();
		
		printf("%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu\n", frameCount, tread - tstart, tup - tread, tcolor - tup, tblur - tcolor, tdetect - tblur, tdown - tdetect, tui - tdown);
		
		//waitKey(1);
	}
}
