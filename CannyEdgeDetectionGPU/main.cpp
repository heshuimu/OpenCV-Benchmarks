//
//  main.cpp
//  CannyEdgeDetectionCPU
//
//  Created by heshuimu on 4/2/17.
//  Copyright © 2017 雪竜. All rights reserved.
//

#include <iostream>

#include "opencv2/core/types.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/cudawarping.hpp"
#include "opencv2/cudaimgproc.hpp"
#include "opencv2/cudafilters.hpp"

using namespace cv;
using namespace std;

int main(int a, char** b)
{
	cuda::GpuMat d_greyscale, d_blurred, d_dst;
	
	VideoCapture captureDevice(b[1]);
	
	unsigned long frameCount = 0;

	Ptr<cuda::CannyEdgeDetector> canny = cuda::createCannyEdgeDetector(30, 90);
	Ptr<cuda::Filter> filter = cv::cuda::createBoxFilter(d_greyscale.type(), d_blurred.type(), cv::Size(3, 3));
	
	cout << "Frame,Read Time,Upload Time,Change Color Time,Blur Time,Edge Detetion Time, Download Time,UI Time\n";
	
	while (true)
	{
		frameCount++;
		
		unsigned long tstart, tread, tup, tcolor, tblur, tdetect,tdown, tui;
		
		tstart = cv::getTickCount();
		
		Mat source;
		captureDevice.read(source);
		
		tread = cv::getTickCount();
		
		if(source.empty()) exit(0);
		
		cuda::GpuMat d_source(source);
		
		tup = cv::getTickCount();
		
		cuda::cvtColor(d_source, d_greyscale, CV_BGR2GRAY);
		
		tcolor = cv::getTickCount();
		
		filter->apply(d_greyscale, d_blurred);
		
		tblur = cv::getTickCount();
		
		canny->detect(d_blurred, d_dst);
		
		tdetect = cv::getTickCount();
		
		Mat cannyEdge(d_dst);
		
		tdown = cv::getTickCount();
		
		//imshow("Canny edge video", cannyEdge);
		
		tui = cv::getTickCount();
		
		printf("%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu\n", frameCount, tread - tstart, tup - tread, tcolor - tup, tblur - tcolor, tdetect - tblur, tdown - tdetect, tui - tdown);
		
		//waitKey(1);
	}
}
