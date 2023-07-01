#include "plane_detection.h"
#include <stdint.h>
#include <iomanip> // output double value precision
#include <opencv2/core/ocl.hpp> //GPU OPENCL


//LANDING
//#include "clipper.cpp"
//v0.1 - add polylabel
//#include "mapbox/polylabel.hpp"
//using namespace ClipperLib;


cv::Mat depth_IMG;

PlaneDetection::PlaneDetection()
{
	cloud.vertices.resize(kDepthHeight * kDepthWidth);
	cloud.w = kDepthWidth;
	cloud.h = kDepthHeight;
}

PlaneDetection::~PlaneDetection()
{
	cloud.vertices.clear();
	seg_img_.release();
	opt_seg_img_.release();
	color_img_.release();
	opt_membership_img_.release();
	pixel_boundary_flags_.clear();
	pixel_grayval_.clear();
	plane_colors_.clear();
	plane_pixel_nums_.clear();
	opt_plane_pixel_nums_.clear();
	sum_stats_.clear();
	opt_sum_stats_.clear();
}

// Temporarily don't need it since we set intrinsic parameters as constant values in the code.
//bool PlaneDetection::readIntrinsicParameterFile(string filename)
//{
//	ifstream readin(filename, ios::in);
//	if (readin.fail() || readin.eof())
//	{
//		cout << "WARNING: Cannot read intrinsics file " << filename << endl;
//		return false;
//	}
//	string target_str = "m_calibrationDepthIntrinsic";
//	string str_line, str, str_dummy;
//	double dummy;
//	bool read_success = false;
//	while (!readin.eof() && !readin.fail())
//	{
//		getline(readin, str_line);
//		if (readin.eof())
//			break;
//		istringstream iss(str_line);
//		iss >> str;
//		if (str == "m_depthWidth")
//			iss >> str_dummy >> width_;
//		else if (str == "m_depthHeight")
//			iss >> str_dummy >> height_;
//		else if (str == "m_calibrationDepthIntrinsic")
//		{
//			iss >> str_dummy >> fx_ >> dummy >> cx_ >> dummy >> dummy >> fy_ >> cy_;
//			read_success = true;
//			break;
//		}
//	}
//	readin.close();
//	if (read_success)
//	{
//		cloud.vertices.resize(height_ * width_);
//		cloud.w = width_;
//		cloud.h = height_;
//	}
//	return read_success;
//}

bool PlaneDetection::readColorImage(cv::Mat colorImage)
{
	

	const bool useGpu = true;
	cv::ocl::setUseOpenCL(useGpu); //.getUMat(ACCESS_RW)
	//resize(colorImage.getUMat(cv::ACCESS_RW), colorImage, cv::Size(640,480),cv::INTER_LINEAR);
	
	//resize(color_img_, color_img_, cv::Size(640,480),cv::INTER_LINEAR);
	color_img_ = colorImage;// cv::imread(filename, cv::IMREAD_COLOR);
	
	//imshow("COLOR REDUCED", color_img_);
	
	
	if (color_img_.empty() || color_img_.depth() != CV_8U)
	{
		cout << "ERROR: cannot read color image. No such a file, or the image format is not 8UC3" << endl;
		return false;
	}
	return true;
}

bool PlaneDetection::readDepthImage(cv::Mat depth_img)
{


	const bool useGpu = true;
	cv::ocl::setUseOpenCL(useGpu); //.getUMat(ACCESS_RW)
	//resize(depth_img.getUMat(cv::ACCESS_RW), depth_img, cv::Size(640,480),cv::INTER_LINEAR);

	//cv::Mat depth_img = cv::imread(filename,cv::IMREAD_ANYDEPTH);// LOAD_IMAGE_ANYDEPTH);
	
	//resize(depth_img, depth_img, cv::Size(640,480),cv::INTER_LINEAR);
	//depth_img.convertTo(depth_img, CV_16UC1, 1);
	//
	//depth_img.copyTo(depth_IMG);// = depth_img;
	depth_IMG = depth_img;
	
//	imshow("DEPTH REDUCED", depth_img);//
	
	if (depth_img.empty() || depth_img.depth() != CV_16U)
	{
		cout << "WARNING: cannot read depth image. No such a file, or the image format is not 16UC1" << endl;
		return false;
	}
	int rows = depth_img.rows, cols = depth_img.cols;
	int vertex_idx = 0;
	for (int i = 0; i < rows; ++i)
	{
		for (int j = 0; j < cols; ++j)
		{
			double z = (double)(depth_img.at<unsigned short>(i, j)) / kScaleFactor;
			if (_isnan(z))
			{
				cloud.vertices[vertex_idx++] = VertexType(0, 0, z);
				continue;
			}
			double x = ((double)j - kCx) * z / kFx;
			double y = ((double)i - kCy) * z / kFy;
			cloud.vertices[vertex_idx++] = VertexType(x, y, z);
		}
	}
	return true;
}


//https://stackoverflow.com/questions/35479344/how-to-get-a-color-palette-from-an-image-using-opencv
struct lessVec3b
{
    bool operator()(const cv::Vec3b& lhs, const cv::Vec3b& rhs) const {
        return (lhs[0] != rhs[0]) ? (lhs[0] < rhs[0]) : ((lhs[1] != rhs[1]) ? (lhs[1] < rhs[1]) : (lhs[2] < rhs[2]));
    }
};
std::map<cv::Vec3b, int, lessVec3b> getPalette(const cv::Mat3b& src)
{
    std::map<cv::Vec3b, int, lessVec3b> palette;
    for (int r = 0; r < src.rows; ++r)
    {
        for (int c = 0; c < src.cols; ++c)
        {
            cv::Vec3b color = src(r, c);
            if (palette.count(color) == 0)
            {
                palette[color] = 1;
            }
            else
            {
                palette[color] = palette[color] + 1;
            }
        }
    }
    return palette;
}


vector<cv::Mat> PlaneDetection::runPlaneDetection()//bool PlaneDetection::runPlaneDetection()
{
	seg_img_ = cv::Mat(kDepthHeight, kDepthWidth, CV_8UC3);
	plane_filter.run(&cloud, &plane_vertices_, &seg_img_);
	plane_num_ = (int)plane_vertices_.size();

	// Here we set the plane index of a pixel which does NOT belong to any plane as #planes.
	// This is for using MRF optimization later.
	for (int row = 0; row < kDepthHeight; ++row){
		for (int col = 0; col < kDepthWidth; ++col){
			if (plane_filter.membershipImg.at<int>(row, col) < 0){
				plane_filter.membershipImg.at<int>(row, col) = plane_num_;}}}
				
					
					
	cv::Mat dilated;
	const bool useGpu = true;
	cv::ocl::setUseOpenCL(useGpu);
	//seg_img_.copyTo(dilated);
	//cv::dilate(dilated.getUMat(cv::ACCESS_RW), dilated.getUMat(cv::ACCESS_RW), cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(35, 35)));
	//cv::erode(dilated.getUMat(cv::ACCESS_RW), dilated.getUMat(cv::ACCESS_RW), cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(31, 31)));
//	cv::dilate(dilated.getUMat(cv::ACCESS_RW), dilated.getUMat(cv::ACCESS_RW), cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(35, 35)));
//	cv::erode(dilated.getUMat(cv::ACCESS_RW), dilated.getUMat(cv::ACCESS_RW), cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(35, 35)));
	//cv::dilate(dilated, dilated, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(15, 15)));
	//imshow("seg_img_",seg_img_);
////	cv::imshow("seg_img_",dilated);//

	cv::dilate(seg_img_.getUMat(cv::ACCESS_RW), seg_img_.getUMat(cv::ACCESS_RW), cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(35, 35)));
	cv::erode(seg_img_.getUMat(cv::ACCESS_RW), seg_img_.getUMat(cv::ACCESS_RW), cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(35, 35)));
	//imshow("seg_img_AAA",seg_img_);

	//computePlaneSumStats(false);
	int area = seg_img_.rows * seg_img_.cols;
	std::map<cv::Vec3b, int, lessVec3b> palette = getPalette(seg_img_);
	//cout  << "palette.size()=" << palette.size()<<endl;
	
	//make masks to mask depth texture
	vector<cv::Mat> separateShapes;
	
	int shapeCounter= 0;
////	imshow("DEPTH REDUCEDDD", depth_IMG * 20);
	//depth_IMG.convertTo(depth_IMG, CV_16UC1);	
	//cout << "depth_IMG size = " << depth_IMG.rows << "," << depth_IMG.cols <<  ",seg_img_ size = " << seg_img_.rows << "," << seg_img_.cols << endl;
	int pixel_sum=0;
	int secondBrightestBrightness=0;
	int brightestID=-1;
	int secondBrightestID=-1;
	vector<float> brightenessPerShape;
	for (auto color : palette)
    	{	
    		int colR = (int)color.first(0);//pallete found colors
    		int colG = (int)color.first(1);
    		int colB = (int)color.first(2);
    		if(colR > 0 || colB > 0 || colG > 0){
	    		cv::Mat shape;
	    		seg_img_.copyTo(shape);
	    		//cout << "depth_IMG size = " << depth_IMG.rows << "," << depth_IMG.cols <<  ",seg_img_ size = " << seg_img_.rows << "," << seg_img_.cols <<  ",shape size = " << shape.rows << "," << shape.cols << endl;
	    		int countX = 0;
	    		float summer = 0;
	    		int countCOLORS = 0;
	    		for (int r = 0; r < seg_img_.rows; r=r+1)//r++)
			{
				int countY = 0;
				for (int c = 0; c < seg_img_.cols; c=c+1)//c++)
				{
					cv::Vec3b color = cv::Vec3b(colR, colG, colB);//shape(r, c);
					cv::Vec3b imgColor =  seg_img_.at<cv::Vec3b>(r,c);//shape(r, c);
					ushort depthColor =  depth_IMG.at<ushort>(r, c)*255.0/4096.0;
					if(imgColor(0) == color(0) && imgColor(1) == color(1) && imgColor(2) == color(2) && (colR > 0 || colB > 0 || colG > 0)     ){//if(imgColor == color){
				   		shape.at<cv::Vec3b>(cv::Point(c,r)) = cv::Vec3b(depthColor,depthColor,depthColor);//depthColor;//cv::Vec3b(depthColor(0),0,0);
				   		summer += ((float)depthColor * 1  );/// ((float)seg_img_.rows * (float)seg_img_.cols * 255.0* 255.0 );
				   		countCOLORS++;
				   	}else{
				   		shape.at<cv::Vec3b>(cv::Point(countY,countX)) = 0*depthColor;
				   		summer += 0;
				   	}
				   	countY=countY+1;
				}
				countX=countX+1;
			}
			
			//check max white
			//cout << "summer =" << to_string((summer*summer*summer)  / pow(((float)seg_img_.rows * (float)seg_img_.cols * 255.0),1)) << endl;
			//if((summer*summer) / pow(((float)seg_img_.rows * (float)seg_img_.cols * 255.0),1) >  pixel_sum ){
			//	pixel_sum = (summer*summer*summer)  / pow(((float)seg_img_.rows * (float)seg_img_.cols * 255.0),1);
			//	brightestID = shapeCounter;
			//}
			
			float adjustedBright = summer / (countCOLORS* 1);// ((float)seg_img_.rows * (float)seg_img_.cols * 1);
			//if(adjustedBright >  pixel_sum ){
			//	pixel_sum = adjustedBright;
			//	brightestID = shapeCounter;
			//}
			
////			cout << "adjustedBright: " << adjustedBright << " , shapeCounter = " << shapeCounter << endl;
			
			brightenessPerShape.push_back(adjustedBright);
			
			if(1==0){
				for (int r = 0; r < seg_img_.rows; r=r+1)//r++)
				{
					for (int c = 0; c < seg_img_.cols; c=c+1)//c++)
					{
						cv::Vec3b color = cv::Vec3b(colR, colG, colB);//shape(r, c);
						cv::Vec3b imgColor =  seg_img_.at<cv::Vec3b>(r,c);//shape(r, c);
						if(imgColor(0) == color(0) && imgColor(1) == color(1) && imgColor(2) == color(2) && (colR > 0 || colB > 0 || colG > 0)     ){
							shape.at<cv::Vec3b>(cv::Point(c,r)) = cv::Vec3b(adjustedBright,adjustedBright,adjustedBright);
						}else{
							shape.at<cv::Vec3b>(cv::Point(c,r)) = 0*summer;
						}
					}
				}
			
		
			
				if(adjustedBright  >=  pixel_sum ){
				
					if(pixel_sum  >  adjustedBright*0.95 && pixel_sum > secondBrightestBrightness){
						secondBrightestBrightness = pixel_sum;
						secondBrightestID = shapeCounter-1;
					}
				
					pixel_sum = adjustedBright;
					brightestID = shapeCounter;
				}else{
					//check against current max, if close to margin and bigger than previous save as second brightest
					if(adjustedBright  >  pixel_sum*0.95 && adjustedBright > secondBrightestBrightness){//if(adjustedBright  >  pixel_sum*0.45 && adjustedBright > secondBrightestBrightness){
						secondBrightestBrightness = adjustedBright;
						secondBrightestID = shapeCounter;
					}
					else{
						//SET CURRENT 2OND 
						secondBrightestBrightness = adjustedBright;
					}
				}
			}
	    		separateShapes.push_back(shape);
			//cout << "Color: " << to_string((int)color.first(0)) << " \t - Area: " << 100.f * float(color.second) / float(area) << "%" << endl;
////			cout << "Color: " << color.first<< " \t - Area: " << 100.f * float(color.second) / float(area) << "%" << endl;
			//imshow("shape " + to_string(shapeCounter), shape);
			shapeCounter++;
        	}
    	}
    	
    	
    	//FIND close ids
    	//vector<int> brightenessPerShape;
    	float thresholdBRIGTH = 0.95;//how far of brightest to take as same plane
    	vector<int> brightestClosest;
    	vector<cv::Mat> brightestClosestIMGS;
    	vector<float> orderedbrightenessPerShape;
    	orderedbrightenessPerShape = brightenessPerShape;
    	sort(orderedbrightenessPerShape.begin(), orderedbrightenessPerShape.end(),greater<int>());
    	for (int r = 0; r < orderedbrightenessPerShape.size(); r=r+1)
	{
////		cout << "adjustedBright: " << orderedbrightenessPerShape[r] << endl;// << " , shapeCounter = " << r << endl;
		
		if(r > 0){
			if(orderedbrightenessPerShape[r] > thresholdBRIGTH *orderedbrightenessPerShape[0]){ //if(r == 1){ //ADD 2ond BRIGHTEST
				int findID = -1;
				for (int j = 0; j< brightenessPerShape.size(); j=j+1)
				{
					if(brightenessPerShape[j] == orderedbrightenessPerShape[r] ){
						findID =j;
					}
				}
				brightestClosest.push_back(findID);
				secondBrightestID = findID;
			}
		}else{
		//find in ordered list to extract ID
				int findID = -1;
				for (int j = 0; j< brightenessPerShape.size(); j=j+1)
				{
					if(brightenessPerShape[j] == orderedbrightenessPerShape[r] ){
						findID =j;
					}
				}
				brightestClosest.push_back(findID);
				brightestID = findID;
		}
    	}
    	
    	//imshow("shape MAX WHITE " + to_string(brightestID) +" ," +to_string(pixel_sum), separateShapes[brightestID]);
 ////   	cout << "brightestID DDDDDDD =" << brightestID << ", secondBrightestID =" << secondBrightestID << endl;
    	if(brightestID >=0){
    		//imshow("shape MAX WHITE ", separateShapes[brightestID]);
	}
	if(secondBrightestID >=0){
    		//imshow("shape MAX WHITE second Brightest ", separateShapes[secondBrightestID]);
	}
	for (int r = 0; r < brightestClosest.size(); r=r+1)
	{
		//imshow("shape MAX WHITE " + to_string(r), separateShapes[brightestClosest[r]]);
		
		brightestClosestIMGS.push_back(separateShapes[brightestClosest[r]]);
		
		//FIND CONTOURS and LANDING CIRCLES
		
		if(1==0){
			//CONTOURS
			int threshA = 100;
			cv::Mat thresh;
			cv::cvtColor(separateShapes[brightestClosest[r]], thresh, cv::COLOR_BGR2GRAY);
			//cv::Canny( thresh, thresh, threshA, threshA*2);
			//	cv::imshow("GREY",thresh);
			//threshold(img_gray, thresh, 254, 255, THRESH_BINARY);
			//dilated.convertTo(thresh, CV_8UC1);	
			vector<vector<cv::Point>> contours1;
			vector<cv::Vec4i> hierarchy1;
			cv::findContours(thresh, contours1, hierarchy1, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
			// draw contours on the original image
			cv::Mat image_copy1 = separateShapes[brightestClosest[r]].clone();
			cv::drawContours(image_copy1, contours1, -1, cv::Scalar(155, 255, 0), 2, cv::LINE_AA);
////				cv::imshow("image_copy1",image_copy1);
			
			
			vector<vector<cv::Point> > contours_poly( contours1.size() );
			vector<cv::Rect> boundRect( contours1.size() );
			vector<cv::Point2f>centers( contours1.size() );
			vector<float>radius( contours1.size() );
			for( size_t i = 0; i < contours1.size(); i++ )
			{
				cv::approxPolyDP( contours1[i], contours_poly[i], 3, true );
				boundRect[i] = boundingRect( contours_poly[i] );
				cv::minEnclosingCircle( contours_poly[i], centers[i], radius[i] );
			}
			cv::Mat drawing = cv::Mat::zeros( image_copy1.size(), CV_8UC3 );
			for( size_t i = 0; i< contours1.size(); i++ )
			{
				cv::Scalar color = cv::Scalar(i*100,0,0);//   ( cv::rng.uniform(0, 256), cv::rng.uniform(0,256), cv::rng.uniform(0,256) );
				cv::drawContours( drawing, contours_poly, (int)i, color );

				//if(boundRect[i].width > minSize || boundRect[i].height > minSize){
					//cv::rectangle( drawing, boundRect[i].tl(), boundRect[i].br(), color, 2 );
				//}else{
					//draw removed
				 	//cv::rectangle( image_copy1, boundRect[i].tl(), boundRect[i].br(), color, 2 );
				//}
				//cv::circle( drawing, centers[i], (int)radius[i], color, 2 );
			}
////			cv::imshow("image_copy2",drawing);	
		}
	}//END ITERATE GROUND SHAPES	
	
	
	cv::Vec3b cA;
////	cout << "plane_num_ = " << plane_num_ << " (int)plane_vertices_.size()=" << to_string((int)plane_vertices_.size()) << endl;
	for (int pidx = 0; pidx < plane_num_; ++pidx)
	{
		//cout << "pidx= " << pidx << endl;
		//cout << "(sum_stats_[pidx].sz = " << to_string((float)sum_stats_[pidx].sz) << endl;
	
		//out << pidx << " ";
		//if (!run_mrf)
		//	out << plane_pixel_nums_[pidx] << " ";
		//else
		//	out << opt_plane_pixel_nums_[pidx] << " ";
		//
		// Plane color in output image
		int vidx = plane_vertices_[pidx][0];
		cv::Vec3b c = seg_img_.at<cv::Vec3b>(vidx / kDepthWidth, vidx % kDepthWidth);
		//cA = seg_img_.at<cv::Vec3b>(vidx / kDepthWidth, vidx % kDepthWidth);
		
		//if(cA(0) > 2 || cA(1) > 2 || cA(2) > 2){
			cv::Mat mask;
			cv::Scalar lowerb = cv::Scalar(c(0)-1, c(1)-1, c(2)-1);
			cv::Scalar upperb = cv::Scalar(c(0)+1, c(1)+1, c(2)+1);
			cv::inRange(seg_img_, lowerb, upperb, mask);
			//imshow("mask"+to_string(pidx), mask);
		//}
////		cout << "COLOR " +to_string(pidx) +" = " << int(c.val[2]) << " " << int(c.val[1]) << " "<< int(c.val[0]) << " " << endl;
	}
	
	//int specific_hue = 
	//cv::Mat mask = cv::Mat::zeros(dilated.size(),CV_8UC1);

	//for(int i=0;i<dilated.rows;i++){
	//    for(int j=0;j<dilated.cols;i++){
	//	if(dilated.at<uchar>(i,j)==(uchar)specific_hue){
	//	    mask.at<uchar>(i,j)=(uchar)255;
	//	}
	//    }
	//}
	
//	cv::Mat mask;
//	cv::Scalar lowerb = cv::Scalar(cA(0), cA(1), cA(2));
//	cv::Scalar upperb = cv::Scalar(cA(0), cA(1), cA(2));
//	cv::inRange(dilated, lowerb, upperb, mask);
//	imshow("mask", mask);
	//dilated.copyTo(dilated, mask);
	//cv::bitwise_not(dilated,dilated,mask);
	
////	imshow("COLOR REDUCED", color_img_);
	
	//FIND
						
				
	return brightestClosestIMGS;//true;
}

void PlaneDetection::prepareForMRF()
{
	opt_seg_img_ = cv::Mat(kDepthHeight, kDepthWidth, CV_8UC3);
	opt_membership_img_ = cv::Mat(kDepthHeight, kDepthWidth, CV_32SC1);
	pixel_boundary_flags_.resize(kDepthWidth * kDepthHeight, false);
	pixel_grayval_.resize(kDepthWidth * kDepthHeight, 0);

	cv::Mat& mat_label = plane_filter.membershipImg;
	for (int row = 0; row < kDepthHeight; ++row)
	{
		for (int col = 0; col < kDepthWidth; ++col)
		{
			pixel_grayval_[row * kDepthWidth + col] = RGB2Gray(row, col);
			int label = mat_label.at<int>(row, col);
			if ((row - 1 >= 0 && mat_label.at<int>(row - 1, col) != label)
				|| (row + 1 < kDepthHeight && mat_label.at<int>(row + 1, col) != label)
				|| (col - 1 >= 0 && mat_label.at<int>(row, col - 1) != label)
				|| (col + 1 < kDepthWidth && mat_label.at<int>(row, col + 1) != label))
			{
				// Pixels in a fixed range near the boundary pixel are also regarded as boundary pixels
				for (int x = max(row - kNeighborRange, 0); x < min(kDepthHeight, row + kNeighborRange); ++x)
				{
					for (int y = max(col - kNeighborRange, 0); y < min(kDepthWidth, col + kNeighborRange); ++y)
					{
						// If a pixel is not on any plane, then it is not a boundary pixel.
						if (mat_label.at<int>(x, y) == plane_num_)
							continue;
						pixel_boundary_flags_[x * kDepthWidth + y] = true;
					}
				}
			}
		}
	}

	for (int pidx = 0; pidx < plane_num_; ++pidx)
	{
		int vidx = plane_vertices_[pidx][0];
		cv::Vec3b c = seg_img_.at<cv::Vec3b>(vidx / kDepthWidth, vidx % kDepthWidth);
		plane_colors_.push_back(c);
	}
	plane_colors_.push_back(cv::Vec3b(0,0,0)); // black for pixels not in any plane
}

// Note: input filename_prefix is like '/rgbd-image-folder-path/frame-XXXXXX'
void PlaneDetection::writeOutputFiles(string output_folder, string frame_name, bool run_mrf)
{
	computePlaneSumStats(run_mrf);

	if (output_folder.back() != '\\' && output_folder.back() != '/')
		output_folder += "/";	
	string filename_prefix = output_folder + frame_name + "-plane";
	cv::imwrite(filename_prefix + ".png", seg_img_);
	writePlaneLabelFile(filename_prefix + "-label.txt");
	writePlaneDataFile(filename_prefix + "-data.txt");
	if (run_mrf)
	{
		cv::imwrite(filename_prefix + "-opt.png", opt_seg_img_);
		writePlaneLabelFile(filename_prefix + "-label-opt.txt", run_mrf);
		writePlaneDataFile(filename_prefix + "-data-opt.txt", run_mrf);
	}
	
}
void PlaneDetection::writePlaneLabelFile(string filename, bool run_mrf /* = false */)
{
	ofstream out(filename, ios::out);
	out << plane_num_ << endl;
	if (plane_num_ == 0)
	{
		out.close();
		return;
	}
	for (int row = 0; row < kDepthHeight; ++row)
	{
		for (int col = 0; col < kDepthWidth; ++col)
		{
			int label = run_mrf ? opt_membership_img_.at<int>(row, col) : plane_filter.membershipImg.at<int>(row, col);
			out << label << " ";
		}
		out << endl;
	}
	out.close();
}

void PlaneDetection::writePlaneDataFile(string filename, bool run_mrf /* = false */)
{
	ofstream out(filename, ios::out);
	out << "#plane_index number_of_points_on_the_plane plane_color_in_png_image(1x3) plane_normal(1x3) plane_center(1x3) "
		<< "sx sy sz sxx syy szz sxy syz sxz" << endl;

	for (int pidx = 0; pidx < plane_num_; ++pidx)
	{
		out << pidx << " ";
		if (!run_mrf)
			out << plane_pixel_nums_[pidx] << " ";
		else
			out << opt_plane_pixel_nums_[pidx] << " ";

		// Plane color in output image
		int vidx = plane_vertices_[pidx][0];
		cv::Vec3b c = seg_img_.at<cv::Vec3b>(vidx / kDepthWidth, vidx % kDepthWidth);
		out << int(c.val[2]) << " " << int(c.val[1]) << " "<< int(c.val[0]) << " "; // OpenCV uses BGR by default

		// Plane normal and center
		int new_pidx = pid_to_extractedpid[pidx];
		for (int i = 0; i < 3; ++i)
			out << plane_filter.extractedPlanes[new_pidx]->normal[i] << " ";
		for (int i = 0; i < 3; ++i)
			out << plane_filter.extractedPlanes[new_pidx]->center[i] << " ";

		// Sum of all points on the plane
		if (run_mrf)
		{
			out << opt_sum_stats_[pidx].sx << std::setprecision(8) << " " 
				<< opt_sum_stats_[pidx].sy << std::setprecision(8) << " " 
				<< opt_sum_stats_[pidx].sz << std::setprecision(8) << " " 
				<< opt_sum_stats_[pidx].sxx << std::setprecision(8) << " "
				<< opt_sum_stats_[pidx].syy << std::setprecision(8) << " "
				<< opt_sum_stats_[pidx].szz << std::setprecision(8) << " "
				<< opt_sum_stats_[pidx].sxy << std::setprecision(8) << " "
				<< opt_sum_stats_[pidx].syz << std::setprecision(8) << " "
				<< opt_sum_stats_[pidx].sxz << std::setprecision(8) << endl;
		}
		else
		{
			out << sum_stats_[pidx].sx << std::setprecision(8) << " " 
				<< sum_stats_[pidx].sy << std::setprecision(8) << " " 
				<< sum_stats_[pidx].sz << std::setprecision(8) << " " 
				<< sum_stats_[pidx].sxx << std::setprecision(8) << " "
				<< sum_stats_[pidx].syy << std::setprecision(8) << " "
				<< sum_stats_[pidx].szz << std::setprecision(8) << " "
				<< sum_stats_[pidx].sxy << std::setprecision(8) << " "
				<< sum_stats_[pidx].syz << std::setprecision(8) << " "
				<< sum_stats_[pidx].sxz << std::setprecision(8) << endl;
		}

		// NOTE: the plane-sum parameters computed from AHC code seems different from that computed from points belonging to planes shown above.
		// Seems there is a plane refinement step in AHC code so points belonging to each plane are slightly changed.
		//ahc::PlaneSeg::Stats& stat = plane_filter.extractedPlanes[pidx]->stats;
		//cout << stat.sx << " " << stat.sy << " " << stat.sz << " " << stat.sxx << " "<< stat.syy << " "<< stat.szz << " "<< stat.sxy << " "<< stat.syz << " "<< stat.sxz << endl;
	}
	out.close();
}





void PlaneDetection::findFarPlaneData(string filename, bool run_mrf /* = false */)
{
	//ofstream out(filename, ios::out);
	//out << "#plane_index number_of_points_on_the_plane plane_color_in_png_image(1x3) plane_normal(1x3) plane_center(1x3) "
	//	<< "sx sy sz sxx syy szz sxy syz sxz" << endl;

	double farZDIST= 111111111;
	int farIDZ = -1;
	int farCR = -1;
	int farCB = -1;
	int farCG = -1;
	
	///computePlaneSumStats(run_mrf);

	for (int pidx = 0; pidx < plane_num_; ++pidx)
	{
		cout << "pidx= " << pidx << endl;
		//cout << "(sum_stats_[pidx].sz = " << to_string((float)sum_stats_[pidx].sz) << endl;
	
		//out << pidx << " ";
		//if (!run_mrf)
		//	out << plane_pixel_nums_[pidx] << " ";
		//else
		//	out << opt_plane_pixel_nums_[pidx] << " ";
//
		// Plane color in output image
		int vidx = plane_vertices_[pidx][0];
		cv::Vec3b c = seg_img_.at<cv::Vec3b>(vidx / kDepthWidth, vidx % kDepthWidth);
		//out << int(c.val[2]) << " " << int(c.val[1]) << " "<< int(c.val[0]) << " "; // OpenCV uses BGR by default
		cout << "COLOR = " << int(c.val[2]) << " " << int(c.val[1]) << " "<< int(c.val[0]) << " " << endl;; // OpenCV uses BGR by default

		// Plane normal and center
		if(pid_to_extractedpid.size() > pidx){
		cout << "pid_to_extractedpid = " << pid_to_extractedpid[pidx] << endl;
		int new_pidx = pidx;// pid_to_extractedpid[pidx];
		for (int i = 0; i < 3; ++i){
			//	out << plane_filter.extractedPlanes[new_pidx]->normal[i] << " ";
			if(plane_filter.extractedPlanes.size() > new_pidx){
				cout << "normal = " << plane_filter.extractedPlanes[new_pidx]->normal[i] << " ,at pidx " << pidx << endl;
			}
		}//
		}
		
		//for (int i = 0; i < 3; ++i){
			//	out << plane_filter.extractedPlanes[new_pidx]->center[i] << " ";
		//}
/*
		if(sum_stats_[pidx].sz < farZDIST){
			farIDZ = pidx;
			farCB = int(c.val[2]);
			farCG = int(c.val[1]);
			farCR = int(c.val[0]);
			farZDIST = sum_stats_[pidx].sz;
		}
*/

		// Sum of all points on the plane
		/*	
		
		
		if (run_mrf)
		{
			out << opt_sum_stats_[pidx].sx << std::setprecision(8) << " " 
				<< opt_sum_stats_[pidx].sy << std::setprecision(8) << " " 
				<< opt_sum_stats_[pidx].sz << std::setprecision(8) << " " 
				<< opt_sum_stats_[pidx].sxx << std::setprecision(8) << " "
				<< opt_sum_stats_[pidx].syy << std::setprecision(8) << " "
				<< opt_sum_stats_[pidx].szz << std::setprecision(8) << " "
				<< opt_sum_stats_[pidx].sxy << std::setprecision(8) << " "
				<< opt_sum_stats_[pidx].syz << std::setprecision(8) << " "
				<< opt_sum_stats_[pidx].sxz << std::setprecision(8) << endl;
		}
		else
		{
			out << sum_stats_[pidx].sx << std::setprecision(8) << " " 
				<< sum_stats_[pidx].sy << std::setprecision(8) << " " 
				<< sum_stats_[pidx].sz << std::setprecision(8) << " " 
				<< sum_stats_[pidx].sxx << std::setprecision(8) << " "
				<< sum_stats_[pidx].syy << std::setprecision(8) << " "
				<< sum_stats_[pidx].szz << std::setprecision(8) << " "
				<< sum_stats_[pidx].sxy << std::setprecision(8) << " "
				<< sum_stats_[pidx].syz << std::setprecision(8) << " "
				<< sum_stats_[pidx].sxz << std::setprecision(8) << endl;
		}
		*/

		// NOTE: the plane-sum parameters computed from AHC code seems different from that computed from points belonging to planes shown above.
		// Seems there is a plane refinement step in AHC code so points belonging to each plane are slightly changed.
		//ahc::PlaneSeg::Stats& stat = plane_filter.extractedPlanes[pidx]->stats;
		//cout << stat.sx << " " << stat.sy << " " << stat.sz << " " << stat.sxx << " "<< stat.syy << " "<< stat.szz << " "<< stat.sxy << " "<< stat.syz << " "<< stat.sxz << endl;
	}
	//out.close();

	//cout << "farIDZ" << farIDZ <<  "farZDIST" << farZDIST <<  "farCR" << farCR <<  "farCG" << farCG <<  "farCB" << farCB << endl;

}






void PlaneDetection::computePlaneSumStats(bool run_mrf /* = false */)
{
	sum_stats_.resize(plane_num_);
	for (int pidx = 0; pidx < plane_num_; ++pidx)
	{
		for (int i = 0; i < plane_vertices_[pidx].size(); ++i)
		{
			int vidx = plane_vertices_[pidx][i];
			const VertexType& v = cloud.vertices[vidx];
			sum_stats_[pidx].sx += v[0];		 sum_stats_[pidx].sy += v[1];		  sum_stats_[pidx].sz += v[2];
			sum_stats_[pidx].sxx += v[0] * v[0]; sum_stats_[pidx].syy += v[1] * v[1]; sum_stats_[pidx].szz += v[2] * v[2];
			sum_stats_[pidx].sxy += v[0] * v[1]; sum_stats_[pidx].syz += v[1] * v[2]; sum_stats_[pidx].sxz += v[0] * v[2];
		}
		plane_pixel_nums_.push_back(int(plane_vertices_[pidx].size()));
	}
	for (int pidx = 0; pidx < plane_num_; ++pidx)
	{
		int num = plane_pixel_nums_[pidx];
		sum_stats_[pidx].sx /= num;		sum_stats_[pidx].sy /= num;		sum_stats_[pidx].sz /= num;
		sum_stats_[pidx].sxx /= num;	sum_stats_[pidx].syy /= num;	sum_stats_[pidx].szz /= num;
		sum_stats_[pidx].sxy /= num;	sum_stats_[pidx].syz /= num;	sum_stats_[pidx].sxz /= num;
	}
	// Note that the order of extracted planes in `plane_filter.extractedPlanes` is DIFFERENT from
	// the plane order in `plane_vertices_` after running plane detection function `plane_filter.run()`.
	// So here we compute a mapping between these two types of plane indices by comparing plane centers.
	vector<double> sx(plane_num_), sy(plane_num_), sz(plane_num_);
	for (int i = 0; i < plane_filter.extractedPlanes.size(); ++i)
	{
		sx[i] = plane_filter.extractedPlanes[i]->stats.sx / plane_filter.extractedPlanes[i]->stats.N;
		sy[i] = plane_filter.extractedPlanes[i]->stats.sy / plane_filter.extractedPlanes[i]->stats.N;
		sz[i] = plane_filter.extractedPlanes[i]->stats.sz / plane_filter.extractedPlanes[i]->stats.N;
	}
	extractedpid_to_pid.clear();
	pid_to_extractedpid.clear();
	// If two planes' centers are closest, then the two planes are corresponding to each other.
	for (int i = 0; i < plane_num_; ++i)
	{
		double min_dis = 1000000;
		int min_idx = -1;
		for (int j = 0; j < plane_num_; ++j)
		{
			double a = sum_stats_[i].sx - sx[j], b = sum_stats_[i].sy - sy[j], c = sum_stats_[i].sz - sz[j];
			double dis = a * a + b * b + c * c;
			if (dis < min_dis)
			{
				min_dis = dis;
				min_idx = j;
			}
		}
		if (extractedpid_to_pid.find(min_idx) != extractedpid_to_pid.end())
		{
			cout << "   WARNING: a mapping already exists for extracted plane " << min_idx << ":" << extractedpid_to_pid[min_idx] << " -> " << min_idx << endl;
		}
		pid_to_extractedpid[i] = min_idx;
		extractedpid_to_pid[min_idx] = i;
	}
	if (run_mrf)
	{
		opt_sum_stats_.resize(plane_num_);
		opt_plane_pixel_nums_.resize(plane_num_, 0);
		for (int row = 0; row < kDepthHeight; ++row)
		{
			for (int col = 0; col < kDepthWidth; ++col)
			{
				int label = opt_membership_img_.at<int>(row, col); // plane label each pixel belongs to
				if (label != plane_num_) // pixel belongs to some plane
				{
					opt_plane_pixel_nums_[label]++;
					int vidx = row * kDepthWidth + col;
					const VertexType& v = cloud.vertices[vidx];
					opt_sum_stats_[label].sx += v[0];		  opt_sum_stats_[label].sy += v[1];		    opt_sum_stats_[label].sz += v[2];
					opt_sum_stats_[label].sxx += v[0] * v[0]; opt_sum_stats_[label].syy += v[1] * v[1]; opt_sum_stats_[label].szz += v[2] * v[2];
					opt_sum_stats_[label].sxy += v[0] * v[1]; opt_sum_stats_[label].syz += v[1] * v[2]; opt_sum_stats_[label].sxz += v[0] * v[2];
				}
			}
		}
		for (int pidx = 0; pidx < plane_num_; ++pidx)
		{
			int num = opt_plane_pixel_nums_[pidx];
			opt_sum_stats_[pidx].sx /= num;		opt_sum_stats_[pidx].sy /= num;		opt_sum_stats_[pidx].sz /= num;
			opt_sum_stats_[pidx].sxx /= num;	opt_sum_stats_[pidx].syy /= num;	opt_sum_stats_[pidx].szz /= num;
			opt_sum_stats_[pidx].sxy /= num;	opt_sum_stats_[pidx].syz /= num;	opt_sum_stats_[pidx].sxz /= num;
		}
	}

	//--------------------------------------------------------------
	// Only for debug. It doesn't influence the plane detection.
	for (int pidx = 0; pidx < plane_num_; ++pidx)
	{
		double w = 0;
		//for (int j = 0; j < 3; ++j)
		//	w -= plane_filter.extractedPlanes[pidx]->normal[j] * plane_filter.extractedPlanes[pidx]->center[j];
		w -= plane_filter.extractedPlanes[pidx]->normal[0] * sum_stats_[pidx].sx;
		w -= plane_filter.extractedPlanes[pidx]->normal[1] * sum_stats_[pidx].sy;
		w -= plane_filter.extractedPlanes[pidx]->normal[2] * sum_stats_[pidx].sz;
		double sum = 0;
		for (int i = 0; i < plane_vertices_[pidx].size(); ++i)
		{
			int vidx = plane_vertices_[pidx][i];
			const VertexType& v = cloud.vertices[vidx];
			double dis = w;
			for (int j = 0; j < 3; ++j)
				dis += v[j] * plane_filter.extractedPlanes[pidx]->normal[j];
			sum += dis * dis;
		}
		sum /= plane_vertices_[pidx].size();
		cout << "Distance for plane " << pidx << ": " << sum << endl;
	}
}
