
#include <iostream>
#include <algorithm>
#include <numeric>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "camFusion.hpp"
#include "dataStructures.h"

using namespace std;


// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT)
{
    // loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1)
    {
        // assemble vector for matrix-vector-multiplication
        X.at<double>(0, 0) = it1->x;
        X.at<double>(1, 0) = it1->y;
        X.at<double>(2, 0) = it1->z;
        X.at<double>(3, 0) = 1;

        // project Lidar point into camera
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        pt.x = Y.at<double>(0, 0) / Y.at<double>(0, 2); // pixel coordinates
        pt.y = Y.at<double>(1, 0) / Y.at<double>(0, 2);

        vector<vector<BoundingBox>::iterator> enclosingBoxes; // pointers to all bounding boxes which enclose the current Lidar point
        for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2)
        {
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
            smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
            smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

            // check wether point is within current bounding box
            if (smallerBox.contains(pt))
            {
                enclosingBoxes.push_back(it2);
            }

        } // eof loop over all bounding boxes

        // check wether point has been enclosed by one or by multiple boxes
        if (enclosingBoxes.size() == 1)
        { 
            // add Lidar point to bounding box
            enclosingBoxes[0]->lidarPoints.push_back(*it1);
        }

    } // eof loop over all Lidar points
}


void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait)
{
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    for(auto it1=boundingBoxes.begin(); it1!=boundingBoxes.end(); ++it1)
    {
        // create randomized color for current 3D object
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0,150), rng.uniform(0, 150), rng.uniform(0, 150));

        // plot Lidar points into top view image
        int top=1e8, left=1e8, bottom=0.0, right=0.0; 
        float xwmin=1e8, ywmin=1e8, ywmax=-1e8;
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
        {
            // world coordinates
            float xw = (*it2).x; // world position in m with x facing forward from sensor
            float yw = (*it2).y; // world position in m with y facing left from sensor
            xwmin = xwmin<xw ? xwmin : xw;
            ywmin = ywmin<yw ? ywmin : yw;
            ywmax = ywmax>yw ? ywmax : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle
            top = top<y ? top : y;
            left = left<x ? left : x;
            bottom = bottom>y ? bottom : y;
            right = right>x ? right : x;

            // draw individual point
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }

        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom),cv::Scalar(0,0,0), 2);

        // augment object with some key data
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left-250, bottom+50), cv::FONT_ITALIC, 2, currColor);
        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax-ywmin);
        putText(topviewImg, str2, cv::Point2f(left-250, bottom+125), cv::FONT_ITALIC, 2, currColor);  
    }

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // display image
    string windowName = "3D Objects";
    cv::namedWindow(windowName, 1);
    cv::imshow(windowName, topviewImg);

    if(bWait)
    {
        cv::waitKey(0); // wait for key to be pressed
    }
}


// Associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{
    // Loop over all matches in the current frame
    for (cv::DMatch match : kptMatches) {
        if (boundingBox.roi.contains(kptsCurr[match.trainIdx].pt)) {
            boundingBox.kptMatches.push_back(match);
        }
    }
}


// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, 
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
{
    // The code below is adapted from an example exercise developed earlier in this Udacity course:
    // "Camera Unit > Lesson 3: Engineering a Collision Detection System > Estimating TTC with a camera"
    
    // Compute distance ratios on every pair of keypoints, O(n^2) on the number of matches contained within the ROI
    vector<double> distRatios;
    for (auto it1 = kptMatches.begin(); it1 != kptMatches.end() - 1; ++it1) {
        cv::KeyPoint kpOuterCurr = kptsCurr.at(it1->trainIdx);  // kptsCurr is indexed by trainIdx, see NOTE in matchBoundinBoxes
        cv::KeyPoint kpOuterPrev = kptsPrev.at(it1->queryIdx);  // kptsPrev is indexed by queryIdx, see NOTE in matchBoundinBoxes

        for (auto it2 = kptMatches.begin() + 1; it2 != kptMatches.end(); ++it2) {
            cv::KeyPoint kpInnerCurr = kptsCurr.at(it2->trainIdx);  // kptsCurr is indexed by trainIdx, see NOTE in matchBoundinBoxes
            cv::KeyPoint kpInnerPrev = kptsPrev.at(it2->queryIdx);  // kptsPrev is indexed by queryIdx, see NOTE in matchBoundinBoxes

            // Use cv::norm to calculate the current and previous Euclidean distances between each keypoint in the pair
            double distCurr = cv::norm(kpOuterCurr.pt - kpInnerCurr.pt);
            double distPrev = cv::norm(kpOuterPrev.pt - kpInnerPrev.pt);

            double minDist = 100.0;  // Threshold the calculated distRatios by requiring a minimum current distance between keypoints 

            // Avoid division by zero and apply the threshold
            if (distPrev > std::numeric_limits<double>::epsilon() && distCurr >= minDist) {
                double distRatio = distCurr / distPrev;
                distRatios.push_back(distRatio);
            }
        }
    }

    // Only continue if the vector of distRatios is not empty
    if (distRatios.size() == 0)
    {
        TTC = std::numeric_limits<double>::quiet_NaN();
        return;
    }

    // As with computeTTCLidar, use the median as a reasonable method of excluding outliers
    std::sort(distRatios.begin(), distRatios.end());
    double medianDistRatio = distRatios[distRatios.size() / 2];

    // Finally, calculate a TTC estimate based on these 2D camera features
    TTC = (-1.0 / frameRate) / (1 - medianDistRatio);
}

double medianOfX(std::vector<LidarPoint> &lidarPoints, double index)
{
    int floorIndex = (int)index;
    if(floorIndex == index)
    {
        //whole
        return lidarPoints[floorIndex].x;

    }
    else
    {
        return (lidarPoints[floorIndex].x + lidarPoints[floorIndex+1].x)/2;
    }
    
}

double getInterquartileMedianOfX(std::vector<LidarPoint> &lidarPoints)
{
    // This std::sort with a lambda mutates lidarPoints, a vector of LidarPoint
    std::sort(lidarPoints.begin(), lidarPoints.end(), [](LidarPoint a, LidarPoint b) {
        return a.x < b.x;  // Sort ascending on the x coordinate only
    });

    int n = lidarPoints.size();

    //first quartile
    double q1_index = (n+1)/4;

    double Q1 = medianOfX(lidarPoints, q1_index);

    double q3_index = 3*(n+1)/4;

    double Q3 = medianOfX(lidarPoints, q3_index);

    return Q3 - Q1;
}

// Compute time-to-collision (TTC) based on relevant lidar points
void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{
    // In each frame, take the median x-distance as our more robust estimate.
    // If performance is suffering, consider taking the median of a random subset of the points.
    
    double d0 = getInterquartileMedianOfX(lidarPointsPrev);
    double d1 = getInterquartileMedianOfX(lidarPointsCurr);

    // Using the constant-velocity model (as opposed to a constant-acceleration model)
    // TTC = d1 * delta_t / (d0 - d1)
    // where: d0 is the previous frame's closing distance (front-to-rear bumper)
    //        d1 is the current frame's closing distance (front-to-rear bumper)
    //        delta_t is the time elapsed between images (1 / frameRate)
    // Note: this function does not take into account the distance from the lidar origin to the front bumper of our vehicle.
    // It also does not account for the curvature or protrusions from the rear bumper of the preceding vehicle.
    TTC = d1 * (1.0 / frameRate) / (d0 - d1);
}


void matchBoundingBoxes( std::vector<cv::DMatch> &matches,std::map<int, int> &bbBestMatches,DataFrame &prevFrame,DataFrame &currFrame )
{
    cv::Mat count( prevFrame.boundingBoxes.size(),currFrame.boundingBoxes.size(),CV_16UC1,cv::Scalar( 0 ) );
    for( auto itmatch = matches.begin();itmatch != matches.end();++itmatch ){
        for( auto itprev = prevFrame.boundingBoxes.begin();itprev != prevFrame.boundingBoxes.end();++itprev ){
            if( (*itprev).roi.contains( prevFrame.keypoints[(*itmatch).queryIdx].pt ) ){ // find bb in prev frame
                for( auto itcurr = currFrame.boundingBoxes.begin();itcurr != currFrame.boundingBoxes.end();++itcurr ){
                    if( (*itcurr).roi.contains( currFrame.keypoints[(*itmatch).trainIdx].pt ) ){ // find the correspoding bb in curr frame
                        count.at<u_int16_t>( (*itprev).boxID,(*itcurr).boxID ) += 1; 
                    }
                }       
            }
        }
    }
    int max = 0;
    int ind = 0;
    for( int prev = 0;prev<count.rows;prev++ ){
        max = 0;
        ind = 0;
        for( int curr = 0;curr<count.cols;curr++ ){
            if( count.at<u_int16_t>( prev,curr ) > max ){
                max = count.at<u_int16_t>( prev,curr );
                ind = curr;
            }
        }
        if( max > 0 ) bbBestMatches.insert( std::pair<int,int>( prev,ind ) ); 
    }
}
