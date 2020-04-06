# SFND 3D Object Tracking

Welcome to the final project of the camera course. By completing all the lessons, you now have a solid understanding of keypoint detectors, descriptors, and methods to match them between successive images. Also, you know how to detect objects in an image using the YOLO deep-learning framework. And finally, you know how to associate regions in a camera image with Lidar points in 3D space. Let's take a look at our program schematic to see what we already have accomplished and what's still missing.

<img src="images/course_code_structure.png" width="779" height="414" />

In this final project, you will implement the missing parts in the schematic. To do this, you will complete four major tasks: 
1. First, you will develop a way to match 3D objects over time by using keypoint correspondences. 
2. Second, you will compute the TTC based on Lidar measurements. 
3. You will then proceed to do the same using the camera, which requires to first associate keypoint matches to regions of interest and then to compute the TTC based on those matches. 
4. And lastly, you will conduct various tests with the framework. Your goal is to identify the most suitable detector/descriptor combination for TTC estimation and also to search for problems that can lead to faulty measurements by the camera or Lidar sensor. In the last course of this Nanodegree, you will learn about the Kalman filter, which is a great way to combine the two independent TTC measurements into an improved version which is much more reliable than a single sensor alone can be. But before we think about such things, let us focus on your final project in the camera course. 

## Dependencies for Running Locally
* cmake >= 2.8
  * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1 (Linux, Mac), 3.81 (Windows)
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* Git LFS
  * Weight files are handled using [LFS](https://git-lfs.github.com/)
* OpenCV >= 4.1
  * This must be compiled from source using the `-D OPENCV_ENABLE_NONFREE=ON` cmake flag for testing the SIFT and SURF detectors.
  * The OpenCV 4.1.0 source code can be found [here](https://github.com/opencv/opencv/tree/4.1.0)
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools](https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)

## Basic Build Instructions

1. Clone this repo.
2. Make a build directory in the top level project directory: `mkdir build && cd build`
3. Compile: `cmake .. && make`
4. Run it: `./3D_object_tracking`.



---
# Wright up
## [Rubric](https://review.udacity.com/#!/rubrics/2550/view) Points

![Final TTC estimation](results/TTC_estimation.png)

#### 1. Match 3D Objects

Implement the method "matchBoundingBoxes", which takes as input both the previous and the current data frames and provides as output the ids of the matched regions of interest (i.e. the boxID property). Matches must be the ones with the highest number of keypoint correspondences.

```c++
void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
{
    // NOTE: After calling a cv::DescriptorMatcher::match function, each DMatch
    // contains two keypoint indices, queryIdx and trainIdx, based on the order of image arguments to match.
    // https://docs.opencv.org/4.1.0/db/d39/classcv_1_1DescriptorMatcher.html#a0f046f47b68ec7074391e1e85c750cba
    // prevFrame.keypoints is indexed by queryIdx
    // currFrame.keypoints is indexed by trainIdx

    std::multimap<int, int> mmap {};
    int maxPrevBoxID = 0;

    for (auto match : matches) {
        cv::KeyPoint prevKp = prevFrame.keypoints[match.queryIdx];
        cv::KeyPoint currKp = currFrame.keypoints[match.trainIdx];
        
        int prevBoxID = -1;
        int currBoxID = -1;

        // For each bounding box in the previous frame
        for (auto bbox : prevFrame.boundingBoxes) {
            if (bbox.roi.contains(prevKp.pt)) prevBoxID = bbox.boxID;
        }

        // For each bounding box in the current frame
        for (auto bbox : currFrame.boundingBoxes) {
            if (bbox.roi.contains(currKp.pt)) currBoxID = bbox.boxID;
        }
        
        // Add the containing boxID for each match to a multimap
        mmap.insert({currBoxID, prevBoxID});

        maxPrevBoxID = std::max(maxPrevBoxID, prevBoxID);
    }

    // Setup a list of boxID int values to iterate over in the current frame
    vector<int> currFrameBoxIDs {};
    for (auto box : currFrame.boundingBoxes) currFrameBoxIDs.push_back(box.boxID);

    // Loop through each boxID in the current frame, and get the mode (most frequent value) of associated boxID for the previous frame.
    for (int k : currFrameBoxIDs) {
        // Count the greatest number of matches in the multimap, where each element is {key=currBoxID, val=prevBoxID}
        // std::multimap::equal_range(k) returns the range of all elements matching key = k.
        auto rangePrevBoxIDs = mmap.equal_range(k);

        // Create a vector of counts (per current bbox) of prevBoxIDs
        std::vector<int> counts(maxPrevBoxID + 1, 0);

        // Accumulator loop
        for (auto it = rangePrevBoxIDs.first; it != rangePrevBoxIDs.second; ++it) {
            if (-1 != (*it).second) counts[(*it).second] += 1;
        }

        // Get the index of the maximum count (the mode) of the previous frame's boxID
        int modeIndex = std::distance(counts.begin(), std::max_element(counts.begin(), counts.end()));

        // Set the best matching bounding box map with
        // key   = Previous frame's most likely matching boxID
        // value = Current frame's boxID, k
        bbBestMatches.insert({modeIndex, k});
    }
}
```

#### 2. Compute Lidar-based TTC

Compute the time-to-collision in second for all matched 3D objects using only Lidar measurements from the matched bounding boxes between current and previous frame.

In order to deal with outlier Lidar points in a statistically robust way to avoid severe estimation errors, here only consider Lidar points within ego lane, then get the mean distance to get stable output.

```c++
// Compute time-to-collision (TTC) based on relevant lidar points
void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{
    // In each frame, take the median x-distance as our more robust estimate.
    // If performance is suffering, consider taking the median of a random subset of the points.
    sortLidarPointsX(lidarPointsPrev);
    sortLidarPointsX(lidarPointsCurr);
    double d0 = lidarPointsPrev[lidarPointsPrev.size()/2].x;
    double d1 = lidarPointsCurr[lidarPointsCurr.size()/2].x;

    // Using the constant-velocity model (as opposed to a constant-acceleration model)
    // TTC = d1 * delta_t / (d0 - d1)
    // where: d0 is the previous frame's closing distance (front-to-rear bumper)
    //        d1 is the current frame's closing distance (front-to-rear bumper)
    //        delta_t is the time elapsed between images (1 / frameRate)
    // Note: this function does not take into account the distance from the lidar origin to the front bumper of our vehicle.
    // It also does not account for the curvature or protrusions from the rear bumper of the preceding vehicle.
    TTC = d1 * (1.0 / frameRate) / (d0 - d1);
}
```

#### 3. Associate Keypoint Correspondences with Bounding Boxes

Prepare the TTC computation based on camera measurements by associating keypoint correspondences to the bounding boxes which enclose them. All matches which satisfy this condition must be added to a vector in the respective bounding box.

```c++
// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{
    // Loop over all matches in the current frame
    for (cv::DMatch match : kptMatches) {
        if (boundingBox.roi.contains(kptsCurr[match.trainIdx].pt)) {
            boundingBox.kptMatches.push_back(match);
        }
    }
}
```

#### 4. Compute Camera-based TTC

Compute the time-to-collision in second for all matched 3D objects using only keypoint correspondences from the matched bounding boxes between current and previous frame.

```c++
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
```

#### 5.  Performance Evaluation 1

Find examples where the TTC estimate of the Lidar sensor does not seem plausible. Describe your observations and provide a sound argumentation why you think this happened.

TTC from Lidar is not correct because of some outliers and some unstable points from preceding vehicle's  front mirrors, those need to be filtered out . Here we adapt a bigger shrinkFactor = 0.2, to get more reliable and stable lidar points. Then get a  more accurate results.


#### 6. Performance Evaluation 2

Run several detector / descriptor combinations and look at the differences in TTC estimation. Find out which methods perform best and also include several examples where camera-based TTC estimation is way off. As with Lidar, describe your observations again and also look into potential reasons.

when get a robust  clusterKptMatchesWithROI can get a stable TTC from Camera. if the result get unstable, It's probably the worse keypints matches.

The TOP3 detector / descriptor combinations as the best choice for our purpose of detecting keypoints on vehicles are:
SHITOMASI/BRISK         
SHITOMASI/BRIEF            
SHITOMASI/ORB           