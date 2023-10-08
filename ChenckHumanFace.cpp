#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
using namespace cv;


// 储本地图片的人脸特征
CvMat* local_face = NULL;

// 在图像上画出矩形框
void draw_rect(IplImage* img, CvRect* rect)
{
    cv::Scalar color = cv::Scalar(255, 0, 0);
    cvRectangle(img, cvPoint(rect->x, rect->y), cvPoint(rect->x + rect->width, rect->y + rect->height), color, 2, 8, 0);
}

// 计算两个向量之间的欧氏距离
double euclidean_distance(CvMat* vec1, CvMat* vec2)
{
    double sum = 0.0;
    for (int i = 0; i < vec1->rows; i++)
    {
        double diff = cvmGet(vec1, i, 0) - cvmGet(vec2, i, 0);
        sum += diff * diff;
    }
    return sqrt(sum);
}

// 提取图像中的人脸特征
CvMat* extract_face_feature(IplImage* img, CvHaarClassifierCascade* cascade)
{
    // 将图像转换为灰度图
    IplImage* gray = cvCreateImage(cvGetSize(img), IPL_DEPTH_8U, 1);
    cvCvtColor(img, gray, CV_BGR2GRAY);

    // 检测图像中的人脸
    CvMemStorage* storage = cvCreateMemStorage(0);
    CvSeq* faces = cvHaarDetectObjects(gray, cascade, storage, 1.1, 3, CV_HAAR_DO_CANNY_PRUNING, cvSize(50, 50), cvSize(0, 0));

    // 如果没有检测到人脸，返回NULL
    if (faces->total == 0)
    {
        cvReleaseImage(&gray);
        cvReleaseMemStorage(&storage);
        return NULL;
    }

    // 如果检测到多个人脸，只取第一个
    CvRect* rect = (CvRect*)cvGetSeqElem(faces, 0);

    // 在图像上画出矩形框，并输出人脸的坐标
    draw_rect(img, rect);
    printf("Face detected at (%d, %d), width = %d, height = %d\n", rect->x, rect->y, rect->width, rect->height);

    // 截取人脸区域，并将其缩放为100x100的大小
    IplImage* face = cvCreateImage(cvSize(100, 100), IPL_DEPTH_8U, 1);
    cvSetImageROI(gray, *rect);
    cvResize(gray, face, CV_INTER_LINEAR);
    cvResetImageROI(gray);

    // 将人脸图像转换为向量，并归一化
    CvMat* vec = cvCreateMat(10000, 1, CV_32FC1);
    for (int i = 0; i < face->height; i++)
    {
        for (int j = 0; j < face->width; j++)
        {
            int index = i * face->width + j;
            float value = (float)cvGetReal2D(face, i, j);
            cvmSet(vec, index, 0, value);
        }
    }
    cvNormalize(vec, vec, 1, 0, CV_L2, NULL);

    // 释放内存
    cvReleaseImage(&gray);
    cvReleaseImage(&face);
    cvReleaseMemStorage(&storage);

    // 返回人脸特征向量
    return vec;
}

// 比较两个人脸特征向量是否相似
bool is_similar(CvMat* vec1, CvMat* vec2)
{
    // 计算两个向量之间的欧氏距离
    double dist = euclidean_distance(vec1, vec2);

    // 如果距离小于某个阈值，认为是相似的
    if (dist < 10.0)
    {
        return true;
    }
    else
    {
        return false;
    }
}

int main()
{
    // 加载OpenCV提供的人脸检测器
    CvHaarClassifierCascade* cascade;
    // 此处default的xml要取cuda文件夹内的default
    cascade = (CvHaarClassifierCascade*)cvLoad("D:\\ComputerProgram\\Desktop\\haarcascade_frontalface_default.xml", 0, 0, 0);

    // 读取本地图片，并提取人脸特征
    IplImage* local_img = cvLoadImage("D:\\ComputerProgram\\Desktop\\target.jpg", CV_LOAD_IMAGE_COLOR);
    local_face = extract_face_feature(local_img, cascade);

    // 如果没有提取到人脸特征，退出程序
    if (local_face == NULL)
    {
        printf("No face detected in local image\n");
        return -1;
    }

    // 创建一个窗口，显示本地图片
    cvNamedWindow("Local Image", CV_WINDOW_AUTOSIZE);
    cvShowImage("Local Image", local_img);

    // 创建一个摄像头对象，捕获图像
    CvCapture* capture = cvCreateCameraCapture(0);

    // 如果摄像头打开失败，退出程序
    if (capture == NULL)
    {
        printf("Failed to open camera\n");
        return -1;
    }

    // 循环捕获图像，直到按下ESC键
    while (true)
    {
        // 获取一帧图像，并判断是否为空
        IplImage* frame = cvQueryFrame(capture);
        if (frame == NULL)
        {
            printf("No frame captured\n");
            break;
        }

        // 提取图像中的人脸特征
        CvMat* frame_face = extract_face_feature(frame, cascade);

        // 如果提取到人脸特征，就和本地图片的人脸特征进行比较
        if (frame_face != NULL)
        {
            bool result = is_similar(local_face, frame_face);

            // 如果比较结果为真，则打印字符，表示识别成功
            if (result == true)
            {
                printf("success");
                break;
            }
        }

        // 创建一个窗口，显示摄像头捕获的图像
        cvNamedWindow("Camera", CV_WINDOW_AUTOSIZE);
        cvShowImage("Camera", frame);

        // 等待按键输入，如果是ESC键，就退出循环
        int key = cvWaitKey(10);
        if (key == 27)
        {
            break;
        }
    }

    // 释放内存和资源
    cvReleaseMat(&local_face);
    cvReleaseImage(&local_img);
    cvReleaseCapture(&capture);
    cvDestroyAllWindows();

    return 0;
}
