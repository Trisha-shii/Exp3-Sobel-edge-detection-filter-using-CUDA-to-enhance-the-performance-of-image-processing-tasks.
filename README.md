# Exp3-Sobel-edge-detection-filter-using-CUDA-to-enhance-the-performance-of-image-processing-tasks.
<h3>AIM:How can the implementation of a Sobel edge detection filter using CUDA enhance the performance of image processing tasks compared to a traditional CPU-based approach, and what are the potential challenges and optimizations associated with this parallelization? </h3>
<h3>NAME: TRISHA PRIYADARSHNI PARIDA</h3>
<h3>REGISTER NO : 212224230293 </h3>
<h3>EX. NO</h3>
<h3>DATE : 26-05-2026/h3>
<h1> <align=center> Sobel edge detection filter using CUDA </h3>
  Implement Sobel edge detection filtern using GPU.</h3>
Experiment Details:
  
## AIM:
  The Sobel operator is a popular edge detection method that computes the gradient of the image intensity at each pixel. It uses convolution with two kernels to determine the gradient in both the x and y directions. This lab focuses on utilizing CUDA to parallelize the Sobel filter implementation for efficient processing of images.

Code Overview: You will work with the provided CUDA implementation of the Sobel edge detection filter. The code reads an input image, applies the Sobel filter in parallel on the GPU, and writes the result to an output image.
## EQUIPMENTS REQUIRED:
Hardware – PCs with NVIDIA GPU & CUDA NVCC
Google Colab with NVCC Compiler
CUDA Toolkit and OpenCV installed.
A sample image for testing.

## PROCEDURE:
Tasks: 
a. Modify the Kernel:

Update the kernel to handle color images by converting them to grayscale before applying the Sobel filter.
Implement boundary checks to avoid reading out of bounds for pixels on the image edges.

b. Performance Analysis:

Measure the performance (execution time) of the Sobel filter with different image sizes (e.g., 256x256, 512x512, 1024x1024).
Analyze how the block size (e.g., 8x8, 16x16, 32x32) affects the execution time and output quality.

c. Comparison:

Compare the output of your CUDA Sobel filter with a CPU-based Sobel filter implemented using OpenCV.
Discuss the differences in execution time and output quality.

## PROGRAM:
```python
!apt-get update -qq
!apt-get install -y libopencv-dev

%%writefile sobelEdgeDetectionFilter.cu

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <opencv2/opencv.hpp>

using namespace cv;

__global__ void sobelFilter(unsigned char *srcImage,
                            unsigned char *dstImage,
                            unsigned int width,
                            unsigned int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x > 0 && x < width - 1 &&
        y > 0 && y < height - 1)
    {
        int Gx =
            -srcImage[(y-1)*width + (x-1)]
            -2*srcImage[y*width + (x-1)]
            -srcImage[(y+1)*width + (x-1)]
            +srcImage[(y-1)*width + (x+1)]
            +2*srcImage[y*width + (x+1)]
            +srcImage[(y+1)*width + (x+1)];

        int Gy =
            -srcImage[(y-1)*width + (x-1)]
            -2*srcImage[(y-1)*width + x]
            -srcImage[(y-1)*width + (x+1)]
            +srcImage[(y+1)*width + (x-1)]
            +2*srcImage[(y+1)*width + x]
            +srcImage[(y+1)*width + (x+1)];

        int magnitude = sqrtf((float)(Gx * Gx + Gy * Gy));

        if (magnitude > 255)
            magnitude = 255;

        dstImage[y * width + x] = (unsigned char)magnitude;
    }
}

void checkCudaErrors(cudaError_t r)
{
    if (r != cudaSuccess)
    {
        fprintf(stderr,
                "CUDA Error: %s\n",
                cudaGetErrorString(r));
        exit(EXIT_FAILURE);
    }
}

int main()
{
    Mat image = imread("/content/img.jpg",
                       IMREAD_GRAYSCALE);

    if (image.empty())
    {
        printf("Error: Image not found.\n");
        return -1;
    }

    int width = image.cols;
    int height = image.rows;

    size_t imageSize =
        width * height * sizeof(unsigned char);

    unsigned char *h_outputImage =
        (unsigned char*)malloc(imageSize);

    unsigned char *d_inputImage;
    unsigned char *d_outputImage;

    checkCudaErrors(
        cudaMalloc(&d_inputImage, imageSize));

    checkCudaErrors(
        cudaMalloc(&d_outputImage, imageSize));

    checkCudaErrors(
        cudaMemcpy(d_inputImage,
                   image.data,
                   imageSize,
                   cudaMemcpyHostToDevice));

    cudaMemset(d_outputImage, 0, imageSize);

    dim3 blockSize(16,16);

    dim3 gridSize(
        (width + blockSize.x - 1)/blockSize.x,
        (height + blockSize.y - 1)/blockSize.y);

    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    sobelFilter<<<gridSize, blockSize>>>(
        d_inputImage,
        d_outputImage,
        width,
        height);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;

    cudaEventElapsedTime(
        &milliseconds,
        start,
        stop);

    checkCudaErrors(
        cudaMemcpy(h_outputImage,
                   d_outputImage,
                   imageSize,
                   cudaMemcpyDeviceToHost));

    Mat outputImage(
        height,
        width,
        CV_8UC1,
        h_outputImage);

    imwrite("/content/output_sobel.jpeg",
            outputImage);

    printf("Total time taken: %f ms\n",
           milliseconds);

    free(h_outputImage);

    cudaFree(d_inputImage);
    cudaFree(d_outputImage);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}

!nvcc sobelEdgeDetectionFilter.cu -o sobel `pkg-config --cflags --libs opencv4`

!./sobel

!ls /content


import cv2
import matplotlib.pyplot as plt

img = cv2.imread(
    "/content/output_sobel.jpeg",
    cv2.IMREAD_GRAYSCALE
)

plt.figure(figsize=(10,6))
plt.imshow(img, cmap='gray')
plt.axis('off')
plt.title("Sobel Edge Detection")
plt.show()
```

## OUTPUT:
<img width="969" height="586" alt="image" src="https://github.com/user-attachments/assets/2ed99c51-3b3a-437b-bca5-9e84d2566c17" />


## RESULT:

Thus the program has been executed by using CUDA to accelerate Sobel edge detection and improve image processing performance using parallel computation on GPU.


## Questions:

What challenges did you face while implementing the Sobel filter for color images?
How did changing the block size influence the performance of your CUDA implementation?
What were the differences in output between the CUDA and CPU implementations? Discuss any discrepancies.
Suggest potential optimizations for improving the performance of the Sobel filter.

Deliverables:

Modified CUDA code with comments explaining your changes.
A report summarizing your findings, including graphs of execution times and a comparison of outputs.
Answers to the questions posed in the experiment.
Tools Required:

