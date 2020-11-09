#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <string>
#include <cuda.h>
#include <stdio.h>
#include <opencv\cv.h>
#include <opencv\highgui.h>
#include <iostream>



#define CUDA_ERROR_CHECK

#define CudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )
#define CudaCheckError()    __cudaCheckError( __FILE__, __LINE__ )

inline void __cudaSafeCall(cudaError err, const char *file, const int line)
{
#ifdef CUDA_ERROR_CHECK
	if (cudaSuccess != err)
	{
		fprintf(stderr, "cudaSafeCall() failed at %s:%i : %s\n",
			file, line, cudaGetErrorString(err));
		exit(-1);
	}
#endif

	return;
}

inline void __cudaCheckError(const char *file, const int line)
{
#ifdef CUDA_ERROR_CHECK
	cudaError err = cudaGetLastError();
	if (cudaSuccess != err)
	{
		fprintf(stderr, "cudaCheckError() failed at %s:%i : %s\n",
			file, line, cudaGetErrorString(err));
		exit(-1);
	}


	err = cudaDeviceSynchronize();
	if (cudaSuccess != err)
	{
		fprintf(stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
			file, line, cudaGetErrorString(err));
		exit(-1);
	}
#endif

	return;
}

__global__ void rgb_2_grey(uchar* const greyImage, const uchar4* const rgbImage, int rows, int columns)
{
	int rgb_x = blockIdx.x * blockDim.x + threadIdx.x; //x coordinate of pixel
	int rgb_y = blockIdx.y * blockDim.y + threadIdx.y; //y coordinate of pixel
	
	//stops function here if condition is met
	if ((rgb_x >= columns) || (rgb_y >= rows)) 
	{
		return;
	}

	int rgb_ab = rgb_y*columns + rgb_x; //absolute pixel position
	uchar4 rgb_Img = rgbImage[rgb_ab];
	greyImage[rgb_ab] = uchar((float(rgb_Img.x))*0.299f + (float(rgb_Img.y))*0.587f + (float(rgb_Img.z))*0.114f);
}

using namespace cv;
using namespace std;

void Load_img(string& filename);
void Proc_Img(uchar4 **d_RGBImage, uchar** d_greyImage);
void RGB_2_Greyscale(uchar* const d_greyImage, uchar4* const d_RGBImage, size_t num_Rows, size_t num_Cols);
void Save_Img(string& filename);

Mat img_RGB;
Mat img_Grey;
uchar4 *d_rgbImg;
uchar *d_greyImg; 

int main()
{
		string input_img = "C:\\Users\\Austin\\Pictures\\wallpapers\\IMG_3575.JPG"; //input file path
		string output_img = "C:\\Users\\Austin\\Pictures\\wallpapers\\IMG_3575GR2.JPG";//out put file path

		Load_img(input_img);//loads input image and creates a Mat object, then converts colors from blue, green, red(BGR) format to standard red, green, blue(RGB) format, 
		                    //finally creates an array(allocates memory) for grey image
		Proc_Img(&d_rgbImg, &d_greyImg);//allocates memory on gpu and copies data to gpu
		RGB_2_Greyscale(d_greyImg, d_rgbImg, img_RGB.rows, img_RGB.cols);//calls kernel which turns image to grayscale 
		Save_Img(output_img);//writes final image to drive





    return 0;
}

void Load_img(string& filename)
{

	//loads image into a matrix object along with the colors in BGR format (must convert to rgb).
	Mat img = imread(filename.c_str(), CV_LOAD_IMAGE_COLOR); 
	
	//check if image loaded correctly
	if (img.empty())
	{
		cerr << "File located at " << filename << " not read " << endl;
		exit(1);
	}

	//converts color type from BGR to RGB
	cvtColor(img, img_RGB, CV_BGR2RGBA);

	//allocate memory for new greyscale image.
	img_Grey.create(img.rows, img.cols, CV_8UC1); //img.rows returns the range of pixels in y, img.cols returns range of pixels in x
	                                              //CV_8UC1 means 8 bit unsigned(non-negative) single channel of color, aka greyscale.
	                                              //all three of the parameters allow the create function in the Mat class to determine how much memory to allocate  
												  
}

void Proc_Img(uchar4 **d_RGBImage, uchar** d_greyImage)
{
	
	cudaFree(0);
	CudaCheckError();
	
	//creates rgb and greyscale image arrays
	uchar4 *h_RGBImage = (uchar4*)img_RGB.ptr<uchar>(0); //.ptr is a method in the mat class that returns a pointer to the first element of the matrix.
                                                         //this is just like a regular array/pointer mem address to first element of the array. This is templated
														 //in this case the compiler runs the function for returning pointer of type unsigned char. for rgb image it is
														 //cast to uchar4 struct to hold r,g,b, and alpha(ignored in program) values.

	const size_t num_pix = (img_RGB.rows) * (img_RGB.cols); //amount of pixels 

	//allocate memory on gpu
	cudaMalloc(d_RGBImage, sizeof(uchar4) * num_pix); //bites of 1 uchar4 times # of pixels gives number of bites necessary for array
	CudaCheckError();
	cudaMalloc(d_greyImage, sizeof(uchar) * num_pix); //bites of uchar times # pixels gives number of bites necessary for array
	CudaCheckError();
	cudaMemset(*d_greyImage, 0, sizeof(uchar) * num_pix); //makes sure all data in allocated space is set to 0
	CudaCheckError();


	//copy array into allocated space
	cudaMemcpy(*d_RGBImage, h_RGBImage, sizeof(uchar4)*num_pix, cudaMemcpyHostToDevice);
	CudaCheckError();


	d_rgbImg = *d_RGBImage;
	d_greyImg = *d_greyImage; 
}


void RGB_2_Greyscale(uchar* const d_greyImage, uchar4* const d_RGBImage, size_t num_Rows, size_t num_Cols)
{

	const int BS = 32;
	const dim3 blockSize(BS, BS);
	const dim3 gridSize((num_Cols / BS) + 1, (num_Rows / BS) + 1); 

	rgb_2_grey <<<gridSize, blockSize>>>(d_greyImage, d_RGBImage, num_Rows, num_Cols);

	cudaDeviceSynchronize(); CudaCheckError();


}



void Save_Img(string& filename)
{

	const size_t num_pix = (img_RGB.rows) * (img_RGB.cols); //number of pixels
	cudaMemcpy(img_Grey.ptr<uchar>(0), d_greyImg, sizeof(uchar)*num_pix, cudaMemcpyDeviceToHost); //copy array from gpu to cpu
	CudaCheckError();


	imwrite(filename.c_str(), img_Grey); //save image to drive

	cudaFree(d_rgbImg); //deallocate memory on gpu
	cudaFree(d_greyImg);//deallocate memory on gpu

}


