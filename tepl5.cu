#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <malloc.h>
#include <time.h>

#include <cub/cub.cuh>
#include <cuda_runtime.h>

//                 i*length +j

/////////////////////////////////////
// Main ideas of algirythm are            //
// filling matrix from edges to center   //
// claculating error on each iteration  //
// finding max error and reducing it // 
/////////////////////////////////

//////////////////////////////////////
//    Initializes matrix and frame values   //
/////////////////////////////////////
void make_array(double* arr, int length, int lt, int lb, int rt, int rb) {

	//////////////////////////////////////
	//                Corner values                  //
	/////////////////////////////////////
	arr[0] = lt;
	arr[length - 1] = rt;
	arr[(length - 1)*length] = lb;
	arr[(length - 1)*length + length - 1] = rb;

	//////////////////////////////////////
	//   Calculating steps for filling frames     //
	/////////////////////////////////////
	double stepx1 = (double)(rt - lt) / (double)(length - 1);
	double stepx2 = (double)(rb - lb) / (double)(length - 1);
	double stepyl = (double)(lb - lt) / (double)(length - 1);
	double stepyr = (double)(rb - rt) / (double)(length - 1);

	//////////////////////////////////////
	//                  Filling frames                //
	/////////////////////////////////////
	for (size_t i = 1; i < length - 1; i++) arr[(length - 1)*length + i] = lb + i*stepx2;
	for (size_t i = 1; i < length - 1; i++) arr[i*length] = arr[(i - 1)*length] + stepyl;
	for (size_t i = 1; i < length - 1; i++) arr[i*length + length - 1] = arr[(i - 1)*length + length - 1] + stepyr;
	for (size_t i = 1; i < length - 1; i++) arr[i] = lt + i*stepx1;


}


//////////////////////////////////////
//      Calculating matrix elements         //
/////////////////////////////////////
__global__  void  calc_array(double* arr, double* anew, int length){

	 /////////////////////////////////////
	//        Getting element index              //
	////////////////////////////////////
	int i = blockIdx.x;
	int j = threadIdx.x;

	if(i>0 && j>0 && i<length && j<length)
		anew[i*length + j] = (arr[(i + 1)*length + j] + arr[(i - 1)*length + j] + arr[i*length + j - 1] + arr[i*length + j + 1]) / 4;
}

//////////////////////////////////////
//            Calculating errors                //
/////////////////////////////////////
__global__  void calc_error(double* arr, double* anew, double* errors, int length){

	 /////////////////////////////////////
	//        Getting element index              //
	////////////////////////////////////
	int i = blockIdx.x;
	int j = threadIdx.x;
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	 ////////////////////////////////////////
	//   Calculating error as absolute difference  //
       //   between new and old matrix values      //
       ///////////////////////////////////////
	if(i>0 && j>0 && i<length && j<length)
		errors[idx] = fabs(anew[idx] - arr[idx]);
}

///////////////////////////////////
//            Main function                //
/////////////////////////////////

int main(int argc, char** argv) {    //length iter error

	double before = clock();

	int length = atoi(argv[1]);
	int g = 0;
	double error = 1;

	 //////////////////////////////////////////
	//   Allocating memory for matricies on host  //
       /////////////////////////////////////////
	double* arr = (double*)calloc(length*length, sizeof(double));
	make_array(arr,length, 10, 20, 20, 30);

	double* anew = (double*)calloc(length*length, sizeof(double));	
	make_array(anew,length, 10, 20, 20, 30);

	double* errors = (double*)calloc(length*length, sizeof(double));

	 ///////////////////////////////
	//   Choosing device to work on   //
       //////////////////////////////
	cudaSetDevice(3);

	size_t tmp_size = 0;

	 //////////////////////////////////////////////////
	//   Creating pointers for device matricies and variables  //
       //////////////////////////////////////////////////
	double *dev_arr, *dev_anew, *dev_errs, *dev_err, *tmp = NULL;

	 ////////////////////////////////////////////////////
	//   Allocating memory for device matricies and variables  //
       ///////////////////////////////////////////////////
	cudaMalloc((void**)(&dev_arr), sizeof(double) * length * length);
	cudaMalloc((void**)(&dev_anew), sizeof(double) * length * length);
	cudaMalloc((void**)(&dev_err), sizeof(double));
	cudaMalloc((void**)(&dev_errs), sizeof(double) * length * length);

	cub::DeviceReduce::Max(tmp, tmp_size, dev_errs, dev_err, length * length);
	cudaMalloc((void**)&tmp, tmp_size);

	 ///////////////////////////////////////
	//  Copying matricies from host to device   //
       ///////////////////////////////////////
	cudaMemcpy(dev_arr, arr, sizeof(double) * length * length, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_anew, anew, sizeof(double) * length * length, cudaMemcpyHostToDevice);

	while (error > atof(argv[3]) && g < atoi(argv[2])) {

	 	///////////////////////////
		//  Calculating new values   //
       		//////////////////////////
		calc_array<<<length-1, length-1>>>(dev_arr, dev_anew, length);
		
		if (g % 100 == 0){	
	 		///////////////////////
			//  Calculating errors   //
       			//////////////////////
			calc_error<<<length-1, length-1>>>(dev_arr, dev_anew, dev_errs, length);
	 		///////////////////////////
			//  Getting the max error   //
       			/////////////////////////
			cub::DeviceReduce::Max(tmp, tmp_size, dev_errs, dev_err, length*length);
	 		///////////////////////////
			//  Updating error on host  //
       			/////////////////////////
			cudaMemcpy(&error, dev_err, sizeof(double), cudaMemcpyDeviceToHost);
		}

	 	/////////////////////////
		//  Swapping matricies   //
       		///////////////////////
		double* c = dev_arr;
		dev_arr = dev_anew;
		dev_anew = c;

		g++;
	}

	printf("Last iteration: %d Error: %.6lf\n", g, error);

	///////////////////////////////////
	//          Memory release                //
	/////////////////////////////////
	free(arr);
	free(anew);
	cudaFree(dev_arr);
	cudaFree(dev_anew);
	cudaFree(dev_errs);
	cudaFree(tmp);

	double t = clock() - before;
	t /= CLOCKS_PER_SEC;
	printf("%lf\n", t);
	return 0;
}
