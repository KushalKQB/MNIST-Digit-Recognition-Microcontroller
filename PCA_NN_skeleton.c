/*******************************************************************************
 * File			: Skeleton Program
 * Author		: Kushal Kumar Kasina
 * Description	: The code given, is a general structure and gives a good idea
 *				  on how the functions are supposed to be used. This is not a
 *				  plug-n-play file nor are there any provided. Please consider
 *				  this as more of a proof of concept and not a well-polished lib
 *				  style code.
 *				  Demonstrated here is the usage of the functions to apply PCA
 *				  to a sample 28x28 raw MNIST digit Image, then obtain a feature
 *				  of size 28x1, then input this to a 1-layer Neural Network. For
 *				  a working example, see repo.
 * Requirements : CMSIS_DSP is required for matrix ops.
 ******************************************************************************/

/* Defines */
/*...*/
#define ARM_MATH_CM4
 
/* Includes */
/*...*/
#include "arm_math.h"

/* Function Prototypes */
/*...*/
/* Computes Activations from hidden layer 1 of NN */
void layer0_forwardPass(arm_matrix_instance_f32* pcaVec,
						arm_matrix_instance_f32* layer0_weights,
						arm_matrix_instance_f32* layer0_biases,
						arm_matrix_instance_f32* nnMat);

/* Computes Activations from output layer of NN*/
void outputLayer_forwardPass(arm_matrix_instance_f32* nnMat,
						arm_matrix_instance_f32* outputLayer_weights,
						arm_matrix_instance_f32* outputLayer_biases,
						arm_matrix_instance_f32* predictionMat);

/* Computes L2 Norm of a Vector */
static float32_t computeL2Norm(arm_matrix_instance_f32* mat);

/* Applied PCA and generates most dominant feature from the input matrix */
void computePCA(arm_matrix_instance_f32* pcaVec, arm_matrix_instance_f32* mat);

/* Computes column-wise mean of a matrix */
void columnMean(arm_matrix_instance_f32* colMean,
					   arm_matrix_instance_f32* mat);

/* Adds Bias as in Z = W.X + b */
void addBias(arm_matrix_instance_f32* mat,
                    arm_matrix_instance_f32* addMat);

/* ReLU Function */
void relu(arm_matrix_instance_f32* mat);

/* Softmax Function */
void softmax(arm_matrix_instance_f32* mat);

/* main */
int main(void)
{
	/*...*/
	
	/* Input Image from MNIST (It's a digit 9) */
	float32_t imgMat_arr[784] = { /* 28x28 Image data */ };

	/* Weights and Biases of Hidden Layer 1 and Output Layer */
	float32_t layer0_weights_arr[784] = { /* 28x28 weights of Layer-0 */ };
	float32_t layer1_weights_arr[280] = { /* 10x28 weights of output Layer */ };
	float32_t layer0_biases_arr[28] = { /* 28x1 biases of Layer-0 */ };
	float32_t layer1_biases_arr[10] = { /* 10x1 biases of output Layer */};

	/* Empty arrays */
	float32_t pcaVec_arr[28] = {0};
	float32_t nnMat_arr[28] = {0};
	float32_t predictions[10] = {0};

	/* Matrix Instances using CMSIS_DSP */
	arm_matrix_instance_f32 imgMat = {28, 28, imgMat_arr};
	arm_matrix_instance_f32 pcaVec = {28, 1, pcaVec_arr};
	arm_matrix_instance_f32 nnMat = {28, 1, nnMat_arr};
	arm_matrix_instance_f32 predictionMat = {10, 1, predictions};

	/* Neural Network Weights and Biases
	 * Total Hidden Layers: 1
	 * Number of Neurons in the Hidden Layer: 28
	 * Output Number of Classes: 10
	 */
	arm_matrix_instance_f32 layer0_weights = {28, 28, layer0_weights_arr};
	arm_matrix_instance_f32 layer0_biases  = {28, 1 , layer0_biases_arr};
	arm_matrix_instance_f32 layer1_weights = {10, 28, layer1_weights_arr};
	arm_matrix_instance_f32 layer1_biases  = {10, 1 , layer1_biases_arr};

	/* Compute PCA Feature on input image */
	computePCA(&pcaVec, &imgMat);

	/* Layer 0 forward pass */
	layer0_forwardPass(&pcaVec, &layer0_weights, &layer0_biases, &nnMat);

	/* Output Layer */
	outputLayer_forwardPass(&nnMat, &layer1_weights, &layer1_biases,
							&predictionMat);

	/*...*/
	
	while(1);
}

/*...*/

/* Supporting Functions */

/*------------------------------------------------------------------------------
 * layer0_forwardPass
 * Description: Computes Activations from hidden layer 1 of NN
 * Parameters : pcaVec 				(type=arm_matrix_instance_f32, size=(28,1)
 * 				layer0_weights		(type=arm_matrix_instance_f32, size=(28,28)
 * 				layer0_biases 		(type=arm_matrix_instance_f32, size=(28,1)
 * 				nnMat		 		(type=arm_matrix_instance_f32, size=(28,1)
 *----------------------------------------------------------------------------*/
void layer0_forwardPass(arm_matrix_instance_f32* pcaVec,
						arm_matrix_instance_f32* layer0_weights,
						arm_matrix_instance_f32* layer0_biases,
						arm_matrix_instance_f32* nnMat)
{
	/* Matrix Multiply Weights and Input Activations */
	arm_mat_mult_f32(layer0_weights, pcaVec, nnMat);

	/* Adds Biases to the output from the previous step */
	addBias(nnMat, layer0_biases);

	/* Apply ReLU to obtain activations for the next layer */
	relu(nnMat);
}

/*------------------------------------------------------------------------------
 * layer0_forwardPass
 * Description: Computes Activations from output layer of NN
 * Parameters : nnMat 				(type=arm_matrix_instance_f32, size=(28,1)
 * 				outputLayer_weights	(type=arm_matrix_instance_f32, size=(28,28)
 * 				outputLayer_biases	(type=arm_matrix_instance_f32, size=(28,1)
 * 				predictionMat		(type=arm_matrix_instance_f32, size=(28,1)
 *----------------------------------------------------------------------------*/
void outputLayer_forwardPass(arm_matrix_instance_f32* nnMat,
						arm_matrix_instance_f32* outputLayer_weights,
						arm_matrix_instance_f32* outputLayer_biases,
						arm_matrix_instance_f32* predictionMat)
{
	/* Matrix Multiply Weights and Input Activations */
	arm_mat_mult_f32(outputLayer_weights, nnMat, predictionMat);

	/* Adds Biases to the output from the previous step */
	addBias(predictionMat, outputLayer_biases);

	/* Apply ReLU to obtain activations for the next layer */
	softmax(predictionMat);
}

/*------------------------------------------------------------------------------
 * computePCA
 * Description: Computes the most dominant PCA feature of the input matrix
 * Parameters : pcaVec   (type=arm_matrix_instance_f32, size=(28,1)
 * 				mat		 (type=arm_matrix_instance_f32, size=(28,28)
 *----------------------------------------------------------------------------*/
void computePCA(arm_matrix_instance_f32* pcaVec,
		               arm_matrix_instance_f32* mat)
{
	uint8_t row = 0, col = 0;			// variables for matrix ops
	uint8_t i = 0, j = 0;				// variables for loops
	float32_t tempFloat32 = 0;			// temp. variables for interm. results

	/* temporary matrix with 28 elements
	 * default dimensions: 1x28
	 * to track matrix elements, use array tempMat_28_0_arr */
	float32_t tempMat_28_0_arr[28] = {0};
	arm_matrix_instance_f32 tempMat_28_0 = {1, 28, tempMat_28_0_arr};

	/* temporary matrix with 784 elements
	 * default dimensions: 28x28
	 * to track matrix elements, use array tempMat_784_0_arr */
	float32_t tempMat_784_0_arr[784] = {0};
	arm_matrix_instance_f32 tempMat_784_0 = {28, 28, tempMat_784_0_arr};

	/* temporary matrix with 784 elements
	 * default dimensions: 28x28
	 * to track matrix elements, use array tempMat_784_1_arr */
	float32_t tempMat_784_1_arr[784] = {0};
	arm_matrix_instance_f32 tempMat_784_1 = {28, 28, tempMat_784_1_arr};

	/*---------------------- Input Matrix Standardisation --------------------*/

	/* compute column-wise mean
	 * column-wise stored in tempMat_28_0 (dim: 1x28) */
	columnMean(&tempMat_28_0, mat);

	/* standardise input matrix
	 * subtract column-wise mean(s) from corresponding column(s)
	 * in-place computation, original input matrix holds the results */
	for (col = 0; col < (mat->numCols); col++)
		for (row = 0; row < (mat->numRows); row++)
			mat->pData[row * (mat->numCols) + col] -= tempMat_28_0.pData[col];

	/*------------------------------ mat.T DOT mat ---------------------------*/

	/* transpose of input matrix, stored in tempMat_784_0 */
	arm_mat_trans_f32(mat, &tempMat_784_0);

	/* multiply mat's transpose with mat (exact positions)
	 * results in tempMat_784_1 */
	arm_mat_mult_f32(/*mat.T*/ &tempMat_784_0, mat,
			             /*mat.T dot mat*/ &tempMat_784_1);

	/*----------------------- Power Iteration Algorithm ----------------------*/

	/* generate 28 random numbers
	 * stored in tempMat_28_0 */
	for(i = 0; i < 28; i++)
	{
		HAL_RNG_GenerateRandomNumber(&hrng,
			/*random number will be stored here*/ (uint32_t*)&tempFloat32);
		tempMat_28_0.pData[i] = tempFloat32;
	}

	/* scale the random numbers in tempMat_28_0 between 0, 1
	 * find max, divide everything by max */
	tempFloat32 /*(max)*/ = tempMat_28_0.pData[0];
	for(i = 1; i < 28; i++)
		if(tempFloat32 /*(max)*/ < tempMat_28_0.pData[i])
			tempFloat32 /*(max)*/ = tempMat_28_0.pData[i];

	for(i = 0; i < 28; i++)
		tempMat_28_0.pData[i] = tempMat_28_0.pData[i] / /*(max)*/ tempFloat32;

	/* reshape temporary matrices for upcoming calculations
	 * calc overview: tempMat_784_1 (28x28) dot tempMat_28_0 (28x1)
	 * = tempMat_784_0 (28x1)
	 * note: tempMat_784_0 can support upto 28x28. As it is not in use,
	 * it makes sense to change its dimensions to 28x1,
	 * than making new 28x1 vec. */
	tempMat_28_0.numRows = tempMat_784_0.numRows = 28;
	tempMat_28_0.numCols = tempMat_784_0.numCols = 1;

	/* Iterate here for eigenvector */
	for(i = 0; i < 100; i++)
	{
		arm_mat_mult_f32(&tempMat_784_1, &tempMat_28_0, &tempMat_784_0);
		tempFloat32 /*(l2norm)*/ = computeL2Norm(&tempMat_784_0);
		for(j = 0; j < 28; j++)
			tempMat_28_0.pData[j] =
				tempMat_784_0.pData[j] / /*(l2norm)*/ tempFloat32;
	}
	/* note: tempMat_28_0 is the dominant eigenvector now */

	/* manipulate signs:
	 * it is to be ensured that the highest absolute value in the vector be
	 * positive. */
	j = 0;
	tempFloat32 /*(max)*/ = fabsf(tempMat_28_0.pData[0]);
	for(i = 1; i < 28; i++)
		if(tempFloat32 /*(max)*/ < fabsf(tempMat_28_0.pData[i]))
		{
			j = i;
			tempFloat32 /*(max)*/ = fabsf(tempMat_28_0.pData[i]);
		}

	if (tempMat_28_0.pData[j] < 0)
		for(i = 0; i < 28; i++)
			tempMat_28_0.pData[i] *= -1;

	/* note: tempMat_28_0 is still the dominant eigenvector now
	 * only it's signs are manipulated to ensure that the highest absolute
	 * value in the vector is positive */

	/*--------------------- Final Output of PCA ------------------------------*/

	/* dot product of input (standardised) matrix and eigenvector gives the PCA
	 * component (or feature) */
	arm_mat_mult_f32(mat, &tempMat_28_0, pcaVec);

	/*
	j = 0;
	max = fabsf(pcaVec->pData[0]);
	for(i = 1; i < 28; i++)
	{
		if(max < fabsf(pcaVec->pData[i]))
		{
			j = i;
			max = fabsf(pcaVec->pData[i]);
		}
	}*/
}

/*------------------------------------------------------------------------------
 * computeL2Norm
 * Description: Computes the L2 norm of the vector mat
 * Parameters : mat		 (type=arm_matrix_instance_f32, size=(28,1)
 *----------------------------------------------------------------------------*/
static float32_t computeL2Norm(arm_matrix_instance_f32* mat)
{
	int i;								// variable for looping
	float32_t sum = 0;		// variable L2 norm
	float32_t l2norm;			//

	// sum of squares of all elements
	for(i = 0; i < 28; i++)
		sum += (mat->pData[i] * mat->pData[i]);

	// square root of sum
	arm_sqrt_f32(sum, &l2norm);
	return l2norm;
}

/*------------------------------------------------------------------------------
 * columnMean
 * Description: Computes the column-wise mean of the input matrix
 * Parameters : colMean   (type=arm_matrix_instance_f32, size=(1,28)
 * 				mat		  (type=arm_matrix_instance_f32, size=(28,28)
 *----------------------------------------------------------------------------*/
static void columnMean(arm_matrix_instance_f32* colMean,
					   arm_matrix_instance_f32* mat)
{
    int row, col;
    float temp = 0;

    for (col = 0; col < (mat->numCols); col++)
    {
        for (row = 0; row < (mat->numRows); row++)
        {
            temp += mat->pData[row * (mat->numCols) + col];
        }
        temp /= 28;
        colMean->pData[col] = temp;
        temp = 0;
    }
}

/*------------------------------------------------------------------------------
 * addBias
 * Description: adds bias as in Z = W.X + b
 * Parameters : mat    (type=arm_matrix_instance_f32, size=(1,28)
 * 							addMat (type=arm_matrix_instance_f32, size=(28,28)
 *----------------------------------------------------------------------------*/
static void addBias(arm_matrix_instance_f32* mat, arm_matrix_instance_f32* bias)
{
	int i;

	for(i = 0; i < mat->numRows; i++)
		mat->pData[i] += bias->pData[i];
}

/*------------------------------------------------------------------------------
 * relu
 * Description: computes ReLU on every element of input vector
 * Parameters : mat    (type=arm_matrix_instance_f32, size=(28,1)
 *----------------------------------------------------------------------------*/
static void relu(arm_matrix_instance_f32* mat)
{
	int i;

	for(i = 0; i < mat->numRows; i++)
		if(mat->pData[i] < 0)
			mat->pData[i] = 0;
}

/*------------------------------------------------------------------------------
 * softmax
 * Description: computes softmax on every element of input vector
 * Parameters : mat    (type=arm_matrix_instance_f32, size=(28,1)
 *----------------------------------------------------------------------------*/
static void softmax(arm_matrix_instance_f32* mat)
{
	int i;
	float32_t sum = 0;

	/* Compute exponent of every element */
	for(i = 0; i < mat->numRows; i++)
		mat->pData[i] = exp(mat->pData[i]);

	/* Compute sum of every exponent-ed element */
	for(i = 0; i < mat->numRows; i++)
		sum += mat->pData[i];

	/* divide every exponent-ed element with the sum */
	for(i = 0; i < mat->numRows; i++)
		mat->pData[i] /= sum;
}

/*...*/
