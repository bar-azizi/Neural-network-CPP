//MlpNetwork.h

#ifndef MLPNETWORK_H
#define MLPNETWORK_H

#include "Dense.h"

#define MLP_SIZE 4
#define WEIGHT_SIZE_ERR_MSG "weight matrix size err"
#define BIAS_SIZE_ERR_MSG "bias matrix size err"

using activation::relu;
using activation::softmax;

/**
 * @struct digit
 * @brief Identified (by Mlp network) digit with
 *        the associated probability.
 * @var value - Identified digit value
 * @var probability - identification probability
 */
typedef struct digit {
	unsigned int value;
	float probability;
} digit;

const matrix_dims img_dims = {28, 28};
const matrix_dims weights_dims[] = {{128, 784},
									{64,  128},
									{20,  64},
									{10,  20}};
const matrix_dims bias_dims[] = {{128, 1},
								 {64,  1},
								 {20,  1},
								 {10,  1}};

class MlpNetwork
{
 public:

  //constructor
  /**
   * constructor of MlpNetwork
   * @param weights An array of Matrix objects - the weights matrices
   * @param biases An array of biases vectors
   */
  MlpNetwork (const Matrix weights[], const Matrix biases[]);

  //operators
  /**
   * activate the network
   * @param mat A Matrix object
   * @return A digit struct, with the result number and score
   */
  digit operator()(Matrix & mat);

  private:
  //Network layers
  Dense _layer_1;
  Dense _layer_2;
  Dense _layer_3;
  Dense _layer_4;

};

#endif // MLPNETWORK_H