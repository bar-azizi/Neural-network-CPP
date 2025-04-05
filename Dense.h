#ifndef DENSE_H
#define DENSE_H

#include "Activation.h"


class Dense
{
 public:

  /**
   * constructor - represents a layer in neural network
   * @param weight a Matrix represents the weights
   * @param bias a Vector (one col Matrix)
   * @param activation_func A function that acts on a Matrix object
   */
  Dense(const Matrix& weight, const Matrix& bias, activation_t
  activation_func);

  // getters
  /**
   * @returns the weight Matrix object
   */
  const Matrix& get_weights() const;

  /**
   * @returns the bias Matrix object
   */
  const Matrix& get_bias() const;

  /**
   * @returns the activation function
   */
  activation_t get_activation() const;

  // methods
  Matrix operator()(const Matrix& input);

  // operators
 private:
  activation_t _activation_func;
  Matrix _weights;
  Matrix _bias;
};


#endif //DENSE_H
