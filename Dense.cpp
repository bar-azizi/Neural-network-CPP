#include "Dense.h"

Dense::Dense(const Matrix& weight, const Matrix& bias, const activation_t
activation_func):
_activation_func(activation_func), _weights(weight), _bias(bias)
{
  if(bias.get_cols() != ONE_COL || weight.get_rows() != bias.get_rows())
  {
    throw length_error (LEN_ERR_MSG);
  }
}


const Matrix &Dense::get_weights () const
{
  return _weights;
}


const Matrix &Dense::get_bias () const
{
  return _bias;
}


activation_t Dense::get_activation () const
{
  return _activation_func;
}


Matrix Dense::operator()(const Matrix& input)
{
  Matrix result = _weights*input;
  result += _bias;
  return _activation_func(result);
}