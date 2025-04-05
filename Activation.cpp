#include "Activation.h"


Matrix activation::relu(const Matrix& mat)
{
  Matrix new_mat = Matrix(mat.get_rows(), mat.get_cols());
  for(int index=0; index<ALL_COORDS; index++)
  {
    new_mat[index] = (mat[index]<=0) ? 0 : mat[index];
  }
  return new_mat;
}



Matrix activation::softmax(const Matrix& mat)
{
  Matrix new_mat = Matrix(mat.get_rows(), mat.get_cols());
  float sum = 0;
  for(int index=0; index<ALL_COORDS; index++)
  {
    sum += exp (mat[index]);
    new_mat[index] = exp (mat[index]);
  }
  sum = 1/sum;
  new_mat = new_mat*sum;
  return new_mat;
}