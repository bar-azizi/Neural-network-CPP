#include "MlpNetwork.h"


MlpNetwork::MlpNetwork (const Matrix weights[MLP_SIZE], const Matrix
biases[MLP_SIZE]):
    _layer_1(weights[0], biases[0], relu),
    _layer_2(weights[1], biases[1], relu),
    _layer_3(weights[2], biases[2], relu),
    _layer_4(weights[3], biases[3], softmax)
{
   for(int i=0; i<MLP_SIZE-1; i++)
   {
     if(weights[i].get_rows() != weights[i+1].get_cols())
     {
       throw length_error(WEIGHT_SIZE_ERR_MSG);
     }
   }
}


digit MlpNetwork::operator()(Matrix & mat)
{
  mat.vectorize();
  Matrix res1 (_layer_1(mat));
  Matrix res2 (_layer_2(res1));
  Matrix res3 (_layer_3(res2));
  Matrix res4 (_layer_4(res3));
  return digit{(unsigned int) res4.argmax(),
               res4[res4.argmax()]};
}