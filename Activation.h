#ifndef ACTIVATION_H
#define ACTIVATION_H

#include "Matrix.h"
#define ALL_COORDS (mat.get_rows()*mat.get_cols())

typedef Matrix (*activation_t)(const Matrix&);

// Insert Activation class here...
namespace activation
{
    /**
    * An activation function
    * every coordinate of the new matrix is 0 if mat[i][j]<0, else mat[i][j]
    * @param mat A Matrix object
    * @return A new allocated Matrix object, after the change
    */
    Matrix relu(const Matrix& mat);


    /**
    * An activation function
    * every coordinate of the new matrix is a distribution vector
    * relate to the matrix as a long vector
    * @param mat A Matrix object
    * @return
    */
    Matrix softmax(const Matrix& mat);
}

#endif //ACTIVATION_H