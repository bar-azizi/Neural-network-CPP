// Matrix.h
#ifndef MATRIX_H
#define MATRIX_H

////////////////////////////////  INCLUDES AND USING //////////////////////////
#include <iostream>
#include <cmath>
#include <fstream>
#include <cmath>

using std::cout;
using std::endl;
using std::sqrt;
using std::ostream;
using std::istream;
using std::runtime_error;
using std::out_of_range;
using std::length_error;
using std::exp;
using std::string;
using std::bad_alloc;
/////////////////////////// DEFINES & TYPEDEFS ////////////////////////////////
#define ONE_ROW 1
#define ONE_COL 1
#define TOTAL_COORDS (dims.rows*dims.cols)
#define THRESHOLD 0.1
#define VAL_ERR_MSG "Not enough values for the matrix"
#define RANGE_ERR_MSG "No Matrix"
#define DIFFER_SIZE_ERR_MSG "Trying to multiply a different size matrix !"
#define LEN_ERR_MSG "Value not in the right length !"
#define MAT_MULT_ERR_MSG "Matrices are not in the right size to multiply"
#define OUT_OF_RNG_ERR_MSG "Indexes are out of range !"
#define CELL_SIZE sizeof(float)
#define FLOAT_ZERO 0.0F
#define NOT_FOUND (-1)

///////////////////////////////////////////////////////////////////////////////

/**
 * @struct matrix_dims
 * @brief Matrix dimensions container. Used in MlpNetwork.h and main.cpp
 */
typedef struct matrix_dims
{
	int rows, cols;
} matrix_dims;

// Insert Matrix class here...
class Matrix
{

 public:
  //constructors

  /**
 * constructor
 * @param rows - num of rows the matrix will have
 * @param cols - num of cols
 * builds a matrix of rows X cols
 * @return a matrix object
 */
  Matrix(int rows, int cols);

  /**
  * default constructor
  * builds a matrix of 1X1
  * @return a matrix object
  */
  Matrix();

  //cpy constructor

  /**
 * copy constructor
 * @param other - a Matrix object
 * makes a new Matrix copy of the matrix
 * returns the new allocated matrix
 */
  Matrix(const Matrix& other);

  //destructor

  /**
 * destroys a matrix
 * free all it's memory allocations
 */
  ~Matrix();


  //getters

  /**
 * @return the rows number of the matrix
 */
  int get_rows() const;

  /**
 * @return the columns number of the matrix
 */
  int get_cols() const;

  //methods

  /**
 * build the transpose form of the matrix
 * @returns the new allocated transpose matrix
 */
  Matrix& transpose();


  /**
 * flatten the matrix into vector
 * each coordinate of the vector is cell (matrix_columns_umber*row+col) of
 * the matrix
 * returns the vector
 */
  Matrix& vectorize();


  /**
 * prints the matrix to the user
 */
  void plain_print() const;


  /**
 * calculates dot matrix with other matrix
 * @param other - a matrix object
 * @return a new allocated matrix object,
 * each coordinate is this(i,j)*other(i,j)
 */
  Matrix dot(const Matrix& other) const;


  /**
 * @returns the Frobenius norm value of the matrix
 */
  float norm() const;


  /**
 * @returns the index of the largest coordinate of the matrix
 */
  int argmax() const;


  /**
 * @return the sum of all mat values (sum(Matrix[i][j]))
 */
  float sum() const;

  //operators


  /**
 * @param rhs a Matrix object
 * @return a new allocated Matrix object reference after the change
 * chaining this operator is possible
 */
  Matrix operator+(const Matrix& rhs) const;


  /**
 * @param rhs a Matrix object - added to this from the right
 * add to every coordinate of lhs the coordinate value of rhs
 * @return lhs reference after the change
 * chaining this operator is not possible
 */
  Matrix& operator+=(const Matrix& rhs);


  /**
 * change all matrix values to the rhs matrix values
 */
  Matrix& operator=(const Matrix& rhs);

  /**
  * does Matrix multiplication
  * @param rhs - an Matrix object, the right matrix
  * @returns a new allocated multiplication result Matrix object
  */
  Matrix operator*(const Matrix& rhs) const;


  /**
 * multiplies the matrix by scalar from the right
 * @param c float
 * @return a new allocated Matrix object. with values after the multiplication
 */
  Matrix operator*(float c) const;

  /**
 * @param i row index
 * @param j column index
 * @return Matrix[i][j], doesnt allow index change
 */
  const float& operator()(int i, int j) const;


  /**
 * @param i row index
 * @param j column index
 * @return Matrix[i][j], allows index change
 */
  float& operator()(int i, int j);


  /**
 * allows matrix indexing
 * @param index int index to get
 * @return the value of the index as if the matrix is a vector
 * doesnt allow index change
 */
  const float& operator[](int index) const;


  /**
 * allows matrix indexing
 * @param index int index to get
 * @return the value of the index as if the matrix is a vector
 * allow index change
 */
  float& operator[](int index);

  //friends

/**
 * multiplies the matrix by scalar from the left
 * @param c float number
 * @param rhs a Matrix object
 * @return a new allocated Matrix object, values are multiplied by c
 */
  friend Matrix operator*(float c, const Matrix& rhs);


  /**
 * pretty prints the image in the matrix
 * @param os out stream to print to
 * @param mat Matrix object to pretty print
 * @return the out stream for chaining
 */
  friend ostream& operator<<(ostream& os, const Matrix& mat);


  /**
 * fill matrix with values
 * @param is input stream to get information from
 * @param mat Matrix object too put values in
 * @return the input stream for chaining
 */
  friend istream& operator>>(istream& is, Matrix& mat);


  /**
   * calculates the Reduced Row Echelon Form of the matrix
   * @returns A new allocated Matrix object, that is the rref of this
   */
  Matrix rref() const;

 private:
  //num of rows and cols
  matrix_dims dims;

  //Matrix itself
  float *_matrix;

  Matrix& rref_helper(int ro);
  bool is_zero_matrix() const;
  int is_zero_col (int col, int r) const;
  Matrix& divide_row(int row, float num);
  Matrix& sub_row(int from, int to, float mult);
  bool is_zero_row (int row) const;
  void swap_rows(int r1, int r2);
  void reverse_reduce(int row);

  /**
 * calculates one cell in matrix multiplication
 * @param lhs the Matrix object on the left side
 * @param rhs the Matrix object on the right size
 * @param r row number
 * @param c column number
 * @return the cell value
 */
  friend float mult(const Matrix& lhs, const Matrix& rhs, int r, int c);

};
#endif //MATRIX_H