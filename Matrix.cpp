#include "Matrix.h"

/////////////////////////////////// CONSTRUCTORS //////////////////////////////

Matrix::Matrix(): Matrix (ONE_ROW, ONE_COL){}


Matrix::Matrix(int rows, int cols):
    dims{rows, cols}
{
  if(rows<=0 || cols<=0)
  {
    throw length_error(LEN_ERR_MSG);
  }
  _matrix = new float[rows*cols];
  for(int index=0; index<TOTAL_COORDS; index++)
  {
    _matrix[index] = 0.0F;
  }
}


/// cpy constructor
Matrix::Matrix(const Matrix& other):
dims{other.dims.rows, other.dims.cols}
{
  _matrix = new float[TOTAL_COORDS];
  for(int index=0; index<TOTAL_COORDS; index++)
  {
    _matrix[index] = other._matrix[index];
  }
}


//////////////////////////////////// DESTRUCTOR ///////////////////////////////

Matrix::~Matrix()
{
  delete[] _matrix;
}

//////////////////////////////// GETTERS //////////////////////////////////////

int Matrix::get_rows() const
{
  return dims.rows;
}


int Matrix::get_cols() const
{
  return dims.cols;
}

////////////////////////////// OTHER METHODS //////////////////////////////////

Matrix& Matrix::transpose()
{
  //initialize new matrix, columns and rows are opposite
  float *t_mat = new float[TOTAL_COORDS];
  for(int t_row=0; t_row < dims.cols; t_row++)
  {
    for(int t_col=0; t_col < dims.rows; t_col++)
    {
      //assign transpose[i][j] = mat[j][i]
      t_mat[(dims.rows * t_row) + t_col] = _matrix[(dims.cols * t_col) +
                                                   t_row];
    }
  }

  delete[] _matrix; // deallocate old matrix

  // assign new matrix
  int new_rows = dims.cols;
  _matrix = t_mat;
  dims.cols = dims.rows;
  dims.rows = new_rows;
  return *this;
}


Matrix& Matrix::vectorize()
{
  //change only rows and cols number
  dims.rows = TOTAL_COORDS;
  dims.cols = ONE_COL;
  return *this;
}


void Matrix::plain_print() const
{
  for(int row=0; row<dims.rows; row++)
  {
    for(int col=0; col<dims.cols; col++)
    {
      cout << _matrix[row*dims.cols + col] << " ";
    }
    cout << endl;
  }
}


Matrix Matrix::dot(const Matrix& other) const
{
  if((dims.cols != other.dims.cols) || (dims.rows != other.dims.rows))
  {
    throw length_error(DIFFER_SIZE_ERR_MSG);
  }
  Matrix dotted_mat = Matrix(dims.rows, dims.cols);
  for(int index=0; index<TOTAL_COORDS; index++)
  {
    dotted_mat._matrix[index] = _matrix[index] * other._matrix[index];
  }
  return dotted_mat;
}


float Matrix::norm() const
{
  float mat_norm = 0;
  for(int index=0; index<TOTAL_COORDS; index++)
  {
    mat_norm += (_matrix[index]*_matrix[index]);
  }
  return sqrt(mat_norm);
}


int Matrix::argmax() const
{
  float max_coord = _matrix[0];
  int max_index = 0;
  for(int index=0; index<TOTAL_COORDS; index++)
    {
      if (_matrix[index] > max_coord)
      {
        max_coord = _matrix[index];
        max_index = index;
      }
    }
  return max_index;
}


float Matrix::sum() const
{
  float mat_sum = 0;
  for(int index=0; index<TOTAL_COORDS; index++)
  {
    mat_sum += _matrix[index];
  }
  return mat_sum;
}


Matrix Matrix::operator+(const Matrix& rhs) const
{
  if((dims.cols != rhs.dims.cols) || (dims.rows != rhs.dims.rows))
  {
    throw length_error(DIFFER_SIZE_ERR_MSG);
  }
  Matrix added_mat = Matrix(dims.rows, dims.cols);
  for(int index=0; index<TOTAL_COORDS; index++)
  {
    added_mat._matrix[index] = _matrix[index] + rhs._matrix[index];
  }
  return added_mat;
}


Matrix& Matrix::operator+=(const Matrix& rhs)
{
  if((dims.cols != rhs.dims.cols) || (dims.rows != rhs.dims.rows))
  {
    throw length_error(DIFFER_SIZE_ERR_MSG);
  }
  for(int index=0; index<TOTAL_COORDS; index++)
  {
    _matrix[index] += rhs._matrix[index];
  }
  return *this;
}


Matrix& Matrix::operator=(const Matrix& rhs)
{
  if(this == &rhs)
  {
    return *this;
  }
  dims.cols = rhs.dims.cols;
  dims.rows = rhs.dims.rows;
  delete[] _matrix;
  _matrix = new float[TOTAL_COORDS];
  for(int index=0; index<TOTAL_COORDS; index++)
  {
    _matrix[index] = rhs._matrix[index];
  }
  return *this;
}


Matrix Matrix::operator*(const Matrix& rhs) const
{
  if(dims.cols != rhs.dims.rows)
  {
    throw length_error(MAT_MULT_ERR_MSG);
  }
  Matrix mult_mat = Matrix(dims.rows, rhs.dims.cols);
  for(int row=0; row<dims.rows; row++)
  {
    for(int col=0; col<rhs.dims.cols; col++)
    {
      mult_mat._matrix[row*rhs.dims.cols + col] = mult(*this, rhs, row, col);
    }
  }
  return mult_mat;
}


Matrix Matrix::operator*(float c) const
{
  Matrix new_mat = Matrix(dims.rows, dims.cols);
  for(int index=0; index<TOTAL_COORDS; index++)
  {
    new_mat._matrix[index] = _matrix[index]*c;
  }
  return new_mat;
}


Matrix operator*(const float c, const Matrix& rhs)
{
  return rhs*c;
}


const float& Matrix::operator()(int i, int j) const
{
  if(j>dims.cols || i>dims.rows || i<0 || j<0)
  {
    throw out_of_range(OUT_OF_RNG_ERR_MSG);
  }
  return _matrix[i*dims.cols + j];
}


float& Matrix::operator()(int i, int j)
{
  if(j>dims.cols || i>dims.rows || i<0 || j<0)
  {
    throw out_of_range(OUT_OF_RNG_ERR_MSG);
  }
  return _matrix[i*dims.cols + j];
}


const float& Matrix::operator[](const int index) const
{
  if(index>TOTAL_COORDS || index<0)
  {
    throw out_of_range(OUT_OF_RNG_ERR_MSG);
  }
  return _matrix[index];
}


float& Matrix::operator[](int index)
{
  if(index>TOTAL_COORDS || index<0)
  {
    throw out_of_range(OUT_OF_RNG_ERR_MSG);
  }
  return _matrix[index];
}


ostream& operator<<(ostream& os, const Matrix& mat)
{
  if(!mat._matrix)
  {
    throw out_of_range(RANGE_ERR_MSG);
  }
  for(int row=0; row<mat.dims.rows; row++)
  {
    for(int col=0; col<mat.dims.cols; col++)
    {
      if(mat(row, col) > THRESHOLD)
      {
        os << "**";
      }
      else
      {
        os << "  ";
      }
    }
    os << "\n";
  }
  return os;
}


istream& operator>>(istream& is, Matrix& mat)
{
  if(!mat._matrix)
  {
    throw out_of_range(RANGE_ERR_MSG);
  }

  //check if stream length is too short
  is.seekg (0, std::ios::end);
  long int f_size = is.tellg ();
  is.seekg (0, std::ios::beg);
  long int exp_size = ((long int) (mat.dims.cols))*((long int) mat.dims
      .rows);
  exp_size *= CELL_SIZE;
  if(f_size != exp_size)
  {
    throw runtime_error(VAL_ERR_MSG);
  }

  //fill matrix
  for(int index=0; index<mat.dims.rows*mat.dims.cols; index++)
  {
    if(!is.good())
    {
      throw runtime_error(VAL_ERR_MSG);
    }
    is.read((char*) &mat._matrix[index], sizeof(float));
  }
  //check if last value read
  if(is.fail())
  {
    throw runtime_error(VAL_ERR_MSG);
  }
  return is;
}


/////////////////////////////////////// HELPER ////////////////////////////////
float mult(const Matrix& lhs, const Matrix& rhs, int r, int c)
{
  float cell_value = 0.0F;
  //calculate one cell in AB
  for(int k=0; k<lhs.dims.cols; k++)
  {
    cell_value += lhs._matrix[r*lhs.dims.cols+k]*rhs._matrix[k*rhs.dims
                                                             .cols+c];
  }
  return cell_value;
}

//////////////////////////////// BONUS ////////////////////////////////////////
Matrix Matrix::rref() const
{
  Matrix rref_mat (*this);
  if(rref_mat.is_zero_matrix())
  {
    return rref_mat;
  }
  rref_mat = rref_mat.rref_helper (0);

  int zero_index = dims.rows-1;
  for(int row= dims.rows - 1; row >= 0; row--)
  {
    if(rref_mat.is_zero_row (row))
    {
      rref_mat.swap_rows (row, zero_index);
      zero_index--;
    }
  }

  int to_organize = zero_index + 1;
  for(; zero_index>=0; zero_index--)
  {
    rref_mat.reverse_reduce (zero_index);
  }

  int start_row=0, start_col=0;
  while(to_organize > 0)
  {
    int is_zero = rref_mat.is_zero_col (start_col, start_row);
    if(is_zero >= 0)
    {
      rref_mat.swap_rows (is_zero, start_row);
      to_organize--;
      start_row++;
    }
    start_col++;
  }
  return rref_mat;
}

///////////////////////////// BONUS HELPERS ///////////////////////////////////
Matrix& Matrix::rref_helper(int ro)
{
  if(ro==dims.rows)
  {
    return *this;
  }
  int col=0;
  while (is_zero_col(col, ro) < 0){col++;}
  int row=ro;
  while(_matrix[row*dims.cols + col] == 0) {row++;}
  divide_row(row, _matrix[row*dims.cols + col]);
  for(int r=row+1; r<dims.rows; r++)
  {
    if(_matrix[r*dims.cols + col] != 0)
    {
      divide_row (r, _matrix[r*dims.cols + col]);
      sub_row(r, row, 1);
    }
  }
  swap_rows (row, ro);
  return rref_helper(ro+1);
}

bool Matrix::is_zero_matrix() const
{
  for(int index=0; index<TOTAL_COORDS; index++)
  {
    if(_matrix[index] != FLOAT_ZERO)
    {
      return false;
    }
  }
  return true;
}


int Matrix::is_zero_col (int col, int r) const
{
  for(int row=r; row<dims.rows; row++)
  {
    if(_matrix[(row*dims.cols) + col] != FLOAT_ZERO)
    {
      return row;
    }
  }
  return NOT_FOUND;
}


bool Matrix::is_zero_row (int row) const
{
  for(int col=0; col<dims.cols; col++)
  {
    if(_matrix[(row*dims.cols) + col] != FLOAT_ZERO)
    {
      return false;
    }
  }
  return true;
}



Matrix& Matrix::divide_row(int row, float num)
{
  for(int col=0; col<dims.cols; col++)
  {
    if(_matrix[row*dims.cols + col] != 0)
    {
      _matrix[row*dims.cols + col] /= num;
    }
  }
  return *this;
}

Matrix& Matrix::sub_row(int from, int to, float mult)
{
  for(int col=0; col<dims.cols; col++)
  {
    _matrix[from*dims.cols + col] -= mult*_matrix[to*dims.cols + col];
  }
  return *this;
}


void Matrix::swap_rows(int r1, int r2)
{
  if(r1==r2)
  {
    return;
  }
  float* temp_r2 = new float[dims.cols];
  for(int index=0; index<dims.cols; index++)
  {
    temp_r2[index] = _matrix[r1*dims.cols + index];
    _matrix[r1*dims.cols + index] = _matrix[r2*dims.cols + index];
  }
  for(int index=0; index<dims.cols; index++)
  {
   _matrix[r2*dims.cols + index] =  temp_r2[index];
  }
  delete[] temp_r2;
}


void Matrix::reverse_reduce(int row)
{
  int col=0;
  while(_matrix[row*dims.cols + col] == FLOAT_ZERO){col++;}

  for(int r=0; r<dims.rows; r++)
  {
    if((_matrix[r*dims.cols + col] != FLOAT_ZERO) && r!=row)
    {
      sub_row (r, row, _matrix[r*dims.cols + col]);
    }
  }
}