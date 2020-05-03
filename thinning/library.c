#include "library.h"

void m_connectivity(int height, int width, unsigned int **array) {
  for ( int x = 1; x < height - 1; x++ ) {
    for ( int y = 1; y < width - 1; y++ ) {
      if (array[x][y] == 1) {
        int d_1 = (x > 2) && (array[x - 2][y - 1] == 0 || array[x - 2][y] == 1 || array[x - 1][y - 1] == 1);
        int d_2 = (y > 2) && (array[x + 1][y - 2] == 0 || array[x][y - 2] == 1 || array[x + 1][y - 1] == 1);
        int d_3 = (y < width - 2) &&
                  (array[x - 1][y + 2] == 0 || array[x][y + 2] == 1 || array[x - 1][y + 1] == 1);
        int d_4 = (y < width - 2) &&
                  (array[x + 1][y + 2] == 0 || array[x][y + 2] == 1 || array[x + 1][y + 1] == 1);

        if (array[x - 1][y + 1] == 1 && (array[x - 1][y] == 1 && d_1)) {
          array[x - 1][y] = 0;
        }

        if (array[x - 1][y - 1] == 1 && (array[x][y - 1] == 1 && d_2)) {
          array[x][y - 1] = 0;
        }

        if (array[x + 1][y + 1] == 1 && (array[x][y + 1] == 1 && d_3)) {
          array[x + 1][y] = array[x][y + 1] = 0;
        }

        if (array[x - 1][y + 1] == 1 && (array[x][y + 1] == 1 && d_4)) {
          array[x][y + 1] = 0;
        }
      }
    }
  }
}

int zhang_and_suen_binary_thinning_(int height, int width, unsigned int **array) {
  int capacity = height * width;
  int k = 0;
  int flag_removed_point = 1;

  int **removed_points = (int **) malloc(capacity * sizeof(int *));
  for ( int i = 0; i < capacity; i++ ) {
    removed_points[i] = (int *) malloc(sizeof(int) * 2);
  }

  while (flag_removed_point) {
    flag_removed_point = 0;

    for ( int x = 1; x < height - 1; x++ ) {
      for ( int y = 1; y < width - 1; y++ ) {
        if (array[x][y] == 1) {
          // get 8-neighbors
          unsigned int P2 = array[x - 1][y];
          unsigned int P3 = array[x - 1][y + 1];
          unsigned int P4 = array[x][y + 1];
          unsigned int P5 = array[x + 1][y + 1];
          unsigned int P6 = array[x + 1][y];
          unsigned int P7 = array[x + 1][y - 1];
          unsigned int P8 = array[x][y - 1];
          unsigned int P9 = array[x - 1][y - 1];

          unsigned int B_P1 = P2 + P3 + P4 + P5 + P6 + P7 + P8 + P9;
          int condition_1 = (2 <= B_P1) && (B_P1 <= 6);

          unsigned int n_1[] = { P2, P3, P4, P5, P6, P7, P8, P9 };
          unsigned int n_2[] = { P3, P4, P5, P6, P7, P8, P9, P2 };
          int A_P1 = 0;

          for ( int i = 0; i < 8; i++ ) {
            if (n_1[i] == 0 && n_2[i] == 1) {
              A_P1++;
            }
          }

          int condition_2 = A_P1 == 1;
          int condition_3 = P2 * P4 * P6 == 0;
          int condition_4 = P4 * P6 * P8 == 0;

          if (condition_1 && condition_2 && condition_3 && condition_4) {
            removed_points[k][0] = x;
            removed_points[k][1] = y;
            flag_removed_point = 1;
            k++;
          }
        }
      }
    }

    for ( int i = 0; i < k; i++ ) {
      int x = removed_points[i][0];
      int y = removed_points[i][1];
      array[x][y] = 0;
    }

    k = 0;

    for ( int x = 1; x < height - 1; x++ ) {
      for ( int y = 1; y < width - 1; y++ ) {
        if (array[x][y] == 1) {
          // get 8-neighbors
          unsigned int P2 = array[x - 1][y];
          unsigned int P3 = array[x - 1][y + 1];
          unsigned int P4 = array[x][y + 1];
          unsigned int P5 = array[x + 1][y + 1];
          unsigned int P6 = array[x + 1][y];
          unsigned int P7 = array[x + 1][y - 1];
          unsigned int P8 = array[x][y - 1];
          unsigned int P9 = array[x - 1][y - 1];

          // B_P1 is the number of nonzero neighbors of P1=(x, y)
          unsigned int B_P1 = P2 + P3 + P4 + P5 + P6 + P7 + P8 + P9;
          int condition_1 = (2 <= B_P1) && (B_P1 <= 6);

          unsigned int n_1[] = { P2, P3, P4, P5, P6, P7, P8, P9 };
          unsigned int n_2[] = { P3, P4, P5, P6, P7, P8, P9, P2 };
          int A_P1 = 0;

          for ( int i = 0; i < 8; i++ ) {
            if (n_1[i] == 0 && n_2[i] == 1) {
              A_P1++;
            }
          }

          int condition_2 = A_P1 == 1;
          int condition_3 = P2 * P4 * P8 == 0;
          int condition_4 = P2 * P6 * P8 == 0;

          if (condition_1 && condition_2 && condition_3 && condition_4) {
            removed_points[k][0] = x;
            removed_points[k][1] = y;
            flag_removed_point = 1;
            k++;
          }
        }
      }
    }

    for ( int i = 0; i < k; i++ ) {
      int x = removed_points[i][0];
      int y = removed_points[i][1];
      array[x][y] = 0;
    }

    k = 0;
  }

  for ( int i = 0; i < capacity; i++ ) {
    free(removed_points[i]);
  }


  m_connectivity(height, width, array);
}

static
PyObject *zhang_and_suen_binary_thinning(PyObject *_, PyObject *args) {
  PyArrayObject *matrix = NULL;
  int code;

  if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &matrix)) {
    PyErr_SetString(PyExc_TypeError, "Unable to read given arguments!");
    return NULL;
  }

  if (PyArray_NDIM(matrix) != 2) {
    PyErr_SetString(PyExc_TypeError, "matrix must be a 2d array!");
    return NULL;
  }

  PyArray_Descr *descr = PyArray_DESCR(matrix);

  if (descr->type_num != NPY_UINT32) {
    PyErr_SetString(PyExc_TypeError, "Type np.uint32 expected for matrix array!");
    return NULL;
  }

  unsigned int **array;
  npy_intp shape[2];

  // Convert numpy array to c-styled array.
  code = PyArray_AsCArray((PyObject **) &matrix, &array, shape, 2, descr);
  if (code < 0) {
    PyErr_SetString(PyExc_TypeError, "Unable to convert to c array!");
    return NULL;
  }

  zhang_and_suen_binary_thinning_((int) shape[0], (int) shape[1], array);

  code = PyArray_Free((PyObject *) matrix, array);
  if (code < 0) {
    PyErr_SetString(PyExc_TypeError, "unable to cleanup array");
    return NULL;
  }

  Py_RETURN_NONE;
}

static PyMethodDef methods[] = {
        {
                "zhang_and_suen_binary_thinning",
                zhang_and_suen_binary_thinning,
                METH_VARARGS,
                "Applies binary thinning to given digital image"
        },

        /* Sentinel */
        { NULL, NULL, 0, NULL }
};

static struct PyModuleDef module_definition = {
        PyModuleDef_HEAD_INIT,
        "thinning",
        "A Zhang and Suen binary thinning implementation",
        -1,
        methods
};

PyMODINIT_FUNC
PyInit_thinning(void) {
  Py_Initialize();
  import_array();

  return PyModule_Create(&module_definition);
}
