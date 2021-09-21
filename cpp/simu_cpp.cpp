//============================================================================
// Name        : simu_cpp.cpp
// Author      : Damien J
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C, Ansi-style
//============================================================================

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include "simu.hpp" 

std::unique_ptr<SimuICNS> mySimu;

static PyObject * initSimu(PyObject *self, PyObject *args){
    uint16_t x, y;
    if (!PyArg_ParseTuple(args, "HH", &x, &y))
        return NULL;
    mySimu =  std::make_unique<SimuICNS>(x, y);
    return Py_True;
}

static PyObject * getShape(PyObject *self){
    int ndim = 1;
    npy_intp shape[1] = {2};
    PyObject *result_array = PyArray_SimpleNew(ndim, shape, NPY_DOUBLE);
    double *result = (double *)PyArray_DATA((PyArrayObject *)result_array);
    result[0] = mySimu->getXShape();
    result[1] = mySimu->getYShape();
    return result_array;
}

static PyObject * disableNoise(PyObject *self){
    mySimu->disableNoise();
    return Py_True;
}


static PyObject * masterRst(PyObject *self){
    mySimu->masterRst();
    return Py_True;
}

static PyObject * getCurv(PyObject *self){
    int ndim = 2;
    npy_intp shape[2] = {static_cast<long int>(mySimu->getXShape()), static_cast<long int>(mySimu->getYShape())};
    PyObject *result_array = PyArray_SimpleNew(ndim, shape, NPY_DOUBLE);
    std::memcpy(PyArray_DATA((PyArrayObject*) result_array ),  mySimu->getCurV().data(), sizeof(double) *  mySimu->getXShape() * mySimu->getYShape());
    return result_array;
}

static PyObject * initContrast(PyObject *self, PyObject *args){
    double_t c_pos_arg, c_neg_arg, c_noise_arg;
    if (!PyArg_ParseTuple(args, "ddd", &c_pos_arg, &c_neg_arg, &c_noise_arg))
        return NULL;
    mySimu->set_th(c_pos_arg, c_neg_arg, c_noise_arg);
    return Py_True;
}

static PyObject * setDebug(PyObject *self, PyObject *args){
    uint8_t debug_arg;
    if (!PyArg_ParseTuple(args, "B", &debug_arg))
        return NULL;
    mySimu->setDebug(debug_arg);
    return Py_True;
}

static PyObject * initLatency(PyObject *self, PyObject *args){
    double_t lat_arg, jit_arg, ref_arg, tau_arg;
    if (!PyArg_ParseTuple(args, "dddd", &lat_arg, &jit_arg, &ref_arg, &tau_arg))
        return NULL;
    mySimu->set_lat(lat_arg, jit_arg, ref_arg, tau_arg);
    return Py_True;
}

static PyObject * initNoise(PyObject *self, PyObject *args){
    PyObject *pos_dist_arg;
    PyObject *neg_dist_arg;
    if (!PyArg_ParseTuple(args, "OO", &pos_dist_arg, &neg_dist_arg))
        return NULL;
    PyObject *pos_dist_array = PyArray_FROM_OTF(pos_dist_arg, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS);
    PyObject *neg_dist_array = PyArray_FROM_OTF(neg_dist_arg, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS);
    if (pos_dist_array == NULL || neg_dist_array == NULL){
        PyErr_SetString(PyExc_TypeError, "Could not convert argument to numpy array.");
        return NULL;
    }
    auto pos_dist_shape = PyArray_SHAPE(reinterpret_cast<PyArrayObject *>(pos_dist_array));
    auto neg_dist_shape = PyArray_SHAPE(reinterpret_cast<PyArrayObject *>(neg_dist_array));
    auto ndim = PyArray_NDIM(reinterpret_cast<PyArrayObject *>(pos_dist_array)) * PyArray_NDIM(reinterpret_cast<PyArrayObject *>(neg_dist_array));
    if (ndim != 4){
        PyErr_SetString(PyExc_ValueError, "The Dimension must be one n x 72");
    }
    if(pos_dist_shape[1] != 72 || neg_dist_shape[1] != 72 || neg_dist_shape[0] != pos_dist_shape[0]){
        PyErr_SetString(PyExc_ValueError, "The shape of the dist must be n x 72 .");
        return NULL;
    }
    const double *pos_dist = (double *)PyArray_DATA((PyArrayObject *)pos_dist_array);
    const double *neg_dist = (double *)PyArray_DATA((PyArrayObject *)neg_dist_array);
    mySimu->init_noise(pos_dist, neg_dist, pos_dist_shape[0]);
    Py_DECREF(pos_dist_array);
    Py_DECREF(neg_dist_array);
    return Py_True;
}

static PyObject * updateImg(PyObject *self, PyObject *args){
    PyObject *img_arg;
    uint64_t time_arg;
    if (!PyArg_ParseTuple(args, "OK", &img_arg, &time_arg))
        return NULL;
    PyObject *img_array = PyArray_FROM_OTF(img_arg, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS);
    auto img_shape = PyArray_SHAPE(reinterpret_cast<PyArrayObject *>(img_array));
    if(img_shape[1] != mySimu->getYShape() || img_shape[0] != mySimu->getXShape()){
        std::unique_ptr<char[]> buf( new char[ 100 ] );
        sprintf(buf.get(), "The Dimension of the image must be %d x %d and time is a one value array\n", mySimu->getYShape(), mySimu->getXShape());
        PyErr_SetString(PyExc_ValueError, buf.get());
        return NULL;
    }
    const double *img = (double *)PyArray_DATA((PyArrayObject *)img_array);
    std::vector<Event> ev;
    mySimu->update_img(img, time_arg, ev);
    auto stream = PyDict_New();
    npy_intp shape[1] = {static_cast<long int>(ev.size())};
    PyObject *ts_array = PyArray_SimpleNew(1, shape, NPY_UINT64);
    PyObject *x_array = PyArray_SimpleNew(1, shape, NPY_UINT16);
    PyObject *y_array = PyArray_SimpleNew(1, shape, NPY_UINT16);
    PyObject *p_array = PyArray_SimpleNew(1, shape, NPY_UINT8);
    uint64_t *ts = (uint64_t *)PyArray_DATA((PyArrayObject *)ts_array);
    uint16_t *x = (uint16_t *)PyArray_DATA((PyArrayObject *)x_array);
    uint16_t *y = (uint16_t *)PyArray_DATA((PyArrayObject *)y_array);
    uint8_t *p = (uint8_t *)PyArray_DATA((PyArrayObject *)p_array);
    for(long unsigned int i = 0; i < ev.size(); i++){
        ts[i] = ev.at(i).ts_;
        x[i] = ev.at(i).x_;
        y[i] = ev.at(i).y_;
        p[i] = ev.at(i).p_;
    }
    PyDict_SetItem(stream, PyUnicode_FromString("ts"), ts_array);
    PyDict_SetItem(stream, PyUnicode_FromString("x"), x_array);
    PyDict_SetItem(stream, PyUnicode_FromString("y"), y_array);
    PyDict_SetItem(stream, PyUnicode_FromString("p"), p_array);
    Py_DECREF(img_array);
    Py_DECREF(x_array);
    Py_DECREF(y_array);
    Py_DECREF(ts_array);
    Py_DECREF(p_array);
    ev.clear();
    return stream;
}

static PyObject * initImg(PyObject *self, PyObject *args){
    PyObject *img_arg;
    if (!PyArg_ParseTuple(args, "O", &img_arg))
        return NULL;
    PyObject *img_array = PyArray_FROM_OTF(img_arg, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS);
    if (img_array == NULL){
        PyErr_SetString(PyExc_TypeError, "Could not convert argument to numpy array.");
        return NULL;
    }
    auto img_shape = PyArray_SHAPE(reinterpret_cast<PyArrayObject *>(img_array));
    auto ndim = PyArray_NDIM(reinterpret_cast<PyArrayObject *>(img_array));
    if (ndim != 2){
        PyErr_SetString(PyExc_ValueError, "The Dimension must be one n x m. Dont forget to taKe L from Lab.");
    }
    if(img_shape[1] != mySimu->getYShape() || img_shape[0] != mySimu->getXShape()){
        std::unique_ptr<char[]> buf( new char[ 100 ] );
        sprintf(buf.get(), "The shape of the image (%d, %d) is not the shape of the simulator (%d, %d) .\n", static_cast<int>(img_shape[1]), static_cast<int>(img_shape[0]), mySimu->getYShape(), mySimu->getXShape());
        PyErr_SetString(PyExc_ValueError, buf.get());
        return NULL;
    }
    const double *img = (double *)PyArray_DATA((PyArrayObject *)img_array);
    mySimu->init_img(img);
    Py_DECREF(img_array);
    return Py_True;
}

/*
    Module declaration
*/
static char module_docstring[] = "DVS pixel simulator";

PyDoc_STRVAR(
    disableNoise_doc,
    "Disable the background noise sensor.\n\n"
    "Parameters\n"
    "----------\n"
    "Returns\n"
    "----------\n"
    "result: bool\n"
    "   sucess\n\n");

PyDoc_STRVAR(
    initSimu_doc,
    "Initialize the simulator.\n\n"
    "Parameters\n"
    "----------\n"
    "y: numpy_array\n"
    "   Definition along y\n"
    "x: numpy_array\n"
    "   Definition along x axis\n"
    "Returns\n"
    "----------\n"
    "result: bool (sucess)\n"
    "   \n\n");

PyDoc_STRVAR(
    initContrast_doc,
    "Initialize the thresholds (log units) of each pixel\n\n"
    "Parameters\n"
    "----------\n"
    "th_pos: numpy_array\n"
    "   Positive threshold\n"
    "th_neg: numpy_array\n"
    "   Negative threshold\n"
    "th_noise: numpy_array\n"
    "   Noise threshold\n"
    "Returns\n"
    "----------\n"
    "result: bool (sucess)\n"
    "   \n\n");

PyDoc_STRVAR(
    setDebug_doc,
    "Activate/Desactivate debug flags\n\n"
    "Parameters\n"
    "----------\n"
    "debug: bool numpy_array\n"
    "   Display or not\n"
    "Returns\n"
    "----------\n"
    "result: bool (sucess)\n"
    "   \n\n");


PyDoc_STRVAR(
    initLatency_doc,
    "Initialize the latency.\n\n"
    "Parameters\n"
    "----------\n"
    "lat: numpy_array\n"
    "   Latency (us)\n"
    "jit: numpy_array\n"
    "   Jitter (us)\n"
    "ref: numpy_array\n"
    "   Refactory period (us)\n"
    "tau: numpy_array\n"
    "   First stage time constant (us)\n"
    "Returns\n"
    "----------\n"
    "result: bool (sucess)\n"
    "   \n\n");

PyDoc_STRVAR(
    initNoise_doc,
    "Initialize the Noise of the sensor based on real sensor noise distributions.\n\n"
    "Parameters\n"
    "----------\n"
    "pos_dist: numpy_array\n"
    "   n * 72 array corresponding to n different cumulative distribution functions of the positive noise\n"
    "neg_dist: numpy_array\n"
    "   n * 72 array corresponding to n different cumulative distribution functions of the negative noise\n"
    "Returns\n"
    "----------\n"
    "result: bool (sucess)\n"
    "   \n\n");

PyDoc_STRVAR(
    initImg_doc,
    "Initialize the simulator with the first image.\n\n"
    "Parameters\n"
    "----------\n"
    "img: numpy_array\n"
    "   y x image (L component of Lab)\n"
    "Returns\n"
    "----------\n"
    "result: bool (sucess)\n"
    "   \n\n");

PyDoc_STRVAR(
    updateImg_doc,
    "Update the simulator with a next image simulated after dt.\n\n"
    "Parameters\n"
    "----------\n"
    "img: numpy_array\n"
    "   y x image (L component of Lab)\n"
    "dt: numpy_array\n"
    "   time between the last image\n"
    "Returns\n"
    "----------\n"
    "result: Dict of events, {'ts', 'x', 'y', 'p'}\n"
    "   \n\n");

PyDoc_STRVAR(
    getCurv_doc,
    "Return the current potential of the pixels.\n\n"
    "Parameters\n"
    "----------\n"
    "Returns\n"
    "----------\n"
    "result: numpy_array containing cur_v_\n"
    "  \n\n");

PyDoc_STRVAR(
    getShape_doc,
    "Return the size of the imager.\n\n"
    "Parameters\n"
    "----------\n"
    "Returns\n"
    "----------\n"
    "result: numpy_array of the shape\n"
    "   \n\n");


PyDoc_STRVAR(
    masterRst_doc,
    "Reset alll the pixels Without creating events.\n\n"
    "Parameters\n"
    "----------\n"
    "Returns\n"
    "----------\n"
    "   \n\n");

//Module specification
static PyMethodDef module_methods[] = {
    {"initSimu", (PyCFunction)initSimu, METH_VARARGS, initSimu_doc},
    {"setDebug", (PyCFunction)setDebug, METH_VARARGS, setDebug_doc},
    {"initContrast", (PyCFunction)initContrast, METH_VARARGS, initContrast_doc},
    {"initLatency", (PyCFunction)initLatency, METH_VARARGS, initLatency_doc},
    {"initNoise", (PyCFunction)initNoise, METH_VARARGS, initNoise_doc},
    {"initImg", (PyCFunction)initImg, METH_VARARGS, initImg_doc},
    {"updateImg", (PyCFunction)updateImg, METH_VARARGS, updateImg_doc},
    {"getShape", (PyCFunction)getShape, METH_VARARGS, getShape_doc},
    {"getCurv", (PyCFunction)getCurv, METH_VARARGS, getCurv_doc},
    {"masterRst", (PyCFunction)masterRst, METH_VARARGS, masterRst_doc},
    {"disableNoise", (PyCFunction)disableNoise, METH_VARARGS, disableNoise_doc},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef dsi_def = {
    PyModuleDef_HEAD_INIT,
    "dsi",
    module_docstring,
    -1,
    module_methods};

//Initialize module
PyMODINIT_FUNC
PyInit_dsi(void)
{
    PyObject *m = PyModule_Create(&dsi_def);
    if (m == NULL)
        return NULL;
    //numpy functionallity
    import_array();
    return m;
}