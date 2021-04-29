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
#include "simu.hpp" //this is the function that we want to call
//Docstring
static char module_docstring[] = "Example module providing a function to add two numbers";

PyDoc_STRVAR(
    add_doc,
    "Add two numbers or arrays.\n\n"
    "Parameters\n"
    "----------\n"
    "a: numpy_array\n"
    "   or single number\n"
    "b: numpy_array\n"
    "   or single number\n"
    "Returns\n"
    "----------\n"
    "result: numpy_array\n"
    "   added numbers\n\n");

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

std::unique_ptr<SimuICNS> mySimu;

//function declaration
static PyObject *add(PyObject *self, PyObject *args);
static PyObject *initSimu(PyObject *self, PyObject *args);
static PyObject *setDebug(PyObject *self, PyObject *args);
static PyObject *initContrast(PyObject *self, PyObject *args);
static PyObject *initLatency(PyObject *self, PyObject *args);
static PyObject *initNoise(PyObject *self, PyObject *args);
static PyObject *initImg(PyObject *self, PyObject *args);
static PyObject *updateImg(PyObject *self, PyObject *args);
static PyObject *getShape(PyObject *self);
static PyObject *disableNoise(PyObject *self);
static PyObject *getCurv(PyObject *self);

//Module specification
static PyMethodDef module_methods[] = {
    {"add", (PyCFunction)add, METH_VARARGS, add_doc},
    {"initSimu", (PyCFunction)initSimu, METH_VARARGS, initSimu_doc},
    {"setDebug", (PyCFunction)setDebug, METH_VARARGS, setDebug_doc},
    {"initContrast", (PyCFunction)initContrast, METH_VARARGS, initContrast_doc},
    {"initLatency", (PyCFunction)initLatency, METH_VARARGS, initLatency_doc},
    {"initNoise", (PyCFunction)initNoise, METH_VARARGS, initNoise_doc},
    {"initImg", (PyCFunction)initImg, METH_VARARGS, initImg_doc},
    {"updateImg", (PyCFunction)updateImg, METH_VARARGS, updateImg_doc},
    {"getShape", (PyCFunction)getShape, METH_VARARGS, getShape_doc},
    {"getCurv", (PyCFunction)getCurv, METH_VARARGS, getCurv_doc},
    {"disableNoise", (PyCFunction)disableNoise, METH_VARARGS, disableNoise_doc},
    {NULL, NULL, 0, NULL}};

static PyObject * initSimu(PyObject *self, PyObject *args){
    PyObject *a_obj;
    PyObject *b_obj;
    if (!PyArg_ParseTuple(args, "OO", &a_obj, &b_obj))
        return NULL;
    PyObject *a_array = PyArray_FROM_OTF(a_obj, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS);
    PyObject *b_array = PyArray_FROM_OTF(b_obj, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS);
    if (a_array == NULL || b_array == NULL){
        PyErr_SetString(PyExc_TypeError, "Could not convert argument to numpy array.");
        return NULL;
    }
    auto a_shape = PyArray_SHAPE(reinterpret_cast<PyArrayObject *>(a_array));
    auto b_shape = PyArray_SHAPE(reinterpret_cast<PyArrayObject *>(b_array));
    auto ndim = PyArray_NDIM(reinterpret_cast<PyArrayObject *>(a_array)) * PyArray_NDIM(reinterpret_cast<PyArrayObject *>(a_array));
    if((a_shape[0] != 1)|(b_shape[0] != 1)){
        PyErr_SetString(PyExc_ValueError, "The size of the arguments must be equal to one.");
        return NULL;
    }
    if (ndim > 1){
        PyErr_SetString(PyExc_ValueError, "The Dimenssion must be one.");
    }
    const double *a = (double *)PyArray_DATA((PyArrayObject *)a_array);
    const double *b = (double *)PyArray_DATA((PyArrayObject *)b_array);
    mySimu =  std::make_unique<SimuICNS>(*a, *b);
    Py_DECREF(a_array);
    Py_DECREF(b_array);
    std::cout<<"Simu init"<<std::endl;
    return Py_True;
}

static PyObject * getShape(PyObject *self){
    int ndim = 1;
    const long int shape[1] = {2};
    PyObject *result_array = PyArray_SimpleNew(ndim, shape, NPY_DOUBLE);
    double *result = (double *)PyArray_DATA((PyArrayObject *)result_array);
    result[0] = mySimu->x_;
    result[1] = mySimu->y_;
    std::cout<< "getShape" << std::endl;
    return result_array;
}

static PyObject * disableNoise(PyObject *self){
    mySimu->disableNoise();
    std::cout<< "Noise Disabled" << std::endl;
    return Py_True;
}

static PyObject * getCurv(PyObject *self){
    int ndim = 2;
    const long int shape[2] = {static_cast<long int>(mySimu->x_), static_cast<long int>(mySimu->y_)};
    PyObject *result_array = PyArray_SimpleNew(ndim, shape, NPY_DOUBLE);

    std::memcpy(PyArray_DATA((PyArrayObject*) result_array ),  mySimu->cur_v_.data(), sizeof(double) *  mySimu->x_ * mySimu->y_);
    return result_array;
}

static PyObject * initContrast(PyObject *self, PyObject *args){
    PyObject *c_pos_arg;
    PyObject *c_neg_arg;
    PyObject *c_noise_arg;
    if (!PyArg_ParseTuple(args, "OOO", &c_pos_arg, &c_neg_arg, &c_noise_arg))
        return NULL;
    PyObject *c_pos_array = PyArray_FROM_OTF(c_pos_arg, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS);
    PyObject *c_neg_array = PyArray_FROM_OTF(c_neg_arg, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS);
    PyObject *c_noise_array = PyArray_FROM_OTF(c_noise_arg, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS);
    if (c_pos_array == NULL || c_noise_array == NULL || c_noise_array == NULL){
        PyErr_SetString(PyExc_TypeError, "Could not convert argument to numpy array.");
        return NULL;
    }
    auto c_pos_shape = PyArray_SHAPE(reinterpret_cast<PyArrayObject *>(c_pos_array));
    auto c_neg_shape = PyArray_SHAPE(reinterpret_cast<PyArrayObject *>(c_neg_array));
    auto c_noise_shape = PyArray_SHAPE(reinterpret_cast<PyArrayObject *>(c_noise_array));
    auto ndim = PyArray_NDIM(reinterpret_cast<PyArrayObject *>(c_pos_array)) * PyArray_NDIM(reinterpret_cast<PyArrayObject *>(c_neg_array)) * PyArray_NDIM(reinterpret_cast<PyArrayObject *>(c_noise_array));
    if(c_pos_shape[0] != 1 || c_neg_shape[0] != 1 || c_noise_shape[0] != 1){
        PyErr_SetString(PyExc_ValueError, "The size of the arguments must be equal to one.");
        return NULL;
    }
    if (ndim > 1){
        PyErr_SetString(PyExc_ValueError, "The Dimenssion must be one.");
    }
    const double *c_pos = (double *)PyArray_DATA((PyArrayObject *)c_pos_array);
    const double *c_neg = (double *)PyArray_DATA((PyArrayObject *)c_neg_array);
    const double *c_noise = (double *)PyArray_DATA((PyArrayObject *)c_noise_array);
    mySimu->set_th(*c_pos, *c_neg, *c_noise);
    Py_DECREF(c_pos_array);
    Py_DECREF(c_neg_array);
    Py_DECREF(c_noise_array);
    std::cout<<"Contrast Init"<<std::endl;
    return Py_True;
}

static PyObject * setDebug(PyObject *self, PyObject *args){
    PyObject *debug_arg;
    if (!PyArg_ParseTuple(args, "O", &debug_arg))
        return NULL;
    PyObject *debug_array = PyArray_FROM_OTF(debug_arg, NPY_BOOL, NPY_ARRAY_C_CONTIGUOUS);

    if (debug_arg == NULL){
        PyErr_SetString(PyExc_TypeError, "Could not convert argument to numpy array.");
        return NULL;
    }
    auto debug_shape = PyArray_SHAPE(reinterpret_cast<PyArrayObject *>(debug_array));
    auto ndim = PyArray_NDIM(reinterpret_cast<PyArrayObject *>(debug_array));
    if(debug_shape[0] != 1 ){
        PyErr_SetString(PyExc_ValueError, "The size of the arguments must be equal to one.");
        return NULL;
    }
    if (ndim > 1){
        PyErr_SetString(PyExc_ValueError, "The Dimenssion must be one.");
    }
    const bool *debug = (bool *)PyArray_DATA((PyArrayObject *)debug_array);
    mySimu->debug_ = debug[0];
    Py_DECREF(debug_array);
    std::cout<<"Debug set"<<std::endl;
    return Py_True;
}

static PyObject * initLatency(PyObject *self, PyObject *args){
    PyObject *lat_arg;
    PyObject *jit_arg;
    PyObject *ref_arg;
    PyObject *tau_arg;
    if (!PyArg_ParseTuple(args, "OOOO", &lat_arg, &jit_arg, &ref_arg, &tau_arg))
        return NULL;
    PyObject *lat_array = PyArray_FROM_OTF(lat_arg, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS);
    PyObject *jit_array = PyArray_FROM_OTF(jit_arg, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS);
    PyObject *ref_array = PyArray_FROM_OTF(ref_arg, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS);
    PyObject *tau_array = PyArray_FROM_OTF(tau_arg, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS);
    if (lat_array == NULL || jit_array == NULL || ref_array == NULL || tau_array == NULL){
        PyErr_SetString(PyExc_TypeError, "Could not convert argument to numpy array.");
        return NULL;
    }
    auto lat_shape = PyArray_SHAPE(reinterpret_cast<PyArrayObject *>(lat_array));
    auto jit_shape = PyArray_SHAPE(reinterpret_cast<PyArrayObject *>(jit_array));
    auto ref_shape = PyArray_SHAPE(reinterpret_cast<PyArrayObject *>(ref_array));
    auto tau_shape = PyArray_SHAPE(reinterpret_cast<PyArrayObject *>(tau_array));
    auto ndim = PyArray_NDIM(reinterpret_cast<PyArrayObject *>(lat_array)) * PyArray_NDIM(reinterpret_cast<PyArrayObject *>(jit_array)) * PyArray_NDIM(reinterpret_cast<PyArrayObject *>(ref_array)) * PyArray_NDIM(reinterpret_cast<PyArrayObject *>(tau_array));
    if(lat_shape[0] != 1 || jit_shape[0] != 1 || ref_shape[0] != 1 || tau_shape[0] != 1){
        PyErr_SetString(PyExc_ValueError, "The size of the arguments must be equal to one.");
        return NULL;
    }
    if (ndim > 1){
        PyErr_SetString(PyExc_ValueError, "The Dimension must be one.");
    }
    const double *lat = (double *)PyArray_DATA((PyArrayObject *)lat_array);
    const double *jit = (double *)PyArray_DATA((PyArrayObject *)jit_array);
    const double *ref = (double *)PyArray_DATA((PyArrayObject *)ref_array);
    const double *tau = (double *)PyArray_DATA((PyArrayObject *)tau_array);
    mySimu->set_lat(*lat, *jit, *ref, *tau);
    Py_DECREF(lat_array);
    Py_DECREF(jit_array);
    Py_DECREF(ref_array);
    Py_DECREF(tau_array);
    std::cout<<"Latency initialized"<<std::endl;
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
    std::cout<<"Noise Init"<<std::endl;
    return Py_True;
}

static PyObject * updateImg(PyObject *self, PyObject *args){
    PyObject *img_arg;
    PyObject *time_arg;
    if (!PyArg_ParseTuple(args, "OO", &img_arg, &time_arg))
        return NULL;
    PyObject *img_array = PyArray_FROM_OTF(img_arg, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS);
    PyObject *time_array = PyArray_FROM_OTF(time_arg, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS);
    if (img_array == NULL || time_array == NULL){
        PyErr_SetString(PyExc_TypeError, "Could not convert argument to numpy array.");
        return NULL;
    }
    auto img_shape = PyArray_SHAPE(reinterpret_cast<PyArrayObject *>(img_array));
    auto time_shape = PyArray_SHAPE(reinterpret_cast<PyArrayObject *>(time_array));
    if(img_shape[1] != mySimu->y_ || img_shape[0] != mySimu->x_ || time_shape[0] != 1){
        std::unique_ptr<char[]> buf( new char[ 100 ] );
        sprintf(buf.get(), "The Dimension of the image must be %d x %d and time is a one value array\n", mySimu->y_, mySimu->x_);
        PyErr_SetString(PyExc_ValueError, buf.get());
        return NULL;
    }
    const double *img = (double *)PyArray_DATA((PyArrayObject *)img_array);
    const double *time = (double *)PyArray_DATA((PyArrayObject *)time_array);
    std::vector<Event> ev;
    mySimu->update_img(img, (uint32_t)*time, ev);
    std::sort(ev.begin(), ev.end(), [](const Event & a, const Event & b) -> bool{ return a.ts_ < b.ts_;});
    auto stream = PyDict_New();
    const long int shape[1] = {static_cast<long int>(ev.size())};
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
    Py_DECREF(time_array);
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
    if(img_shape[1] != mySimu->y_ || img_shape[0] != mySimu->x_){
        std::unique_ptr<char[]> buf( new char[ 100 ] );
        sprintf(buf.get(), "The shape of the image (%d, %d) is not the shape of the simulator (%d, %d) .\n", static_cast<int>(img_shape[1]), static_cast<int>(img_shape[0]), mySimu->y_, mySimu->x_);
        PyErr_SetString(PyExc_ValueError, buf.get());
        return NULL;
    }
    const double *img = (double *)PyArray_DATA((PyArrayObject *)img_array);
    mySimu->init_img(img);
    Py_DECREF(img_array);
    std::cout<<"Image Init"<<std::endl;
    return Py_True;
}

static PyObject *add(PyObject *self, PyObject *args)
{
    //PyObjects that should be parsed from args
    PyObject *a_obj;
    PyObject *b_obj;

    //Check and parse..
    if (!PyArg_ParseTuple(args, "OO", &a_obj, &b_obj))
        return NULL;

    //Numpy array from the parsed objects
    //Yes you could check for type etc. but here we just convert to double
    PyObject *a_array = PyArray_FROM_OTF(a_obj, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS);
    PyObject *b_array = PyArray_FROM_OTF(b_obj, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS);

    //If parsing of a or b fails we throw an exception in Python
    if (a_array == NULL || b_array == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "Could not convert argument to numpy array.");
        return NULL;
    }

    //Dimensions should agree
    if (PyArray_NDIM((PyArrayObject *)a_array) != PyArray_NDIM((PyArrayObject *)b_array))
    {
        PyErr_SetString(PyExc_ValueError, "The size of the arguments need to match.");
        return NULL;
    }

    auto ndim = PyArray_NDIM(reinterpret_cast<PyArrayObject *>(a_array));
    auto a_shape = PyArray_SHAPE(reinterpret_cast<PyArrayObject *>(a_array));
    auto b_shape = PyArray_SHAPE(reinterpret_cast<PyArrayObject *>(b_array));

    for (int i = 0; i != ndim; ++i)
    {
        if (a_shape[i] != b_shape[i])
        {
            PyErr_SetString(PyExc_ValueError, "The shape of the arguments need to match.");
            return NULL;
        }
    }

    //Create array for return values
    PyObject *result_array = PyArray_SimpleNew(ndim, a_shape, NPY_DOUBLE);

    // Get a pointer to the data for our function call
    // I don't recommend this style but lets at least make it const =)
    const double *a = (double *)PyArray_DATA((PyArrayObject *)a_array);
    const double *b = (double *)PyArray_DATA((PyArrayObject *)b_array);

    //And a pointer to the resutls
    double *result = (double *)PyArray_DATA((PyArrayObject *)result_array);

    //Now call add wih pointers and size
    auto size = PyArray_Size(a_array);

    for (int i=0; i!=size; ++i){
        result[i] = a[i] + b[i];
    }

    //Clean up
    Py_DECREF(a_array);
    Py_DECREF(b_array);

    return result_array;
}


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