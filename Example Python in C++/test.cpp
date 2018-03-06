#include <python2.7/Python.h>

int main() {
	Py_Initialize();
	// Import the module
	PyObject* myModuleString = PyString_FromString((char*)"mytest");
	PyObject* myModule = PyImport_Import(myModuleString);
	// Getting a reference to your function
	PyObject* myFunction = PyObject_GetAttrString(myModule,(char*)"myabs");
	PyObject* args = PyTuple_Pack(1,PyFloat_FromDouble(2.0));
	// Getting your result
	PyObject* myResult = PyObject_CallObject(myFunction, args);
	Py_Finalize();

    return 0;
}
