// call_function.cpp - A sample of calling 
// python functions from C code
// 

#include <python2.7/Python.h>

int main()
{
    PyObject *pName, *pModule, *pDict, *pFunc, *pValue;

    // Initialize the Python Interpreter
    Py_Initialize();

    // Build the name object
    pName = PyString_FromString("py_function");

    // Load the module object
    pModule = PyImport_Import(pName);

	if (pModule == NULL){
		fprintf(stderr, "Failed to load \"%s\"\n", "py_function");
	}

    // pDict is a borrowed reference 
    pDict = PyModule_GetDict(pModule);

    // pFunc is also a borrowed reference 
    pFunc = PyDict_GetItemString(pDict, "multiply");

    if (PyCallable_Check(pFunc)) 
    {
        PyObject_CallObject(pFunc, NULL);
    } else 
    {
        PyErr_Print();
    }

    // Clean up
    Py_DECREF(pModule);
    Py_DECREF(pName);

    // Finish the Python Interpreter
    Py_Finalize();

    return 0;
}
