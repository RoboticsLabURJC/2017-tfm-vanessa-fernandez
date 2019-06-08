---
layout: default
---
# Week 10: Embedding Python in C++, SSD

This week, I searched for information on embedding Python code in C++. You can find information in the following links: [1](https://docs.python.org/2/extending/embedding.html, https://www.codeproject.com/Articles/11805/Embedding-Python-in-C-C-Part-I), [2](https://realmike.org/blog/2012/07/05/supercharging-c-code-with-embedded-python/), [3](https://skebanga.github.io/embedded-python-pybind11/. I have made a simple example (hello.cpp)): 

<pre>
#include <stdio.h>
#include <python3.5/Python.h>

int main()
{
	PyObject* pInt;

	Py_Initialize();

	PyRun_SimpleString("print('Hello World from Embedded Python!!!')");
	
	Py_Finalize();

	printf("\nPress any key to exit...\n");
}
</pre>

It is possible to have some compilation errors if you use: 

<pre>
gcc hello.cpp -o hello
</pre>

That's why I compiled in the following way: 

<pre>
gcc hello.cpp -o hello -L/usr/lib/python2.7/config/ -lpython2.7
</pre>

The result is: 

<pre>
Hello World from Embedded Python!!!

Press any key to exit...
</pre>

I have also made another more complex example:

* C++ code:

<pre>
#include <python2.7/Python.h>

int
main(int argc, char *argv[])
{
    PyObject *pName, *pModule, *pDict, *pFunc;
    PyObject *pArgs, *pValue;
    int i;

    if (argc < 3) {
        fprintf(stderr,"Usage: call pythonfile funcname [args]\n");
        return 1;
    }

    Py_Initialize();
    pName = PyString_FromString(argv[1]);
    /* Error checking of pName left out */

    pModule = PyImport_Import(pName);
    Py_DECREF(pName);

    if (pModule != NULL) {
        pFunc = PyObject_GetAttrString(pModule, argv[2]);
        /* pFunc is a new reference */

        if (pFunc && PyCallable_Check(pFunc)) {
            pArgs = PyTuple_New(argc - 3);
            for (i = 0; i < argc - 3; ++i) {
                pValue = PyInt_FromLong(atoi(argv[i + 3]));
                if (!pValue) {
                    Py_DECREF(pArgs);
                    Py_DECREF(pModule);
                    fprintf(stderr, "Cannot convert argument\n");
                    return 1;
                }
                /* pValue reference stolen here: */
                PyTuple_SetItem(pArgs, i, pValue);
            }
            pValue = PyObject_CallObject(pFunc, pArgs);
            Py_DECREF(pArgs);
            if (pValue != NULL) {
                printf("Result of call: %ld\n", PyInt_AsLong(pValue));
                Py_DECREF(pValue);
            }
            else {
                Py_DECREF(pFunc);
                Py_DECREF(pModule);
                PyErr_Print();
                fprintf(stderr,"Call failed\n");
                return 1;
            }
        }
        else {
            if (PyErr_Occurred())
                PyErr_Print();
            fprintf(stderr, "Cannot find function \"%s\"\n", argv[2]);
        }
        Py_XDECREF(pFunc);
        Py_DECREF(pModule);
    }
    else {
        PyErr_Print();
        fprintf(stderr, "Failed to load \"%s\"\n", argv[1]);
        return 1;
    }
    Py_Finalize();
    return 0;
}
</pre>


* Python code:

<pre>
def multiply(a,b):
    print "Will compute", a, "times", b
    c = 0
    for i in range(0, a):
        c = c + b
    return c
</pre>

I have compiled the code by putting in the terminal: 

<pre>
gcc multiply.cpp -o multiply -L/usr/lib/python2.7/config/ -lpython2.7
</pre>

To execute: 

<pre>
./multiply multiply multiply 3 4
</pre>

It is possible to have some errors. If you have errors, like ImportError: No module named multiply, you have to use: 

<pre>
PYTHONPATH=. ./multiply multiply multiply 3 4
</pre>

And the result is: 

<pre>
Will compute 3 times 4
Result of call: 12
</pre>

Also, this week I made a simple SSD code to detect cars based on the one that implements in [1](https://github.com/ksketo/CarND-Vehicle-Detection). This code can be found at [2](https://github.com/RoboticsURJC-students/2017-tfm-vanessa-fernandez/tree/master/Simple%20net%20Object%20detection/car_detection). To detect the cars we execute the code detectionImages.py. The result is as follows:

![car_ssd](http://jderobot.org/store/vmartinezf/uploads/images/car_ssd.png)

 Also, I made a simple SSD to detect persons. The result is as follows: 
 
![person_ssd](http://jderobot.org/store/vmartinezf/uploads/images/person_ssd.png)

