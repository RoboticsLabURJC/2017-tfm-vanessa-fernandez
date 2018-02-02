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
