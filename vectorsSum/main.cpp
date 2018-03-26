#include <iostream>
#include "vectorsum.h"

using namespace std;

int main()
{
    int numElements = 50000;
    int ret = vectorSum(numElements);
    if (ret == 0)
        cout << "Successful vector sum test." << endl;
    else
        cout << "Test error." << endl;

    return 0;
}

