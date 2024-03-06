#include "pixelflow/Pixelflow.h"
#include <iostream>

using namespace std;

int main() {

    int device = pixelflow::core::DeviceCount();

    cout << device << endl;

    pixelflow::core::PfType dtype = pixelflow::core::Float32;

    cout << dtype.getSize() << endl;

    pixelflow::core::ShapeArray a = {7, 512, 512, 6};
    pixelflow::core::ShapeArray b;

    cout << a.back() << endl;

    cout << b.NumElems() << endl;
    cout << b.size() << endl;

    cout << a.Shape() << endl;

    cout << a.GetDims() << endl;
    return 0;
}