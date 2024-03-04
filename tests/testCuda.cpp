#include "pixelflow/Pixelflow.h"
#include <iostream>

using namespace std;

int main() {

    int device = pixelflow::core::DeviceCount();

    cout << device << endl;

    pixelflow::core::PfType dtype = pixelflow::core::Float32;

    cout << dtype.getSize() << endl;

    pixelflow::core::ImgSize a = {3, 512, 512};
    pixelflow::core::ImgSize b;

    

    cout << a.back() << endl;

    cout << b.NumElems() << endl;
    cout << b.GetDims() << endl;

    cout << a.Shape() << endl;
    return 0;
}