#include "pixelflow/Pixelflow.h"
#include <iostream>

using namespace std;

int main() {

    int device = pixelflow::core::DeviceCount();

    cout << device << endl;

    pixelflow::core::PfType dtype = pixelflow::core::Float32;

    cout << dtype.getSize() << endl;

    pixelflow::core::ShapeArray a = {3, 1, 2};
    pixelflow::core::ShapeArray b =    {5, 1};

    pixelflow::core::ShapeArray aa = {100, 200, 1, 2};
    pixelflow::core::ShapeArray bb =  {100, 200, 4,2};


    cout << b.NumElems() << endl;
    cout << b.size() << endl;

    cout << a.Shape() << endl;

    cout << a.GetDims() << endl;

    bool es = pixelflow::core::IsBroadcastable(a, b);
    cout << "Es broadcastable: " << es << endl;

    pixelflow::core::ShapeArray d = pixelflow::core::ExpandDims(a, 6);

    cout << d.Shape() << endl;

    pixelflow::core::ShapeArray e(5);

    pixelflow::core::ShapeArray f = pixelflow::core::BroadcastShape(aa, bb);

    cout << f.Shape() << endl;

    return 0;
}