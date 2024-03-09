#include "pixelflow/Pixelflow.h"
#include <iostream>

using namespace std;

int main() {

    int device = pixelflow::core::cuda::DeviceCount();

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

    std::vector<std::string> elem = pixelflow::utility::SplitString("CUDA:0", ":");

    cout << elem[0] << endl;

    pixelflow::core::Device myDevice("CUDA:3");

    std::vector<pixelflow::core::Device> res = pixelflow::core::Device::GetAvailableDevices();

    cout << "Devices: " << endl;
    for (auto it = res.begin(); it != res.end(); ++it) {
        cout << it->IsCUDA() << endl;
    }

    return 0;
}