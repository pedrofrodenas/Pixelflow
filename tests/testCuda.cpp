#include "pixelflow/Pixelflow.h"
#include <iostream>

#include "pixelflow/core/Image.h"

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


    cout << b.NumElements() << endl;
    cout << b.size() << endl;

    cout << a.ToString() << endl;

    cout << a.GetDims() << endl;

    bool es = pixelflow::core::IsBroadcastable(a, b);
    cout << "Es broadcastable: " << es << endl;

    pixelflow::core::ShapeArray d = pixelflow::core::ExpandDims(a, 6);

    cout << d.ToString() << endl;

    pixelflow::core::ShapeArray e(5);

    pixelflow::core::ShapeArray f = pixelflow::core::BroadcastShape(aa, bb);

    cout << f.ToString() << endl;

    std::vector<std::string> elem = pixelflow::utility::SplitString("CUDA:0", ":");

    cout << elem[0] << endl;

    std::vector<pixelflow::core::Device> res = pixelflow::core::Device::GetAvailableDevices();

    pixelflow::core::Device::PrintAvailableDevices();

    pixelflow::core::ShapeArray strides = pixelflow::core::DefaultStrides({5, 5, 3});

    cout << strides.ToString() << endl;

    pixelflow::core::Image img({5,4,3}, pixelflow::core::Float32, pixelflow::core::Device("CPU:0"));

    // Define a Image with initial values
    pixelflow::core::Image img1(vector<int>{3,4,6}, {3}, pixelflow::core::PfType::Int32,
        pixelflow::core::Device("CPU:0"));

    cout << "Num dims: " << img.NumDims() << " Num Elements: " << img.NumElements() << endl;

    for (auto it = img.begin(); it != img.end(); ++it) {
        cout << "Hola" << endl;
    }

    pixelflow::core::Image img2 = img.Slice(0, 1, 5, 2);

    cout << "img contiguous?: " << img.IsContiguous() << " img2 contiguous?: " << img2.IsContiguous() << endl;

    cout << "Image shape at dim 1: " << img.GetShape(1) << endl;

    cout << "Image stride at dim 0: " << img.GetStride(0) << endl;

    pixelflow::core::ShapeArray g = {1, 0, 2};

    int64_t ndims_ = g.GetDims();
    bool conditio = std::all_of(g.begin(), g.end(),
                                    [ndims_](int64_t d) { return d <= (ndims_-1); });

    cout << "Ndims: " << ndims_ << endl;
    cout << conditio << endl;

    pixelflow::core::ImageRef ref(img);
    ref.Permute(pixelflow::core::ShapeArray {1,0,2});

    cout << "img shape: " << img.Shape().ToString() << endl;
    cout << "Is ref contiguous: "<< ref.IsContiguous() << endl;

    pixelflow::core::ImageRef ref2(img);

    cout << "Are equal ref and ref2 ?: " << (ref == ref2) << endl;

    return 0;
}