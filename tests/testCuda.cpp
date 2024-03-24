#include "pixelflow/Pixelflow.h"
#include <iostream>

#include "pixelflow/core/Image.h"

using namespace std;

int main() {

    using namespace pixelflow::core;

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

    pixelflow::core::Image img({5,2,3}, pixelflow::core::Float32, pixelflow::core::Device("CPU:0"));

    // Define a Image with initial values
    pixelflow::core::Image img1(vector<int>{3,4,6}, {3}, pixelflow::core::PfType::Int32,
        pixelflow::core::Device("CPU:0"));

    cout << "Num dims: " << img.NumDims() << " Num Elements: " << img.NumElements() << endl;

    for (auto it = img.begin(); it != img.end(); ++it) {
        cout << "Hola" << endl;
    }

    pixelflow::core::ShapeArray g = {1, 0, 2};

    pixelflow::core::ImageRef ref(img);
    ref.Permute(pixelflow::core::ShapeArray {1,0,2});

    cout << "img shape: " << img.Shape().ToString() << endl;
    cout << "Is ref contiguous: "<< ref.IsContiguous() << endl;

    pixelflow::core::ImageRef ref2(img);

    pixelflow::core::ShapeArray reducted = pixelflow::core::ReductionShape(aa, {0, 2}, true);

    cout << reducted.ToString() << endl;

    Image imga({3, 10, 4, 5}, PfType::Float32, Device("CPU:0"));
    Image imgb({3, 10, 4, 5}, PfType::Float32, Device("CPU:0"));
    Image imgc({1, 1, 4, 5}, PfType::Float32, Device("CPU:0"));

    cout << "imga stride" << imga.Stride().ToString() << endl;
    cout << "imgc stride" << imgc.Stride().ToString() << endl;

    vector<Image> vimg = {imga};
    vector<Image> vimg2 = {imgc};

    Indexer cosita(vimg, vimg2, DtypePolicy::ALL_SAME);

    pixelflow::core::Image img2 = img.Slice(0, 1, 5, 2);

    cout << img2.Stride().ToString() << endl;
    cout << img2.Shape().ToString() << endl;
    cout << img2.IsContiguous() << endl;
    Image img3 = img2.Contiguous();

    Image img_test(vector<int>{0, 1, 2, 3, 4, 5, 6, 7, 8}, {3, 3}, pixelflow::core::PfType::Int32,
            pixelflow::core::Device("CPU:0"));

    cout << "img_test shape: " << img_test.Shape().ToString() << endl;
    cout << "img_test stride: " << img_test.Stride().ToString() << endl;


    Image img_test2 = img_test.IndexExtract(2, 0);
    cout << "img_test2 Shape: " << img_test2.Shape().ToString() << endl;
    cout << "NumDims: " << img_test.NumDims() << endl;

    cout << img_test2.ToString() << endl;


    return 0;
}